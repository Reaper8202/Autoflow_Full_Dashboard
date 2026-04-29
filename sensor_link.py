"""Sensor connection layer for the uroflow load cell.

Supports two transport modes:
  1. USB serial streaming
  2. BLE notify streaming from the Xiao nRF52840 device used by AutoFlow

Observed packet formats:
  - Text:   "[timestamp_ms, raw_value]\n"
  - Binary: 8 bytes = [int32_le raw_value, int32_le timestamp_centiseconds]
"""

from __future__ import annotations

import asyncio
import atexit
import platform
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

import serial
import serial.tools.list_ports

try:
    from bleak import BleakClient, BleakScanner
except Exception:  # pragma: no cover - optional dependency at runtime
    BleakClient = None
    BleakScanner = None

BLE_AVAILABLE = BleakClient is not None and BleakScanner is not None


DEFAULT_CALIBRATION_FACTOR = 0.00052587
XIAO_SERVICE_UUID = "eb517b5b-2f71-474f-9559-21f6d95f81c5"
XIAO_NOTIFY_UUID = "b7d28264-88a2-4c19-97b1-946d04fbed59"
BLE_PREFIX = "BLE::"


@dataclass
class BleDeviceInfo:
    name: str
    address: str
    rssi: Optional[int] = None
    service_uuids: Optional[List[str]] = None

    @property
    def label(self) -> str:
        name = self.name or "Unknown BLE device"
        rssi = f" RSSI={self.rssi}" if self.rssi is not None else ""
        return f"{name} ({self.address}){rssi}"

    @property
    def token(self) -> str:
        return f"{BLE_PREFIX}{self.address}::{self.name or ''}"



def list_sensor_serial_ports() -> List[str]:
    return [p.device for p in serial.tools.list_ports.comports()]


async def _discover_ble_devices_async(timeout: float = 6.0) -> List[BleDeviceInfo]:
    if BleakScanner is None:
        return []

    try:
        devices_and_adv = await BleakScanner.discover(timeout=timeout, return_adv=True)
    except TypeError:
        devices = await BleakScanner.discover(timeout=timeout)
        infos = []
        for device in devices:
            infos.append(
                BleDeviceInfo(
                    name=device.name or "",
                    address=device.address,
                    rssi=getattr(device, "rssi", None),
                    service_uuids=[],
                )
            )
        return infos

    infos = []
    seen_addresses = set()
    for _, payload in devices_and_adv.items():
        if isinstance(payload, tuple) and len(payload) == 2:
            device, adv = payload
        else:
            device = payload
            adv = None

        if not getattr(device, "address", None):
            continue
        if device.address in seen_addresses:
            continue
        seen_addresses.add(device.address)

        uuids = []
        if adv is not None:
            uuids = [u.lower() for u in (getattr(adv, "service_uuids", None) or [])]

        infos.append(
            BleDeviceInfo(
                name=device.name or "",
                address=device.address,
                rssi=getattr(device, "rssi", None),
                service_uuids=uuids,
            )
        )

    infos.sort(
        key=lambda d: (
            XIAO_SERVICE_UUID not in (d.service_uuids or []),
            not bool(d.name),
            d.name.lower() if d.name else "",
            d.address,
        )
    )
    return infos


def discover_ble_devices(timeout: float = 6.0) -> List[BleDeviceInfo]:
    if BleakScanner is None:
        raise RuntimeError("BLE requires the 'bleak' package")
    try:
        return asyncio.run(_discover_ble_devices_async(timeout=timeout))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_discover_ble_devices_async(timeout=timeout))
        finally:
            loop.close()
    except Exception as e:
        raise RuntimeError(f"BLE scan failed: {e}") from e


class SensorLink:
    def __init__(self):
        self.ser = None
        self.port = None
        self.mode = None  # "serial" | "ble" | None

        self.calibration_factor = DEFAULT_CALIBRATION_FACTOR
        self.tare_offset = 0.0

        self._lock = threading.Lock()
        self._timestamps = []
        self._masses = []
        self._raw_values = []
        self._time_offset = None
        self._collecting = False
        self._reader_thread = None
        self._stop_event = threading.Event()

        self._last_live_mass = 0.0
        self._last_live_raw = 0.0
        self._recent_calibrated = deque(maxlen=60)
        self._status = "Offline"
        self._last_error = ""

        self._ble_loop = None
        self._ble_client = None
        self._ble_connected = False
        self._ble_ready = threading.Event()
        self._ble_first_packet = threading.Event()
        self._ble_notify_uuid = None
        self._ble_buffer = b""
        self._serial_buffer = b""
        self._last_ble_time_s = None
        self._ble_debug_count = 0

        # Wall-clock time of the most recent packet from the sensor (any mode).
        # Used in start_collecting() to decide whether the stream is active so
        # we can skip the BLE notify-restart that was the primary cause of
        # intermittent data loss.
        self._last_packet_wall_time: float | None = None

        atexit.register(self.close)

    # ── public status ──────────────────────────────────────────────────

    @property
    def status_text(self) -> str:
        return self._status

    @property
    def last_error(self) -> str:
        return self._last_error

    # ── connection ─────────────────────────────────────────────────────

    def connect(self, target, baud=115200):
        self.close()
        if not target:
            self._set_error("No sensor target provided")
            return False

        if isinstance(target, str) and target.startswith(BLE_PREFIX):
            return self._connect_ble(target)
        return self._connect_serial(target, baud=baud)

    def _connect_serial(self, port, baud=115200):
        try:
            self.ser = serial.Serial(port, baud, timeout=0.05)
            time.sleep(0.6)
            self.port = port
            self.mode = "serial"
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            self._serial_buffer = b""
            self._stop_event.clear()
            self._status = f"Serial connected: {port}"
            self._last_error = ""
            self._start_reader_thread(self._serial_read_loop)
            return True
        except Exception as e:
            self.ser = None
            self.mode = None
            self._set_error(f"Serial connect failed: {e}")
            return False

    def _connect_ble(self, token):
        if BleakClient is None:
            self._set_error("BLE requires the 'bleak' package")
            return False

        payload = token[len(BLE_PREFIX):]
        address, _, name_hint = payload.partition("::")
        self.port = address
        self.mode = "ble"
        self._ble_ready.clear()
        self._ble_first_packet.clear()
        self._ble_connected = False
        self._stop_event.clear()
        self._last_error = ""
        self._status = f"Connecting over BLE: {address}"
        self._ble_buffer = b""
        self._ble_notify_uuid = None
        self._last_ble_time_s = None
        self._ble_debug_count = 0
        self._start_reader_thread(lambda: self._run_ble_session(address, name_hint or None))
        self._ble_ready.wait(timeout=15.0)
        if self._ble_connected:
            # Don't fail the connect if no packet has arrived yet. The
            # firmware was historically slow to start streaming, and
            # "Start Collecting" later does a notify-restart cycle that
            # sometimes kicks it loose. Leave the link up so the user can
            # try that path.
            if self._ble_first_packet.wait(timeout=2.0):
                self._status = f"BLE streaming: {address}"
            else:
                self._status = f"BLE connected, no data yet: {address}"
            return True
        self.close()
        if not self._last_error:
            self._set_error("BLE connect timed out")
        return False

    def _start_reader_thread(self, target):
        self._reader_thread = threading.Thread(target=target, daemon=True)
        self._reader_thread.start()

    def close(self):
        self._collecting = False
        self._stop_event.set()

        if self._ble_loop and self._ble_client:
            try:
                future = asyncio.run_coroutine_threadsafe(self._ble_disconnect(), self._ble_loop)
                future.result(timeout=5.0)
            except Exception:
                pass

        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=3.0)

        self.ser = None
        self.port = None
        self.mode = None
        self._ble_loop = None
        self._ble_client = None
        self._ble_connected = False
        self._ble_ready.clear()
        self._ble_first_packet.clear()
        self._ble_notify_uuid = None
        self._reader_thread = None
        self._status = "Offline"

    def is_open(self):
        if self.mode == "serial":
            return self.ser is not None and self.ser.is_open
        if self.mode == "ble":
            return self._ble_connected
        return False

    # ── tare ───────────────────────────────────────────────────────────

    def tare(self):
        with self._lock:
            if self._recent_calibrated:
                baseline = sum(self._recent_calibrated) / len(self._recent_calibrated)
                self.tare_offset = baseline
                self._last_live_mass = baseline - self.tare_offset
                return baseline
        return 0.0

    def clear_tare(self):
        self.tare_offset = 0.0

    @property
    def current_reading(self):
        with self._lock:
            return self._last_live_mass

    @property
    def current_raw(self):
        with self._lock:
            return self._last_live_raw

    # ── collection ─────────────────────────────────────────────────────

    def start_collecting(self):
        # Drop samples while we reset the buffer so no stale data leaks in.
        self._collecting = False

        with self._lock:
            self._timestamps.clear()
            self._masses.clear()
            self._raw_values.clear()
            self._time_offset = None

        if self.mode == "ble" and self._ble_loop and self._ble_client and self._ble_connected:
            # Only restart BLE notifications when the stream looks genuinely
            # stuck (no packet in the past 3 s).  Restarting an active stream
            # is the primary cause of intermittent data loss: stop_notify()
            # kills the stream for at least 150 ms, and some BLE devices take
            # much longer to resume — sometimes never recovering within the run.
            # When the stream is healthy we simply clear the buffer above and
            # let the already-flowing notifications start filling it again.
            now = time.time()
            last = self._last_packet_wall_time
            stream_stale = (last is None) or ((now - last) > 3.0)

            if stream_stale:
                # Clear the partial-packet text buffer so a half-received line
                # from before the restart cannot merge with the new stream.
                self._ble_buffer = b""
                # Reset the timestamp-jump guard so the first post-restart
                # sample is not silently dropped as a "suspicious" jump.
                self._last_ble_time_s = None
                self._ble_first_packet.clear()
                self._status = f"BLE connected, refreshing stream: {self.port}"
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        self._ble_restart_notify(), self._ble_loop
                    )
                    future.result(timeout=5.0)
                except Exception as e:
                    self._set_error(f"BLE stream refresh failed: {e}")
            # else: stream is active — notifications are already flowing into
            # the now-empty buffer, no disruption needed.

        self._collecting = True

    def stop_collecting(self):
        self._collecting = False

    def get_data(self):
        with self._lock:
            return list(self._timestamps), list(self._masses), list(self._raw_values)

    def get_data_since(self, index):
        with self._lock:
            return list(self._timestamps[index:]), list(self._masses[index:]), len(self._timestamps)

    @property
    def sample_count(self):
        with self._lock:
            return len(self._timestamps)

    @property
    def last_packet_age(self) -> float | None:
        """Seconds since the last packet from the sensor, or None if never received."""
        last = self._last_packet_wall_time
        return (time.time() - last) if last is not None else None

    # ── reader threads ─────────────────────────────────────────────────

    def _serial_read_loop(self):
        while not self._stop_event.is_set() and self.ser is not None and self.ser.is_open:
            try:
                avail = self.ser.in_waiting
                if avail > 0:
                    chunk = self.ser.read(avail)
                    self._serial_buffer += chunk
                    self._serial_buffer = self._parse_buffer(self._serial_buffer)
                else:
                    time.sleep(0.01)
            except Exception as e:
                self._set_error(f"Serial read error: {e}")
                time.sleep(0.05)

    def _run_ble_session(self, address, name_hint=None):
        loop = asyncio.new_event_loop()
        self._ble_loop = loop
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._ble_session(address, name_hint=name_hint))
        except asyncio.CancelledError:
            self._ble_ready.set()
        except Exception as e:
            self._set_error(f"BLE session failed: {e}")
            self._ble_ready.set()
        finally:
            try:
                pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            except Exception:
                pending = []

            for task in pending:
                task.cancel()

            if pending and not loop.is_closed():
                try:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                except Exception:
                    pass

            if not loop.is_closed():
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception:
                    pass
                try:
                    loop.run_until_complete(loop.shutdown_default_executor())
                except Exception:
                    pass
                try:
                    asyncio.set_event_loop(None)
                except Exception:
                    pass
                try:
                    loop.close()
                except Exception:
                    pass

            self._ble_loop = None

    async def _ble_session(self, address, name_hint=None):
        # Pass the raw address straight to BleakClient on macOS — that's what
        # the original (pre-debug) code did, and it's what produced the
        # "sometimes works" path. Adding a discovery scan ahead of connect
        # changed the ordering of CoreBluetooth state in a way that broke
        # the intermittent-success behavior. On Windows we still need to
        # resolve to a BLEDevice because the WinRT backend behaves better
        # that way.
        device_or_address = address
        if platform.system() == "Windows":
            resolved = await self._resolve_ble_device(address, name_hint=name_hint)
            if resolved is not None:
                device_or_address = resolved

        client = BleakClient(device_or_address)
        self._ble_client = client
        try:
            await client.connect(timeout=15.0)
            services = await self._get_ble_services(client)

            # Print the negotiated MTU. On macOS bleak does not request a
            # specific MTU, so this is whatever CoreBluetooth chose. If the
            # Xiao firmware bundles multiple samples per notification it
            # may refuse to transmit until MTU exceeds the bundle size,
            # which would silently appear as "subscribed but no data".
            try:
                mtu = getattr(client, "mtu_size", None)
                print(f"BLE negotiated MTU: {mtu}")
            except Exception as e:
                print(f"BLE MTU read failed: {e}")

            # Print everything bleak resolved so we can see whether the
            # service/characteristic discovery actually saw the streaming
            # characteristic and what properties it has. If this prints
            # "notify=False", the firmware is exposing the wrong char or
            # the OS is using a stale GATT cache.
            print("BLE discovered services:")
            for svc in services:
                print(f"  service {svc.uuid}")
                for ch in svc.characteristics:
                    props = ",".join(ch.properties) if ch.properties else "-"
                    print(f"    char {ch.uuid}  props={props}")

            target_char = self._find_ble_notify_characteristic(services)
            target_props = ",".join(target_char.properties) if target_char.properties else "-"
            print(f"BLE target char {target_char.uuid} props={target_props}")

            self._ble_notify_uuid = target_char.uuid
            await client.start_notify(target_char.uuid, self._ble_notification_handler)
            # Give the CCCD enable a moment to propagate end-to-end before we
            # declare the link ready. On macOS, start_notify returns as soon
            # as the local CoreBluetooth callback fires, which can race with
            # the peripheral actually flipping into "send notifications" mode.
            await asyncio.sleep(0.3)
            self._ble_connected = True
            self._status = f"BLE connected, waiting for data: {address}"
            self._last_error = ""
            print(f"BLE notify started on {target_char.uuid}; waiting for first packet")
            self._ble_ready.set()

            while not self._stop_event.is_set() and client.is_connected:
                await asyncio.sleep(0.05)
        finally:
            try:
                if client.is_connected and self._ble_notify_uuid:
                    try:
                        await client.stop_notify(self._ble_notify_uuid)
                    except Exception:
                        pass
                if client.is_connected:
                    await client.disconnect()
            except Exception:
                pass
            self._ble_connected = False
            self._ble_notify_uuid = None
            self._ble_first_packet.clear()

    async def _ble_disconnect(self):
        # Snapshot the client and notify UUID locally. close() runs on a
        # different thread and clears these fields concurrently; without the
        # snapshot we can race and dereference None inside this coroutine
        # ("'NoneType' object has no attribute 'disconnect'").
        client = self._ble_client
        notify_uuid = self._ble_notify_uuid
        try:
            if client is not None and client.is_connected:
                if notify_uuid:
                    try:
                        await client.stop_notify(notify_uuid)
                    except Exception:
                        pass
                try:
                    await client.disconnect()
                except Exception:
                    pass
        finally:
            self._ble_connected = False
            self._ble_notify_uuid = None
            self._ble_first_packet.clear()

    async def _ble_restart_notify(self):
        if self._ble_client is None or not self._ble_client.is_connected or not self._ble_notify_uuid:
            return

        try:
            await self._ble_client.stop_notify(self._ble_notify_uuid)
        except Exception:
            pass

        await asyncio.sleep(0.15)
        await self._ble_client.start_notify(self._ble_notify_uuid, self._ble_notification_handler)
        self._status = f"BLE connected, waiting for data: {self.port}"

    async def _resolve_ble_device(self, address, name_hint=None):
        if BleakScanner is None:
            return None
        try:
            devices = await BleakScanner.discover(timeout=8.0)
        except Exception:
            return None

        address_lower = address.lower()
        for device in devices:
            dev_addr = getattr(device, "address", "") or ""
            if dev_addr.lower() == address_lower:
                return device

        if name_hint:
            for device in devices:
                dev_name = getattr(device, "name", "") or ""
                if dev_name and dev_name == name_hint:
                    return device
        return None

    async def _get_ble_services(self, client):
        services = getattr(client, "services", None)
        if services:
            return services

        get_services = getattr(client, "get_services", None)
        if callable(get_services):
            services = await get_services()
            if services:
                return services

        services = getattr(client, "services", None)
        if services:
            return services

        raise RuntimeError("BLE service discovery failed")

    def _find_ble_notify_characteristic(self, services):
        target_char = None
        saw_expected_service = False

        for service in services:
            service_uuid = (service.uuid or "").lower()
            if service_uuid != XIAO_SERVICE_UUID:
                continue

            saw_expected_service = True
            for char in service.characteristics:
                if (char.uuid or "").lower() == XIAO_NOTIFY_UUID:
                    return char

            for char in service.characteristics:
                props = [p.lower() for p in char.properties]
                if "notify" in props:
                    target_char = char
                    break

            if target_char is not None:
                return target_char

        if not saw_expected_service:
            raise RuntimeError(
                f"Expected AutoFlow BLE service {XIAO_SERVICE_UUID} not found. "
                "This is probably the wrong BLE device."
            )

        raise RuntimeError(
            f"Expected AutoFlow notify characteristic {XIAO_NOTIFY_UUID} not found in service {XIAO_SERVICE_UUID}."
        )

    def _ble_notification_handler(self, _sender, data):
        if not data:
            return

        self._ble_first_packet.set()
        if self.port:
            self._status = f"BLE streaming: {self.port}"
        payload = bytes(data)
        if self._ble_debug_count < 25:
            print(f"BLE notify[{self._ble_debug_count}] len={len(payload)} data={payload!r}")
        if self._ble_debug_count == 0:
            print("BLE stream is delivering sensor packets")
        self._ble_debug_count += 1

        # Text mode notifications, e.g. "[timestamp_ms, raw_value]\n"
        if b"\n" in payload or payload.startswith(b"["):
            self._ble_buffer += payload
            self._ble_buffer = self._parse_text_buffer(self._ble_buffer)
            return

        # Binary mode notifications: each sample should be exactly 8 bytes
        # [int32_le raw_value, int32_le timestamp_centiseconds]
        if len(payload) % 8 != 0:
            print(f"Ignoring malformed BLE payload length={len(payload)} bytes: {payload!r}")
            return

        for i in range(0, len(payload), 8):
            chunk = payload[i:i + 8]
            try:
                raw_val = struct.unpack_from("<i", chunk, 0)[0]
                ts_cs = struct.unpack_from("<i", chunk, 4)[0]
            except struct.error:
                continue
            self._add_ble_sample(float(raw_val), ts_cs / 100.0)

    # ── parsing ────────────────────────────────────────────────────────

    def _parse_text_buffer(self, buf: bytes) -> bytes:
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            text = line.decode(errors="replace").strip()
            if text.startswith("[") and text.endswith("]"):
                try:
                    parts = text[1:-1].split(",")
                    ts_ms = float(parts[0])
                    raw_val = float(parts[1])
                    self._add_sample(raw_val, ts_ms / 1000.0)
                except (ValueError, IndexError):
                    pass

        if len(buf) > 4096:
            buf = buf[-1024:]
        return buf

    def _parse_buffer(self, buf: bytes) -> bytes:
        # Serial path may contain either newline-delimited text or packed binary samples.
        if b"\n" in buf:
            return self._parse_text_buffer(buf)

        while len(buf) >= 8:
            try:
                raw_val = struct.unpack_from("<i", buf, 0)[0]
                ts_cs = struct.unpack_from("<i", buf, 4)[0]
                buf = buf[8:]
                self._add_sample(float(raw_val), ts_cs / 100.0)
            except struct.error:
                break

        if len(buf) > 4096:
            buf = buf[-1024:]
        return buf

    def _add_ble_sample(self, raw_value: float, raw_time_s: float):
        # BLE notifications should arrive as clean packet boundaries.
        # Reject obviously corrupted samples instead of letting one bad packet destroy the plot.
        if raw_time_s < 0:
            return
        if self._last_ble_time_s is not None:
            dt = raw_time_s - self._last_ble_time_s
            if dt < 0 or dt > 5.0:
                print(f"Dropping suspicious BLE sample with dt={dt:.3f}s raw_time_s={raw_time_s}")
                return
        calibrated_mass = raw_value * self.calibration_factor
        if not (-1000.0 <= calibrated_mass <= 5000.0):
            print(f"Dropping suspicious BLE sample raw={raw_value} calibrated={calibrated_mass}")
            return

        self._last_ble_time_s = raw_time_s
        self._add_sample(raw_value, raw_time_s)

    def _add_sample(self, raw_value: float, raw_time_s: float):
        # Record arrival time before anything else — this is used by
        # start_collecting() to tell whether the stream is currently active.
        # Written outside the lock; Python GIL makes the assignment atomic.
        self._last_packet_wall_time = time.time()

        calibrated_mass = raw_value * self.calibration_factor
        tare_adjusted_mass = calibrated_mass - self.tare_offset

        with self._lock:
            self._last_live_raw = raw_value
            self._last_live_mass = tare_adjusted_mass
            self._recent_calibrated.append(calibrated_mass)

            if not self._collecting:
                return

            # Use wall-clock time so the axis is always real seconds.
            # The firmware timestamp unit varies (ms, cs, etc.) between
            # builds; wall clock is unambiguous and matches the playback
            # loop timer used on the pump side.
            wall_now = time.time()
            if self._time_offset is None:
                self._time_offset = wall_now
            t = wall_now - self._time_offset
            self._timestamps.append(t)
            self._masses.append(tare_adjusted_mass)
            self._raw_values.append(raw_value)

            if len(self._timestamps) > 60000:
                self._timestamps = self._timestamps[-50000:]
                self._masses = self._masses[-50000:]
                self._raw_values = self._raw_values[-50000:]

    def _set_error(self, msg: str):
        self._last_error = msg
        self._status = msg
        print(msg)

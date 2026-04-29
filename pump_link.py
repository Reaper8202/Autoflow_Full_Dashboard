"""Serial communication with the Feather pump controller."""

import json
import time

import serial
import serial.tools.list_ports

PUMP_MAX_RPM = 1200.0
MIN_RPM_THRESHOLD = 5.0
RPM_WRITE_EPSILON = 0.75


def list_serial_ports():
    return [p.device for p in serial.tools.list_ports.comports()]


class PumpLink:
    def __init__(self):
        self.ser = None
        self.port = None
        self.log = []
        self.last_stop_reason = None

    # -- connection ----------------------------------------------------------

    def connect(self, port, baud=115200):
        self.close()
        try:
            self.ser = serial.Serial(port, baud, timeout=0.5, write_timeout=0.5)
            time.sleep(1.5)
            self.port = port
            self._flush()
            self._log("sys", f"connected to {port}")
            return True
        except Exception as e:
            self._log("err", f"connect failed: {e}")
            self.ser = None
            return False

    def close(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None
        self.port = None

    def is_open(self):
        return self.ser is not None and self.ser.is_open

    # -- I/O -----------------------------------------------------------------

    def send(self, cmd, wait_reply_s=0.12):
        if not self.write_line(cmd):
            return ""
        time.sleep(wait_reply_s)
        return self._read_all()

    def write_line(self, line):
        if not self.is_open():
            self._log("err", "not connected")
            return False
        try:
            payload = (line + "\r\n").encode()
            self.ser.write(payload)
            self.ser.flush()
            self._log("tx", line)
            return True
        except Exception as e:
            self._log("err", f"write failed: {e}")
            return False

    def write_realtime_line(self, line):
        """Streamed-control write.

        Earlier versions of this method called ``reset_output_buffer()`` before
        every send to "drop stale commands." That was actively harmful: on
        Windows it issues ``PurgeComm(PURGE_TXABORT | PURGE_TXCLEAR)`` and on
        macOS ``tcflush(fd, TCOFLUSH)``. Both can abort writes that are already
        in flight to the USB CDC endpoint, so the previous RPM update gets
        killed before the firmware ever sees it. That is exactly what made the
        pump stick at stale RPMs (~348/384) during exact playback.

        Commands are ~9 bytes and the host sends ~5 per second. The firmware
        processes them in order and the last one wins, so we just need to let
        the bytes through.
        """
        if not self.is_open():
            self._log("err", "not connected")
            return False
        try:
            payload = (line + "\r\n").encode()
            self.ser.write(payload)
            self.ser.flush()
            self._log("tx", f"realtime:{line}")
            return True
        except Exception as e:
            self._log("err", f"realtime write failed: {e}")
            return False

    def hard_stop(self, reason=""):
        """Best-effort repeated stop sequence, mainly for flaky Windows serial links."""
        if not self.is_open():
            self._log("err", "hard_stop requested while not connected")
            return False

        self.last_stop_reason = reason or "unspecified"
        self._log("sys", f"hard_stop start reason={self.last_stop_reason}")

        ok = False
        sequence = ["0", "stop", "abort", "0", "stop"]
        for cmd in sequence:
            wrote = self.write_line(cmd)
            ok = ok or wrote
            time.sleep(0.08)
            self.drain()

        try:
            self.ser.flush()
        except Exception:
            pass

        self._log("sys", f"hard_stop end reason={self.last_stop_reason}")
        return ok

    def drain(self):
        if not self.is_open():
            return []
        try:
            avail = self.ser.in_waiting
            if not avail:
                return []
            reply = self.ser.read(avail).decode(errors="replace")
        except Exception as e:
            self._log("err", f"read failed: {e}")
            return []
        lines = []
        for line in reply.splitlines():
            line = line.strip()
            if line:
                self._log("rx", line)
                lines.append(line)
        return lines

    # -- helpers -------------------------------------------------------------

    def get_factor(self):
        reply = self.send("getfactor", wait_reply_s=0.15)
        for line in reply.splitlines():
            if line.startswith("FACTOR="):
                try:
                    return float(line.split("=", 1)[1])
                except ValueError:
                    pass
        return None

    def get_state(self):
        reply = self.send("getstate", wait_reply_s=0.15)
        for line in reply.splitlines():
            if line.startswith("STATE="):
                try:
                    return json.loads(line.split("=", 1)[1])
                except Exception:
                    pass
        return None

    # -- internals -----------------------------------------------------------

    def _flush(self):
        if not self.is_open():
            return
        try:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
        except Exception:
            pass

    def _read_all(self):
        reply = ""
        try:
            while self.ser.in_waiting:
                reply += self.ser.read(self.ser.in_waiting).decode(errors="replace")
                time.sleep(0.01)
        except Exception as e:
            self._log("err", f"read failed: {e}")
        for line in reply.splitlines():
            if line.strip():
                self._log("rx", line)
        return reply

    def clear_log(self):
        self.log = []

    def snapshot_log(self):
        return list(self.log)

    def export_log_text(self):
        return "\n".join(f"[{ts}] {direction.upper():>3} {text}" for ts, direction, text in self.log)

    def _log(self, direction, text):
        self.log.append((time.strftime("%H:%M:%S"), direction, text))
        if len(self.log) > 500:
            self.log = self.log[-500:]

"""
Automated Dashboard — unified uroflowmetry platform.

  streamlit run app.py

Three pages:
  1. Sensor & Calibration — live load cell, tare, calibration workflow
  2. Test Results         — load AutoFlow CSV exports, view graphs + analysis
  3. Run Automated Test   — control the peristaltic pump
"""

import os
import io
import json
import time
import base64
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image

from pump_link import PumpLink, list_serial_ports, PUMP_MAX_RPM, MIN_RPM_THRESHOLD, RPM_WRITE_EPSILON
from sensor_link import (
    SensorLink,
    DEFAULT_CALIBRATION_FACTOR,
    BLE_AVAILABLE,
    discover_ble_devices,
    list_sensor_serial_ports,
)
from analysis import (
    SHAPES, TEMPLATES, shape_curve, build_run_profile, analyze_curve,
    compute_flow_from_mass, kz_filter,
)

CONFIG_PATH = Path.home() / ".autoflow_dashboard_config.json"
DEFAULT_CONFIG = {
    "cal_factor": 0.030,
    "pump_max_rpm": 350.0,
    "last_com_port": "",
    "last_sensor_port": "",
    "sensor_cal_factor": DEFAULT_CALIBRATION_FACTOR,
    "templates": {},
}


def _load_cfg():
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
            for k, v in DEFAULT_CONFIG.items():
                cfg.setdefault(k, v)
            return cfg
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()


def _save_cfg(cfg):
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


# ── session state init ─────────────────────────────────────────────────────

def _init():
    if "cfg" not in st.session_state:
        st.session_state.cfg = _load_cfg()
    else:
        # Patch in any new DEFAULT_CONFIG keys that were added after this session started
        cfg = st.session_state.cfg
        changed = False
        for k, v in DEFAULT_CONFIG.items():
            if k not in cfg:
                cfg[k] = v
                changed = True
        if changed:
            _save_cfg(cfg)
    if "link" not in st.session_state:
        st.session_state.link = PumpLink()
    if "sensor" not in st.session_state:
        sensor = SensorLink()
        cfg = st.session_state.cfg
        sensor.calibration_factor = cfg.get("sensor_cal_factor", DEFAULT_CALIBRATION_FACTOR)
        st.session_state.sensor = sensor
    if "calibration_map" not in st.session_state:
        st.session_state.calibration_map = [[], []]  # [masses, drain_rates]
    if "last_run_analysis" not in st.session_state:
        st.session_state.last_run_analysis = None
    if "run_analysis_status" not in st.session_state:
        st.session_state.run_analysis_status = "Idle"
    if "last_run_pdf" not in st.session_state:
        st.session_state.last_run_pdf = None
    if "last_run_pdf_for" not in st.session_state:
        st.session_state.last_run_pdf_for = None
    if "fit_recording" not in st.session_state:
        st.session_state.fit_recording = False
    if "multi_cal_runs" not in st.session_state:
        st.session_state.multi_cal_runs = []
    if "capture_run_for_multi_fit" not in st.session_state:
        st.session_state.capture_run_for_multi_fit = False
    if "pending_multi_fit_run_name" not in st.session_state:
        st.session_state.pending_multi_fit_run_name = None
    if "last_multi_fit_capture_result" not in st.session_state:
        st.session_state.last_multi_fit_capture_result = None


# ── sidebar ────────────────────────────────────────────────────────────────

def _sidebar():
    cfg = st.session_state.cfg
    link = st.session_state.link
    sensor = st.session_state.sensor

    with st.sidebar:
        ports = list_serial_ports()
        sensor_ports = list_sensor_serial_ports()

        # ── sensor connection ──────────────────────────────────────────
        st.header("Sensor")
        sensor_mode = st.radio("Sensor transport", ["Bluetooth", "USB Serial"], horizontal=True, key="sb_sensor_mode")

        if "ble_devices" not in st.session_state:
            st.session_state.ble_devices = []

        selected_sensor_target = cfg.get("last_sensor_port", "")

        if sensor_mode == "Bluetooth":
            if not BLE_AVAILABLE:
                st.error("BLE support is unavailable in the current Python environment. Install 'bleak' and restart Streamlit.")

            bc1, bc2 = st.columns([1, 1])
            with bc1:
                if st.button("Scan BLE", key="sb_ble_scan", use_container_width=True, disabled=not BLE_AVAILABLE):
                    try:
                        with st.spinner("Scanning for BLE devices..."):
                            st.session_state.ble_devices = discover_ble_devices(timeout=6.0)
                        if st.session_state.ble_devices:
                            st.success(f"Found {len(st.session_state.ble_devices)} BLE device(s)")
                        else:
                            st.warning("BLE scan completed, but no matching devices were found.")
                    except Exception as e:
                        st.session_state.ble_devices = []
                        st.error(str(e))
            with bc2:
                if st.session_state.ble_devices:
                    st.caption(f"Found {len(st.session_state.ble_devices)} BLE device(s)")
                else:
                    st.caption("No BLE scan results yet")

            ble_devices = st.session_state.ble_devices
            if ble_devices:
                ble_labels = [d.label for d in ble_devices]
                default_idx = 0
                last_sensor_port = cfg.get("last_sensor_port", "")
                for i, dev in enumerate(ble_devices):
                    if dev.token == last_sensor_port:
                        default_idx = i
                        break
                ble_idx = st.selectbox("BLE device", range(len(ble_labels)), format_func=lambda i: ble_labels[i], index=default_idx, key="sb_ble_device")
                selected_sensor_target = ble_devices[ble_idx].token
            else:
                manual_ble = st.text_input("BLE address", value="", placeholder="XX:XX:XX:XX:XX:XX", key="sb_ble_manual")
                if manual_ble.strip():
                    selected_sensor_target = f"BLE::{manual_ble.strip()}::manual"
        else:
            s_default = 0
            if cfg["last_sensor_port"] in sensor_ports:
                s_default = sensor_ports.index(cfg["last_sensor_port"])

            if sensor_ports:
                selected_sensor_target = st.selectbox("Sensor port", sensor_ports, index=s_default, key="sb_sensor_port")
            else:
                selected_sensor_target = st.text_input("Sensor port", cfg["last_sensor_port"], key="sb_sensor_port_manual")

        sc1, sc2 = st.columns(2)
        with sc1:
            if st.button("Connect", key="sb_sensor_connect", use_container_width=True):
                if sensor.connect(selected_sensor_target):
                    cfg["last_sensor_port"] = selected_sensor_target
                    _save_cfg(cfg)
                    st.success("Sensor connected")
                else:
                    st.error(sensor.last_error or "Sensor connect failed")
        with sc2:
            if st.button("Disconnect", key="sb_sensor_disconnect", use_container_width=True):
                sensor.close()

        s_status = "green" if sensor.is_open() else "red"
        s_text = sensor.status_text if sensor.is_open() else "Offline"
        st.markdown(f"Sensor: :{s_status}[**{s_text}**]")
        if sensor.is_open():
            age = sensor.last_packet_age
            if age is None:
                st.caption("Stream: no packets yet")
            elif age < 2.0:
                st.caption(f"Stream: :green[live] ({age:.1f}s)")
            else:
                st.caption(f"Stream: :red[stale — {age:.1f}s since last packet]")
        if sensor.last_error and not sensor.is_open():
            st.caption(f"Last error: {sensor.last_error}")

        # ── pump connection ────────────────────────────────────────────
        st.divider()
        st.header("Pump")
        p_default = 0
        if cfg["last_com_port"] in ports:
            p_default = ports.index(cfg["last_com_port"])

        if ports:
            p_port = st.selectbox("Pump port", ports, index=p_default, key="sb_pump_port")
        else:
            p_port = st.text_input("Pump port", cfg["last_com_port"])

        pc1, pc2 = st.columns(2)
        with pc1:
            if st.button("Connect", key="sb_pump_connect", use_container_width=True):
                if link.connect(p_port):
                    cfg["last_com_port"] = p_port
                    _save_cfg(cfg)
                    st.success("Pump connected")
                else:
                    st.error("Pump connect failed")
        with pc2:
            if st.button("Disconnect", key="sb_pump_disconnect", use_container_width=True):
                link.close()

        p_status = "green" if link.is_open() else "red"
        p_text = "Connected" if link.is_open() else "Offline"
        st.markdown(f"Pump: :{p_status}[**{p_text}**]")

        # ── settings ──────────────────────────────────────────────────
        st.divider()
        st.subheader("Settings")
        new_k = st.number_input(
            "Pump cal (mL/s per RPM)", value=float(cfg["cal_factor"]),
            min_value=0.0001, max_value=1.0, format="%.5f", step=0.0001,
        )
        if new_k != cfg["cal_factor"]:
            cfg["cal_factor"] = float(new_k)
            _save_cfg(cfg)

        new_max_rpm = st.number_input(
            "Pump max RPM (hardware cap)", value=float(cfg.get("pump_max_rpm", 350.0)),
            min_value=1.0, max_value=float(PUMP_MAX_RPM), format="%.0f", step=10.0,
        )
        if new_max_rpm != cfg.get("pump_max_rpm"):
            cfg["pump_max_rpm"] = float(new_max_rpm)
            _save_cfg(cfg)

        new_sf = st.number_input(
            "Sensor cal (raw → grams)", value=float(cfg.get("sensor_cal_factor", DEFAULT_CALIBRATION_FACTOR)),
            min_value=0.0000001, max_value=1.0, format="%.8f", step=0.0000001,
        )
        if new_sf != cfg.get("sensor_cal_factor"):
            cfg["sensor_cal_factor"] = float(new_sf)
            sensor.calibration_factor = float(new_sf)
            _save_cfg(cfg)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 1 — TEST RESULTS
# ══════════════════════════════════════════════════════════════════════════

def _find_autoflow_csvs():
    """Scan common locations for AutoFlow combined_export CSVs."""
    candidates = []
    search_dirs = [
        Path.home() / "Documents" / "AutoFlow",
        Path.home() / "Library" / "Containers" / "com.example.firstApp" / "Data" / "Documents" / "AutoFlow",
        Path("/storage/self/primary/Documents/AutoFlow"),
    ]
    # Also check the Farcas_Lab folder itself
    lab_dir = Path(__file__).resolve().parent.parent
    search_dirs.append(lab_dir)
    for d in lab_dir.iterdir():
        if d.is_dir():
            search_dirs.append(d)

    for d in search_dirs:
        if not d.exists():
            continue
        for f in sorted(d.glob("combined_export_*.csv"), reverse=True):
            candidates.append(f)
        for f in sorted(d.glob("*.csv"), reverse=True):
            if f not in candidates:
                candidates.append(f)
    return candidates[:30]


def _parse_autoflow_csv(path_or_upload):
    """Parse an AutoFlow combined CSV and return sections as DataFrames."""
    if hasattr(path_or_upload, "read"):
        text = path_or_upload.read().decode(errors="replace")
    else:
        text = Path(path_or_upload).read_text(errors="replace")

    sections = {}
    current_section = None
    current_rows = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            if current_section and current_rows:
                sections[current_section] = current_rows
            current_section = stripped[2:].split("(")[0].strip()
            current_rows = []
        elif stripped:
            current_rows.append(stripped)

    if current_section and current_rows:
        sections[current_section] = current_rows

    result = {}
    for name, rows in sections.items():
        if len(rows) < 2:
            continue
        from io import StringIO
        csv_text = "\n".join(rows)
        try:
            df = pd.read_csv(StringIO(csv_text))
            if len(df) > 0 and len(df.columns) >= 2:
                result[name] = df
        except Exception:
            pass
    return result


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 1 — SENSOR & CALIBRATION
# ══════════════════════════════════════════════════════════════════════════

def page_sensor():
    st.header("Sensor & Calibration")
    sensor = st.session_state.sensor
    cfg = st.session_state.cfg

    if not sensor.is_open():
        st.info("Connect to the sensor in the sidebar to begin.")

    # ── top controls row ───────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if st.button("Tare (Zero)", use_container_width=True, disabled=not sensor.is_open()):
            val = sensor.tare()
            st.success(f"Tared at {val:.1f} g")
    with c2:
        if st.button("Clear Tare", use_container_width=True):
            sensor.clear_tare()
            st.info("Tare cleared")
    with c3:
        live_reading = sensor.current_reading
        raw_reading = sensor.current_raw
        st.metric("Live Reading", f"{live_reading:.2f} g")
    with c4:
        st.metric("Tare Offset", f"{sensor.tare_offset:.2f} g")

    st.divider()

    # ── three sub-tabs ─────────────────────────────────────────────────
    tab_live, tab_calib, tab_map = st.tabs(["Live Graph", "Calibrate", "Calibration Map"])

    # ── live graph ─────────────────────────────────────────────────────
    with tab_live:
        lc1, lc2, lc3 = st.columns(3)
        start_btn = lc1.button("Start Collecting", use_container_width=True, disabled=not sensor.is_open())
        stop_btn = lc2.button("Stop", key="stop_live", use_container_width=True)
        refresh_btn = lc3.button("Refresh Graph", use_container_width=True)

        if start_btn:
            sensor.start_collecting()
            st.session_state["live_collecting"] = True
        if stop_btn:
            sensor.stop_collecting()
            st.session_state["live_collecting"] = False

        is_collecting = st.session_state.get("live_collecting", False)
        if is_collecting:
            st.caption(f"Collecting... {sensor.sample_count} samples")

        t_data, m_data, _ = sensor.get_data()
        if t_data:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=t_data, y=m_data, name="Mass",
                line=dict(color="#2196F3", width=2),
            ))
            fig.update_layout(
                xaxis_title="Time (s)", yaxis_title="Mass (g)",
                height=400, margin=dict(t=30),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"{len(t_data)} samples collected")
        else:
            st.info("No data yet. Press Start Collecting to begin.")

        # Auto-refresh while collecting — only rerun if calibration isn't active,
        # otherwise the rerun fires before the Calibrate tab code executes and
        # swallows every button press on that tab.
        if is_collecting and not st.session_state.get("calibrating", False):
            time.sleep(0.5)
            st.rerun()

    # ── calibration workflow ───────────────────────────────────────────
    with tab_calib:
        st.subheader("Drain Rate Calibration")
        cal_method = st.radio(
            "Calibration method",
            ["Fill then drain", "Drain open from start"],
            horizontal=True,
            key="cal_method",
        )
        target_start_mass_g = st.number_input(
            "Target start mass (g)",
            value=float(st.session_state.get("cal_target_start_mass_g", 250.0)),
            min_value=10.0,
            max_value=2000.0,
            step=10.0,
            key="cal_target_start_mass_g",
        )

        if cal_method == "Fill then drain":
            st.info(
                "**Pump-only setup (no passive drain):** press **Set Zero Drain** — no calibration needed.\n\n"
                "**Passive-drain setup:** fill the container, press **Start Calibration**, "
                "then open the drain and let water flow out naturally. Press **Stop** when empty "
                "or wait for the 60 s auto-stop."
            )
        else:
            st.info(
                "**Pump-only setup (no passive drain):** press **Set Zero Drain** — no calibration needed.\n\n"
                "**Drain-open setup:** leave the drain open before you start. Press **Start Calibration**, "
                f"pour until the container reaches about **{target_start_mass_g:.0f} g**, then stop pouring "
                "and let it drain out naturally. Press **Stop** when empty or wait for the 60 s auto-stop."
            )

        cc1, cc2, cc3, cc4 = st.columns(4)
        cal_start = cc1.button("Start Calibration", use_container_width=True, disabled=not sensor.is_open())
        cal_stop = cc2.button("Stop Calibration", use_container_width=True)
        cal_save = cc3.button("Save Map to CSV", use_container_width=True)
        cal_zero = cc4.button("Set Zero Drain", use_container_width=True)

        if cal_zero:
            st.session_state.calibration_map = [[], []]
            st.success("Calibration set to zero drain — correct for pump-only setups.")

        if cal_start:
            # Stop live graph loop so its st.rerun() doesn't preempt calibration
            st.session_state["live_collecting"] = False
            sensor.start_collecting()
            st.session_state["calibrating"] = True
            st.session_state["cal_start_time"] = time.time()
            st.session_state.pop("cal_result_msg", None)

        if cal_stop and st.session_state.get("calibrating"):
            sensor.stop_collecting()
            st.session_state["calibrating"] = False
            _build_calibration_map(method=cal_method, target_start_mass_g=target_start_mass_g)

        is_calibrating = st.session_state.get("calibrating", False)

        if is_calibrating:
            elapsed = time.time() - st.session_state.get("cal_start_time", time.time())
            t_data, m_data, _ = sensor.get_data()

            # Determine what the mass is doing so we can guide the user
            if len(m_data) >= 6:
                recent_delta = m_data[-1] - m_data[-6]   # change over last ~3 samples
            else:
                recent_delta = 0.0

            # Step guidance banner
            if cal_method == "Drain open from start":
                current_mass = float(m_data[-1]) if m_data else 0.0
                remaining = float(target_start_mass_g) - current_mass
                if elapsed < 2.0 or len(m_data) < 4:
                    st.success("**Step 1 — Start pouring with the drain already open.**")
                elif current_mass < float(target_start_mass_g) - 10.0:
                    st.success(
                        f"**Keep pouring** — current mass {current_mass:.1f} g, "
                        f"target {target_start_mass_g:.0f} g ({remaining:.1f} g to go)."
                    )
                elif current_mass <= float(target_start_mass_g) + 10.0:
                    st.info(
                        f"**Target reached** — current mass {current_mass:.1f} g. "
                        "Stop pouring and let it drain."
                    )
                elif recent_delta > 0.5:
                    st.warning(
                        f"**Above target** — current mass {current_mass:.1f} g. "
                        "Stop pouring now and let it drain down."
                    )
                elif recent_delta < -0.5:
                    st.info(f"**Drain detected ({recent_delta:.1f} g)** — let it keep draining.")
                else:
                    st.warning("**No mass change detected** — pour more or check that water is flowing.")
            else:
                if elapsed < 2.0 or len(m_data) < 4:
                    st.success("**Step 1 — Start pouring water into the container now.**")
                elif recent_delta > 0.5:
                    st.success(f"**Filling detected (+{recent_delta:.1f} g) — keep pouring.**")
                elif recent_delta < -0.5:
                    st.info(f"**Drain detected ({recent_delta:.1f} g) — keep the drain open.**")
                else:
                    st.warning("**No mass change detected — is water flowing? Pour or open the drain.**")

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Elapsed", f"{elapsed:.0f} s")
            mc2.metric("Samples", len(t_data))
            mc3.metric("Current mass", f"{m_data[-1]:.1f} g" if m_data else "—")

            st.progress(min(1.0, elapsed / 60.0), text=f"Collecting... {elapsed:.0f}/60 s  (press Stop when done)")

            if t_data:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=t_data, y=m_data, name="Mass", line=dict(color="#4CAF50", width=2)))
                fig.update_layout(xaxis_title="Time (s)", yaxis_title="Mass (g)", height=280, margin=dict(t=20))
                st.plotly_chart(fig, use_container_width=True)

            # Auto-stop at 60 s only — the old stability-based stop was the bug:
            # it fired immediately on a resting sensor, silently failed to build
            # the map, and reset state before the user could see any feedback.
            if elapsed >= 60.0:
                sensor.stop_collecting()
                st.session_state["calibrating"] = False
                _build_calibration_map(method=cal_method, target_start_mass_g=target_start_mass_g)
            else:
                time.sleep(0.5)
                st.rerun()

        if cal_save:
            _save_calibration_map_csv()

        st.divider()
        st.subheader("Generate Estimated Map")
        st.caption(
            "Use a reusable physics-shaped drain map when physical drain calibration is too noisy or impractical."
        )
        ec1, ec2, ec3, ec4 = st.columns(4)
        est_max_mass_g = ec1.number_input(
            "Reference mass (g)",
            value=float(st.session_state.get("est_map_max_mass_g", 250.0)),
            min_value=10.0,
            max_value=2000.0,
            step=10.0,
            key="est_map_max_mass_g",
            help="Max mass expected on sensor during runs. Set to typical peak mass.",
        )
        est_drain_rate = ec2.number_input(
            "Drain rate at reference (g/s)",
            value=float(st.session_state.get("est_map_drain_rate_g_s", -8.0)),
            min_value=-100.0,
            max_value=-0.01,
            step=0.1,
            format="%.2f",
            key="est_map_drain_rate_g_s",
            help="How fast water exits at reference mass. Must be negative.",
        )
        est_exponent = ec3.number_input(
            "Physics exponent",
            value=float(st.session_state.get("est_map_exponent", 0.5)),
            min_value=0.1,
            max_value=2.0,
            step=0.05,
            format="%.2f",
            key="est_map_exponent",
            help="0.5 = Torricelli/gravity drain. Increase for faster drain at low mass.",
        )
        est_threshold = ec4.number_input(
            "Drain threshold (g)",
            value=float(st.session_state.get("est_map_threshold_g", 0.0)),
            min_value=0.0,
            max_value=200.0,
            step=5.0,
            format="%.1f",
            key="est_map_threshold_g",
            help="Mass below which drain = 0. Raise this if small-volume tests over-estimate volume.",
        )
        if st.button("Generate Estimated Map", use_container_width=True, key="build_est_map"):
            _build_estimated_calibration_map(est_max_mass_g, est_drain_rate, est_exponent, drain_threshold_g=est_threshold)

        st.divider()
        st.subheader("Fit Estimated Map From Known Volume")
        st.caption(
            "Record one full trusted run, including post-run drain-down, then fit a reusable estimated map to that trace."
        )
        fc1, fc2, fc3 = st.columns(3)
        fit_expected_volume_ml = fc1.number_input(
            "Known total volume (mL)",
            value=float(st.session_state.get("fit_map_expected_volume_ml", 600.0)),
            min_value=10.0,
            max_value=5000.0,
            step=10.0,
            key="fit_map_expected_volume_ml",
        )
        fit_reference_mass_g = fc2.number_input(
            "Fit reference mass (g)",
            value=float(st.session_state.get("fit_map_reference_mass_g", 250.0)),
            min_value=10.0,
            max_value=2000.0,
            step=10.0,
            key="fit_map_reference_mass_g",
        )
        fit_tail_seconds = fc3.number_input(
            "Required drained tail (s)",
            value=float(st.session_state.get("fit_map_tail_seconds", 8.0)),
            min_value=2.0,
            max_value=300.0,
            step=1.0,
            key="fit_map_tail_seconds",
        )
        fit_source_label = None
        fit_source_t = None
        fit_source_q = None
        fit_upload = st.file_uploader(
            "Upload known-volume playback CSV",
            type=["csv"],
            key="fit_run_csv",
        )
        if fit_upload is not None:
            try:
                fit_upload_df = pd.read_csv(fit_upload)
                fit_t_col = st.selectbox("Fit time column", fit_upload_df.columns, index=0, key="fit_t_col")
                fit_q_col = st.selectbox(
                    "Fit flow column",
                    fit_upload_df.columns,
                    index=min(1, len(fit_upload_df.columns) - 1),
                    key="fit_q_col",
                )
                fit_source_label = fit_upload.name
                fit_source_t = fit_upload_df[fit_t_col].to_numpy(dtype=float)
                fit_source_q = fit_upload_df[fit_q_col].to_numpy(dtype=float)
            except Exception as exc:
                st.error(f"Could not parse fit CSV: {exc}")
        else:
            fit_profile_candidates = []
            for candidate in sorted(Path(__file__).resolve().parent.glob("*.csv")):
                if candidate.name.startswith("calibration_map_"):
                    continue
                fit_profile_candidates.append(candidate)
            fit_profile_names = [p.name for p in fit_profile_candidates]
            fit_profile_idx = 0
            default_fit_profile = st.session_state.get("fit_profile_name", "")
            if default_fit_profile in fit_profile_names:
                fit_profile_idx = fit_profile_names.index(default_fit_profile)
            fit_profile_name = st.selectbox(
                "Or choose a local playback CSV",
                fit_profile_names if fit_profile_names else ["No CSV profiles found"],
                index=fit_profile_idx,
                key="fit_profile_name",
            )
            fit_profile_path = next((p for p in fit_profile_candidates if p.name == fit_profile_name), None) if fit_profile_names else None
            if fit_profile_path is not None:
                try:
                    fit_local_df = pd.read_csv(fit_profile_path)
                    fit_t_col = _find_col(fit_local_df, ["time_s", "time"])
                    fit_q_col = _find_col(fit_local_df, ["flow_ml_s", "flow"])
                    if fit_t_col is not None and fit_q_col is not None:
                        fit_source_label = fit_profile_path.name
                        fit_source_t = fit_local_df[fit_t_col].to_numpy(dtype=float)
                        fit_source_q = fit_local_df[fit_q_col].to_numpy(dtype=float)
                except Exception:
                    pass
        fit_playback_speed = st.slider(
            "Fit playback speed",
            0.25,
            4.0,
            1.0,
            0.25,
            key="fit_playback_speed",
        )
        fit_can_start = (
            st.session_state.link.is_open()
            and sensor.is_open()
            and fit_source_t is not None
            and fit_source_q is not None
        )
        frc1, frc2 = st.columns(2)
        if frc1.button(
            "Start Fit Recording",
            use_container_width=True,
            disabled=not fit_can_start,
            key="fit_record_start",
        ):
            st.session_state["live_collecting"] = False
            st.session_state["calibrating"] = False
            sensor.start_collecting()
            st.session_state["fit_recording"] = True
            st.session_state["fit_record_start_time"] = time.time()

            try:
                fit_source_q = np.clip(np.asarray(fit_source_q, dtype=float), 0.0, None)
                fit_source_t = np.asarray(fit_source_t, dtype=float)
                fit_effective_max_rpm = min(float(cfg.get("pump_max_rpm", PUMP_MAX_RPM)), float(PUMP_MAX_RPM))
                fit_peak_rpm = float(np.max(fit_source_q)) / cfg["cal_factor"] if cfg["cal_factor"] > 0 and len(fit_source_q) else 0.0
                fit_profile_scale = 1.0
                if fit_peak_rpm > fit_effective_max_rpm + 1e-6 and fit_peak_rpm > 0:
                    fit_profile_scale = fit_effective_max_rpm / fit_peak_rpm
                fit_scaled_q = fit_source_q * fit_profile_scale
                st.session_state["last_run_expected_volume"] = float(np.trapezoid(fit_scaled_q, fit_source_t)) / max(fit_playback_speed, 1e-6)
                fit_source_duration = float(fit_source_t[-1] - fit_source_t[0]) if len(fit_source_t) > 1 else 0.0
                st.session_state["last_run_source_duration"] = fit_source_duration / max(fit_playback_speed, 1e-6)
                _run_exact(
                    st.session_state.link,
                    fit_source_t,
                    fit_scaled_q,
                    cfg["cal_factor"],
                    fit_playback_speed,
                    fit_effective_max_rpm,
                    manage_sensor_collection=False,
                    analyze_after=False,
                    auto_tare=False,
                )
                if fit_source_label:
                    st.info(
                        f"Known-volume playback finished for {fit_source_label}. "
                        "Leave recording running until drain-down is complete, then press Stop Fit Recording."
                    )
                else:
                    st.info(
                        "Known-volume playback finished. Leave recording running until drain-down is complete, "
                        "then press Stop Fit Recording."
                    )
            except Exception as exc:
                sensor.stop_collecting()
                st.session_state["fit_recording"] = False
                st.error(f"Calibration playback failed: {exc}")

        if frc2.button("Stop Fit Recording", use_container_width=True, key="fit_record_stop"):
            sensor.stop_collecting()
            st.session_state["fit_recording"] = False

        is_fit_recording = st.session_state.get("fit_recording", False)
        if is_fit_recording:
            elapsed = time.time() - st.session_state.get("fit_record_start_time", time.time())
            fit_t_data, fit_m_data, _ = sensor.get_data()
            st.caption(f"Fit recording in progress: {elapsed:.0f} s, {len(fit_t_data)} samples")
            if fit_m_data:
                st.metric("Current fit-recording mass", f"{fit_m_data[-1]:.1f} g")
                if len(fit_m_data) >= 6:
                    tail_delta = fit_m_data[-1] - fit_m_data[-6]
                    if tail_delta < -0.5:
                        st.info(f"Drain-down detected ({tail_delta:.1f} g over recent samples)")
                    elif tail_delta > 0.5:
                        st.info(f"Mass still increasing (+{tail_delta:.1f} g over recent samples)")
                    else:
                        st.info("Mass is relatively stable right now")

            if fit_t_data and fit_m_data:
                fit_fig = go.Figure()
                fit_fig.add_trace(go.Scatter(x=fit_t_data, y=fit_m_data, name="Mass", line=dict(color="#009688", width=2)))
                fit_fig.update_layout(xaxis_title="Time (s)", yaxis_title="Mass (g)", height=260, margin=dict(t=20))
                st.plotly_chart(fit_fig, use_container_width=True)

        if st.button("Fit Map From Current Recording", use_container_width=True, key="fit_est_map"):
            _fit_estimated_calibration_map_from_recording(
                expected_volume_ml=fit_expected_volume_ml,
                reference_mass_g=fit_reference_mass_g,
                required_tail_seconds=fit_tail_seconds,
            )
        if is_fit_recording:
            time.sleep(0.5)
            st.rerun()

        # ── multi-run calibration ──────────────────────────────────────────
        st.divider()
        st.subheader("Multi-Run Calibration (Best Accuracy)")
        st.caption(
            "Upload sensor CSVs from runs at different known volumes (e.g. 100 mL, 400 mL, 600 mL). "
            "The map is fit to minimize error across ALL volumes simultaneously — far more accurate "
            "than single-run fitting, especially for small volumes."
        )
        st.session_state.setdefault("multi_cal_runs", [])
        st.session_state.setdefault("run_queue", [])

        mr_ref_mass = st.number_input(
            "Reference mass (g)",
            value=float(st.session_state.get("multi_cal_ref_mass", 250.0)),
            min_value=10.0, max_value=2000.0, step=10.0,
            key="multi_cal_ref_mass",
            help="Set to the largest typical peak mass you expect on the sensor across all runs.",
        )

        st.markdown("**1. Queue playback CSVs to record sensor traces**")
        multi_playback_uploads = st.file_uploader(
            "Upload playback CSVs to add to the automated test queue",
            type=["csv"],
            accept_multiple_files=True,
            key="multi_cal_playback_uploads",
        )
        mp1, mp2 = st.columns(2)
        if mp1.button("Add Playback CSVs To Run Queue", key="multi_add_to_queue", use_container_width=True):
            if not multi_playback_uploads:
                st.warning("Upload at least one playback CSV first.")
            else:
                existing_names = {item["name"] for item in st.session_state["run_queue"]}
                added = 0
                for queue_file in multi_playback_uploads:
                    if queue_file.name in existing_names:
                        continue
                    try:
                        queue_df = pd.read_csv(queue_file)
                        queue_t_col = _find_col(queue_df, ["time_s", "time"])
                        queue_q_col = _find_col(queue_df, ["flow_ml_s", "flow"])
                        if queue_t_col is None or queue_q_col is None:
                            st.warning(f"{queue_file.name}: missing time_s and flow_ml_s columns")
                            continue
                        queue_df = queue_df.rename(columns={queue_t_col: "time_s", queue_q_col: "flow_ml_s"})
                        st.session_state["run_queue"].append({"name": queue_file.name, "df": queue_df})
                        existing_names.add(queue_file.name)
                        added += 1
                    except Exception as exc:
                        st.warning(f"{queue_file.name}: parse error: {exc}")
                st.success(f"Added {added} playback CSV(s) to the Run Automated Test queue.")
        if mp2.button("Load Queue Result CSVs Here", key="multi_load_queue_results", use_container_width=True):
            queue_results = st.session_state.get("queue_all_results", [])
            loaded = 0
            for res in queue_results:
                csv_bytes = res.get("csv")
                if not csv_bytes:
                    continue
                try:
                    csv_text = csv_bytes.decode(errors="replace")
                    expected_ml = _expected_volume_from_csv_text(csv_text)
                    df_up = pd.read_csv(io.StringIO(csv_text), comment="#")
                    _append_multi_cal_run(res["name"], df_up, expected_ml=expected_ml)
                    loaded += 1
                except Exception as exc:
                    st.warning(f"{res.get('name', 'queue result')}: could not load result CSV ({exc})")
            if loaded:
                st.success(f"Loaded {loaded} queue result CSV(s) into multi-run calibration.")
            else:
                st.warning("No queue result CSVs were available to load.")
        if st.session_state["run_queue"]:
            st.caption(f"Queued playback files available on Run Automated Test page: {len(st.session_state['run_queue'])}")

        st.markdown("**2. Upload sensor recording/result CSVs for fitting**")

        multi_uploads = st.file_uploader(
            "Upload sensor recording CSVs (one per test volume)",
            type=["csv"], accept_multiple_files=True, key="multi_cal_uploads"
        )
        if multi_uploads:
            for uf in multi_uploads:
                try:
                    csv_text = uf.read().decode(errors="replace")
                    expected_ml = _expected_volume_from_csv_text(csv_text)
                    df_up = pd.read_csv(io.StringIO(csv_text), comment="#")
                    _append_multi_cal_run(uf.name, df_up, expected_ml=expected_ml)
                except Exception as exc:
                    st.warning(f"Could not parse {uf.name}: {exc}")

        if st.session_state["multi_cal_runs"]:
            st.markdown("**Set expected volume for each run:**")
            for i, run in enumerate(st.session_state["multi_cal_runs"]):
                mc1, mc2 = st.columns([3, 1])
                mc1.text(run["name"])
                run["expected_ml"] = mc2.number_input(
                    "mL", value=float(run["expected_ml"]) or 400.0,
                    min_value=1.0, max_value=5000.0, step=10.0,
                    key=f"multi_cal_vol_{i}_{run['name']}",
                    label_visibility="collapsed",
                )
            mcc1, mcc2 = st.columns(2)
            if mcc1.button("Clear Runs", key="multi_cal_clear"):
                st.session_state["multi_cal_runs"] = []
                st.rerun()
            if mcc2.button("Fit Multi-Run Map", type="primary", key="fit_multi_run_map",
                           disabled=len(st.session_state["multi_cal_runs"]) < 2):
                _fit_multi_run_calibration_map(st.session_state["multi_cal_runs"], mr_ref_mass)
        else:
            st.info("Upload at least 2 sensor recordings to enable multi-run fitting.")

        # Load calibration from CSV
        st.divider()
        st.subheader("Load Calibration from CSV")

        load_method = st.radio("Source", ["AutoFlow export (auto-find)", "Upload file"], horizontal=True, key="cal_load_method")

        if load_method == "AutoFlow export (auto-find)":
            csvs = _find_autoflow_csvs()
            combined = [f for f in csvs if "combined_export" in f.name]
            if combined:
                names = [f"{f.name}  ({f.parent.name}/)" for f in combined]
                idx = st.selectbox("Calibration file", range(len(names)), format_func=lambda i: names[i], key="cal_csv_select")
                if st.button("Load Calibration Map", key="load_cal_map"):
                    _load_calibration_from_autoflow_csv(combined[idx])
            else:
                st.info("No AutoFlow combined CSVs found.")
        else:
            uploaded = st.file_uploader("Upload calibration CSV", type=["csv"], key="cal_upload")
            if uploaded:
                _load_calibration_from_upload(uploaded)

    # ── calibration map viewer ─────────────────────────────────────────
    with tab_map:
        cal_map = st.session_state.get("calibration_map", [[], []])
        if cal_map[0] and cal_map[1]:
            masses = cal_map[0]
            rates = cal_map[1]

            cols = st.columns(3)
            cols[0].metric("Points", len(masses))
            cols[1].metric("Mass range", f"{min(masses):.1f} – {max(masses):.1f} g")
            cols[2].metric("Rate range", f"{min(rates):.4f} – {max(rates):.4f} g/s")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=masses, y=rates, name="Drain Rate",
                mode="lines", line=dict(color="#2196F3", width=2),
                fill="tozeroy",
            ))
            fig.update_layout(
                title="Calibration Map",
                xaxis_title="Mass (g)", yaxis_title="Drain Rate (g/s)",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Raw data"):
                df = pd.DataFrame({"mass_g": masses, "drain_rate_g_per_s": rates})
                st.dataframe(df, use_container_width=True)
        else:
            st.info("No calibration map loaded. Calibrate the sensor or load from CSV.")


def _build_calibration_map(method="Fill then drain", target_start_mass_g=None):
    """Build drain-rate lookup table from a fill-then-drain mass signal.

    Matches the Flutter app: produces a [masses, drain_rates] map sorted by
    descending mass so _corresponding_drain_rate() can look up the passive
    outflow rate at any given fill level.  Drain rates are stored as negative
    values (mass leaving the container) consistent with the Flutter convention.
    """
    sensor = st.session_state.sensor
    t_data, m_data, _ = sensor.get_data()
    if len(t_data) < 20:
        st.error("Not enough data (need at least 20 samples). Run calibration for longer.")
        return

    t = np.array(t_data, dtype=float)
    y = np.array(m_data, dtype=float)

    from analysis import lowpass_filter, get_derivative

    filt_mass = lowpass_filter(y)

    total_drop = float(filt_mass.max() - filt_mass.min())
    if total_drop < 10.0:
        st.error(
            f"Mass only changed {total_drop:.1f} g — no significant drain detected. "
            "For pump-only setups with no passive drain press **Set Zero Drain** instead."
        )
        return

    # Both calibration methods ultimately need the pure post-pour drain segment.
    # In "Drain open from start" mode the user pours up to a target mass with the
    # drain already open, so the local peak still marks the handoff from fill to
    # passive drain even though there was never a plugged hold phase.
    max_idx = int(np.argmax(filt_mass))
    max_idx = min(max_idx + 20, len(filt_mass) - 2)

    if method == "Drain open from start" and target_start_mass_g is not None:
        peak_mass = float(filt_mass[int(np.argmax(filt_mass))])
        if peak_mass < float(target_start_mass_g) * 0.60:
            st.warning(
                f"Observed peak mass was only {peak_mass:.1f} g against a target of "
                f"{float(target_start_mass_g):.0f} g. The calibration map was built from the "
                "available drain segment, but the target was not reached."
            )

    search_end = len(filt_mass) - 1
    if max_idx >= search_end:
        st.error("Peak is at the very end of the recording — stop pouring earlier and let it drain for several seconds.")
        return

    min_idx = max_idx + int(np.argmin(filt_mass[max_idx:search_end]))
    if min_idx <= max_idx + 5:
        st.error("Drain window too short. Let water drain for several seconds before pressing Stop.")
        return

    deriv = get_derivative(t, filt_mass, ss=2)
    filt_rate = lowpass_filter(deriv)

    # Store the full windowed segment — matches Flutter which keeps every sample
    # rather than binning.  The lookup in _corresponding_drain_rate is O(n) but
    # n is small enough (a few hundred points) that this is not a concern.
    cal_masses = filt_mass[max_idx:min_idx].tolist()
    cal_rates  = filt_rate[max_idx:min_idx].tolist()

    # Sort descending by mass (high fill → low fill) — required by _corresponding_drain_rate
    pairs = sorted(zip(cal_masses, cal_rates), key=lambda x: x[0], reverse=True)
    cal_masses = [p[0] for p in pairs]
    cal_rates  = [p[1] for p in pairs]

    st.session_state.calibration_map = [cal_masses, cal_rates]
    st.success(
        f"Lookup table built: {len(cal_masses)} points, "
        f"mass range {min(cal_masses):.1f}–{max(cal_masses):.1f} g, "
        f"drain rate {min(cal_rates):.4f}–{max(cal_rates):.4f} g/s"
    )


def _build_estimated_calibration_map(reference_mass_g, drain_rate_at_reference_g_s, exponent=0.5, points=120, drain_threshold_g=0.0):
    """Build a reusable drain map from a simple gravity-style drain model.

    drain_threshold_g: mass below which drain is considered zero (low head / drain hole
    above container floor). Improves accuracy for small-volume tests where mass barely
    exceeds the threshold during the run.
    """
    reference_mass_g = float(reference_mass_g)
    drain_rate_at_reference_g_s = float(drain_rate_at_reference_g_s)
    exponent = float(exponent)
    drain_threshold_g = float(drain_threshold_g)
    if reference_mass_g <= drain_threshold_g:
        st.error("Reference mass must be greater than drain threshold.")
        return
    if drain_rate_at_reference_g_s >= 0:
        st.error("Drain rate at reference mass must be negative.")
        return
    if exponent <= 0:
        st.error("Exponent must be positive.")
        return

    masses, rates = _estimated_map_arrays(reference_mass_g, drain_rate_at_reference_g_s, exponent, points, drain_threshold_g)
    st.session_state.calibration_map = [masses.tolist(), rates.tolist()]
    st.success(
        f"Estimated lookup table built: {len(masses)} points, "
        f"reference {reference_mass_g:.1f} g @ {drain_rate_at_reference_g_s:.2f} g/s, "
        f"exponent {exponent:.2f}, threshold {drain_threshold_g:.1f} g"
    )


def _estimated_map_arrays(reference_mass_g, drain_rate_at_reference_g_s, exponent=0.5, points=120, drain_threshold_g=0.0):
    ref = float(reference_mass_g)
    thresh = float(drain_threshold_g)
    rate_ref = float(drain_rate_at_reference_g_s)
    masses = np.linspace(ref, 0.0, int(points))
    span = max(ref - thresh, 1e-9)
    frac = np.clip((masses - thresh) / span, 0.0, None)
    rates = np.where(masses > thresh, rate_ref * (frac ** float(exponent)), 0.0)
    return masses, rates.astype(float)


def _fit_estimated_calibration_map_from_recording(expected_volume_ml, reference_mass_g, required_tail_seconds=8.0):
    sensor = st.session_state.sensor
    t_data, m_data, _ = sensor.get_data()
    if len(t_data) < 40:
        st.error("Not enough recorded data to fit a map. Record a full run first.")
        return

    t = np.asarray(t_data, dtype=float)
    y = np.asarray(m_data, dtype=float)
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    if duration < max(10.0, float(required_tail_seconds) + 2.0):
        st.error("Recording is too short. Capture the full run plus several seconds of drain-down.")
        return

    from analysis import lowpass_filter, get_derivative

    filt_mass = lowpass_filter(y)
    deriv = get_derivative(t, filt_mass, ss=2)
    filt_mass_rate = lowpass_filter(deriv)

    tail_mask = t >= (t[-1] - float(required_tail_seconds))
    if np.mean(filt_mass[tail_mask]) > max(10.0, 0.15 * float(reference_mass_g)):
        st.warning(
            "The recording tail still has substantial mass on the sensor. "
            "Fit may be poor unless you stop after it has mostly drained out."
        )

    ref_mass = float(reference_mass_g)
    if ref_mass <= 0:
        st.error("Reference mass must be positive.")
        return

    # Grid search over (rate, exponent, threshold). Volume matching is primary;
    # tail inflow penalty keeps drain from being over-subtracted after the run ends.
    exponent_grid = np.linspace(0.25, 1.5, 26)
    rate_grid = np.linspace(-25.0, -0.3, 249)
    threshold_grid = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0]

    drain_mask = (filt_mass > 5.0) & (filt_mass_rate < -0.1)
    best = None
    best_payload = None

    progress_bar = st.progress(0.0, text="Fitting calibration map...")
    total = len(threshold_grid) * len(exponent_grid) * len(rate_grid)
    step = 0

    for thresh in threshold_grid:
        if ref_mass <= thresh:
            continue
        span = ref_mass - thresh
        for exponent in exponent_grid:
            for rate_ref in rate_grid:
                step += 1
                if step % 2000 == 0:
                    progress_bar.progress(min(step / total, 1.0), text=f"Fitting… {step}/{total}")

                frac = np.clip((filt_mass - thresh) / span, 0.0, None)
                drain_curve = np.where(filt_mass > thresh, rate_ref * (frac ** exponent), 0.0)
                inflow = np.maximum(0.0, filt_mass_rate - drain_curve)

                total_ml = float(np.trapezoid(inflow, t)) / 0.9982
                vol_err = abs(total_ml - float(expected_volume_ml)) / max(float(expected_volume_ml), 1e-6)
                tail_inflow_penalty = float(np.mean(inflow[tail_mask])) if np.any(tail_mask) else 0.0
                drain_shape_penalty = 0.0
                if np.any(drain_mask):
                    drain_shape_penalty = float(np.mean((drain_curve[drain_mask] - filt_mass_rate[drain_mask]) ** 2))

                score = (6.0 * vol_err) + (0.15 * tail_inflow_penalty) + (0.02 * drain_shape_penalty)
                if best is None or score < best:
                    best = score
                    best_payload = {
                        "rate_ref": float(rate_ref),
                        "exponent": float(exponent),
                        "threshold": float(thresh),
                        "total_ml": total_ml,
                        "vol_err_pct": vol_err * 100.0,
                    }

    progress_bar.progress(1.0, text="Done.")

    if best_payload is None:
        st.error("Could not fit an estimated map from this recording.")
        return

    masses, rates = _estimated_map_arrays(ref_mass, best_payload["rate_ref"], best_payload["exponent"], drain_threshold_g=best_payload["threshold"])
    st.session_state.calibration_map = [masses.tolist(), rates.tolist()]
    st.success(
        "Estimated map fitted from recording: "
        f"drain @ {ref_mass:.0f} g = {best_payload['rate_ref']:.2f} g/s, "
        f"exponent = {best_payload['exponent']:.2f}, "
        f"threshold = {best_payload['threshold']:.0f} g, "
        f"predicted volume = {best_payload['total_ml']:.1f} mL "
        f"({best_payload['vol_err_pct']:.1f}% error)"
    )


def _expected_volume_from_csv_text(text):
    first_line = text.splitlines()[0].strip() if text.splitlines() else ""
    if not first_line.startswith("#"):
        return 0.0
    for part in first_line[1:].split(","):
        part = part.strip()
        if part.startswith("expected_volume_mL="):
            try:
                return float(part.split("=", 1)[1])
            except ValueError:
                return 0.0
    return 0.0


def _append_multi_cal_run(name, df_up, expected_ml=0.0):
    t_col = "time_s" if "time_s" in df_up.columns else df_up.columns[0]
    m_col = "mass_g" if "mass_g" in df_up.columns else df_up.columns[1]
    if any(r["name"] == name for r in st.session_state["multi_cal_runs"]):
        return
    st.session_state["multi_cal_runs"].append({
        "name": name,
        "t": df_up[t_col].to_numpy(dtype=float),
        "m": df_up[m_col].to_numpy(dtype=float),
        "expected_ml": float(expected_ml),
    })


def _append_multi_cal_run_from_csv_bytes(name, csv_bytes):
    if not csv_bytes:
        return False, "No CSV data available."
    try:
        csv_text = csv_bytes.decode(errors="replace")
        expected_ml = _expected_volume_from_csv_text(csv_text)
        df_up = pd.read_csv(io.StringIO(csv_text), comment="#")
        _append_multi_cal_run(name, df_up, expected_ml=expected_ml)
        return True, expected_ml
    except Exception as exc:
        return False, str(exc)


def _fit_multi_run_calibration_map(runs, reference_mass_g):
    """Fit drain map parameters jointly across multiple known-volume runs.

    runs: list of {"name": str, "t": np.ndarray, "m": np.ndarray, "expected_ml": float}
    Searches (rate_ref, exponent, threshold) to minimise aggregate volume error.
    """
    from analysis import lowpass_filter, get_derivative

    ref_mass = float(reference_mass_g)
    if ref_mass <= 0:
        st.error("Reference mass must be positive.")
        return

    # Preprocess each run once
    processed = []
    for run in runs:
        t = np.asarray(run["t"], dtype=float)
        m = np.asarray(run["m"], dtype=float)
        if len(t) < 20:
            st.warning(f"Run '{run['name']}' has too few samples — skipped.")
            continue
        fm = lowpass_filter(m)
        fmr = lowpass_filter(get_derivative(t, fm, ss=2))
        tail_mask = t >= (t[-1] - 8.0)
        processed.append({
            "name": run["name"],
            "t": t,
            "fm": fm,
            "fmr": fmr,
            "tail_mask": tail_mask,
            "expected_ml": float(run["expected_ml"]),
        })

    if not processed:
        st.error("No valid runs to fit.")
        return

    exponent_grid = np.linspace(0.25, 1.5, 26)
    rate_grid = np.linspace(-25.0, -0.3, 249)
    threshold_grid = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0]

    best_score = None
    best_payload = None
    total = len(threshold_grid) * len(exponent_grid) * len(rate_grid)
    prog = st.progress(0.0, text="Fitting multi-run calibration map…")
    step = 0

    for thresh in threshold_grid:
        if ref_mass <= thresh:
            continue
        span = ref_mass - thresh
        for exponent in exponent_grid:
            for rate_ref in rate_grid:
                step += 1
                if step % 2000 == 0:
                    prog.progress(min(step / total, 1.0), text=f"Fitting… {step}/{total}")

                total_vol_err = 0.0
                tail_pen = 0.0
                per_run_ml = []
                for r in processed:
                    frac = np.clip((r["fm"] - thresh) / span, 0.0, None)
                    dc = np.where(r["fm"] > thresh, rate_ref * (frac ** exponent), 0.0)
                    inflow = np.maximum(0.0, r["fmr"] - dc)
                    vol_ml = float(np.trapezoid(inflow, r["t"])) / 0.9982
                    per_run_ml.append(vol_ml)
                    total_vol_err += abs(vol_ml - r["expected_ml"]) / max(r["expected_ml"], 1e-6)
                    if np.any(r["tail_mask"]):
                        tail_pen += float(np.mean(inflow[r["tail_mask"]]))

                avg_err = total_vol_err / len(processed)
                score = (6.0 * avg_err) + (0.1 * tail_pen / len(processed))
                if best_score is None or score < best_score:
                    best_score = score
                    best_payload = {
                        "rate_ref": float(rate_ref),
                        "exponent": float(exponent),
                        "threshold": float(thresh),
                        "per_run_ml": per_run_ml,
                    }

    prog.progress(1.0, text="Done.")
    if best_payload is None:
        st.error("Could not fit a map.")
        return

    masses, rates = _estimated_map_arrays(
        ref_mass, best_payload["rate_ref"], best_payload["exponent"],
        drain_threshold_g=best_payload["threshold"]
    )
    st.session_state.calibration_map = [masses.tolist(), rates.tolist()]

    lines = []
    for r, ml in zip(processed, best_payload["per_run_ml"]):
        err_pct = abs(ml - r["expected_ml"]) / max(r["expected_ml"], 1e-6) * 100.0
        lines.append(f"- **{r['name']}**: predicted {ml:.1f} mL vs {r['expected_ml']:.0f} mL → {err_pct:.1f}% error")
    st.success(
        f"Multi-run map fitted: drain @ {ref_mass:.0f} g = {best_payload['rate_ref']:.2f} g/s, "
        f"exponent = {best_payload['exponent']:.2f}, threshold = {best_payload['threshold']:.0f} g"
    )
    for line in lines:
        st.markdown(line)


def _save_calibration_map_csv():
    cal_map = st.session_state.get("calibration_map", [[], []])
    if not cal_map[0]:
        st.warning("No calibration map to save")
        return
    df = pd.DataFrame({"mass_g": cal_map[0], "drain_rate_g_per_s": cal_map[1]})
    csv_bytes = df.to_csv(index=False).encode()

    # Save to file
    export_dir = Path.home() / "Documents" / "AutoFlow"
    export_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H-%M-%S")
    path = export_dir / f"calibration_map_{ts}.csv"
    path.write_bytes(csv_bytes)
    st.success(f"Saved to {path}")

    st.download_button("Download calibration CSV", csv_bytes, file_name=f"calibration_map_{ts}.csv")


def _load_calibration_from_autoflow_csv(path):
    """Extract calibration map section from an AutoFlow combined CSV."""
    sections = _parse_autoflow_csv(path)
    for name, df in sections.items():
        if "calibration map" in name.lower():
            m_col = _find_col(df, ["mass_g"])
            r_col = _find_col(df, ["drain_rate_g_per_s"])
            if m_col and r_col:
                masses = df[m_col].dropna().tolist()
                rates = df[r_col].dropna().tolist()
                if masses and rates:
                    st.session_state.calibration_map = [masses, rates]
                    st.success(f"Loaded {len(masses)} calibration points from {path.name}")
                    return
    st.warning("No calibration map section found in this CSV")


def _load_calibration_from_upload(uploaded):
    """Load a calibration map from an uploaded CSV."""
    try:
        df = pd.read_csv(uploaded)
        m_col = _find_col(df, ["mass_g", "mass"])
        r_col = _find_col(df, ["drain_rate_g_per_s", "drain_rate", "rate"])
        if m_col and r_col:
            masses = df[m_col].dropna().tolist()
            rates = df[r_col].dropna().tolist()
            st.session_state.calibration_map = [masses, rates]
            st.success(f"Loaded {len(masses)} calibration points")
        else:
            st.error(f"Expected columns 'mass_g' and 'drain_rate_g_per_s'. Found: {list(df.columns)}")
    except Exception as e:
        st.error(f"Parse error: {e}")


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 2 — TEST RESULTS
# ══════════════════════════════════════════════════════════════════════════

def page_results():
    st.header("Test Results")

    tab_file, tab_upload = st.tabs(["Browse files", "Upload CSV"])

    with tab_file:
        csvs = _find_autoflow_csvs()
        if not csvs:
            st.info("No AutoFlow CSVs found. Use the Upload tab or export from the Flutter app.")
            return

        names = [f"{f.name}  ({f.parent.name}/)" for f in csvs]
        idx = st.selectbox("Select a result file", range(len(names)), format_func=lambda i: names[i])
        if st.button("Load", key="load_file"):
            sections = _parse_autoflow_csv(csvs[idx])
            st.session_state["result_sections"] = sections
            st.session_state["result_name"] = csvs[idx].name

    with tab_upload:
        uploaded = st.file_uploader("Upload a CSV", type=["csv"])
        if uploaded:
            # Try AutoFlow combined format first, fall back to 2-column
            sections = _parse_autoflow_csv(uploaded)
            if sections:
                st.session_state["result_sections"] = sections
                st.session_state["result_name"] = uploaded.name
            else:
                # Try as simple 2-column CSV (time, flow) or multi-column
                uploaded.seek(0)
                try:
                    df = pd.read_csv(uploaded)
                    st.session_state["result_sections"] = {"Uploaded Data": df}
                    st.session_state["result_name"] = uploaded.name
                except Exception as e:
                    st.error(f"Could not parse CSV: {e}")

    # ── render loaded results ──────────────────────────────────────────
    sections = st.session_state.get("result_sections")
    if not sections:
        return

    st.divider()
    st.subheader(st.session_state.get("result_name", "Results"))

    # Try to find the test data section
    test_df = None
    calib_df = None
    detail_df = None
    for name, df in sections.items():
        name_lower = name.lower()
        if "detailed" in name_lower or "timeseries" in name_lower:
            detail_df = df
        elif "raw" in name_lower or "test" in name_lower:
            test_df = df
        elif "calibration map" in name_lower:
            calib_df = df

    # If we have detailed timeseries, use that
    if detail_df is not None and len(detail_df) > 5:
        _render_detailed_results(detail_df, calib_df)
    elif test_df is not None and len(test_df) > 5:
        _render_test_data(test_df, calib_df)
    else:
        # Generic: just show whatever we have
        for name, df in sections.items():
            if len(df) > 0:
                _render_generic_section(name, df)


def _render_detailed_results(df, calib_df):
    """Render a detailed timeseries from AutoFlow export."""
    t_col = _find_col(df, ["time_s", "time"])
    mass_col = _find_col(df, ["rawMass_g", "rawMass", "mass_g", "mass"])

    if t_col is None:
        st.warning("Could not find time column")
        return

    if mass_col is None:
        _render_generic_section("Detailed Results", df)
        return

    t = df[t_col].to_numpy(dtype=float)
    mass = df[mass_col].to_numpy(dtype=float)

    cal_map = [[], []]
    if calib_df is not None:
        m_col = _find_col(calib_df, ["mass_g"])
        r_col = _find_col(calib_df, ["drain_rate_g_per_s"])
        if m_col and r_col:
            cal_map = [
                calib_df[m_col].to_numpy(dtype=float).tolist(),
                calib_df[r_col].to_numpy(dtype=float).tolist(),
            ]
    if not cal_map[0]:
        cal_map = st.session_state.get("calibration_map", [[], []])

    filt_mass, filt_inflow, kz_flow, cum_volume, empty, voiding, draining, roi = compute_flow_from_mass(t, mass, cal_map)
    results = {
        "t_arr": np.asarray(t, dtype=float).tolist(),
        "raw_mass": np.asarray(mass, dtype=float).tolist(),
        "filt_mass": np.asarray(filt_mass, dtype=float).tolist(),
        "filt_inflow": np.asarray(filt_inflow, dtype=float).tolist(),
        "kz_flow": np.asarray(kz_flow, dtype=float).tolist(),
        "cum_volume": np.asarray(cum_volume, dtype=float).tolist(),
        "calibration_map": cal_map,
        "empty": empty,
        "voiding": voiding,
        "draining": draining,
        "roi": roi,
        "saved_at": time.time(),
    }

    _render_saved_run_analysis(results)

    with st.expander("Raw data"):
        st.dataframe(df, use_container_width=True)


def _render_test_data(df, calib_df):
    """Render raw test data (time_s, mass_g) with full analysis pipeline."""
    t_col = _find_col(df, ["time_s", "time"])
    mass_col = _find_col(df, ["mass_g", "mass", "rawMass_g", "rawMass"])

    if t_col is None or mass_col is None:
        _render_generic_section("Test Data", df)
        return

    t = df[t_col].to_numpy(dtype=float)
    y = df[mass_col].to_numpy(dtype=float)

    # Build calibration map: prefer CSV section, fall back to session state
    cal_map = [[], []]
    if calib_df is not None:
        m_col = _find_col(calib_df, ["mass_g"])
        r_col = _find_col(calib_df, ["drain_rate_g_per_s"])
        if m_col and r_col:
            cal_map = [
                calib_df[m_col].to_numpy(dtype=float).tolist(),
                calib_df[r_col].to_numpy(dtype=float).tolist(),
            ]
    if not cal_map[0]:
        cal_map = st.session_state.get("calibration_map", [[], []])

    # Run the full analysis pipeline
    filt_mass, filt_inflow, kz_flow, cum_volume, empty, voiding, draining, roi = compute_flow_from_mass(t, y, cal_map)

    # Stats
    cols = st.columns(4)
    duration = t[-1] - t[0] if len(t) > 1 else 0
    cols[0].metric("Duration", f"{duration:.1f} s")
    cols[1].metric("Peak Flow", f"{np.max(kz_flow):.2f} g/s")
    total_vol = cum_volume[-1] if len(cum_volume) > 0 else 0
    cols[2].metric("Volume", f"{total_vol:.1f} g ({total_vol / 0.9982:.0f} mL)")
    cols[3].metric("Zones", f"V:{len(voiding)} D:{len(draining)} E:{len(empty)}")

    def _add_zone_rects(fig, zones, color):
        for z in zones:
            s, e = z[0], min(z[1], len(t) - 1)
            if s < len(t) and e < len(t) and e >= s:
                fig.add_vrect(x0=t[s], x1=t[e], fillcolor=color, line_width=0)

    # Mass + zones graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t[:len(filt_mass)], y=filt_mass, name="Mass", line=dict(color="#2196F3", width=2)))
    _add_zone_rects(fig, empty, "rgba(158,158,158,0.10)")
    _add_zone_rects(fig, draining, "rgba(255,152,0,0.18)")
    _add_zone_rects(fig, voiding, "rgba(76,175,80,0.18)")
    if len(roi) == 2 and len(t):
        rs = max(0, min(int(roi[0]), len(t) - 1))
        re = max(0, min(int(roi[1]), len(t) - 1))
        fig.add_vline(x=t[rs], line_dash="dash", line_color="#616161")
        fig.add_vline(x=t[re], line_dash="dash", line_color="#616161")
    fig.update_layout(title="Mass vs Time (with zones)", xaxis_title="Time (s)", yaxis_title="Mass (g)", height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Filtered inflow
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=t[:len(filt_inflow)], y=filt_inflow, name="Filtered Inflow", line=dict(color="#5E35B1", width=2)))
    _add_zone_rects(fig2, empty, "rgba(158,158,158,0.10)")
    _add_zone_rects(fig2, draining, "rgba(255,152,0,0.18)")
    _add_zone_rects(fig2, voiding, "rgba(76,175,80,0.18)")
    fig2.update_layout(title="Filtered Inflow", xaxis_title="Time (s)", yaxis_title="Flow (g/s)", height=350, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    # KZ flow
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=t[:len(kz_flow)], y=kz_flow, name="KZ Flow", line=dict(color="#E91E63", width=2.5)))
    _add_zone_rects(fig3, empty, "rgba(158,158,158,0.10)")
    _add_zone_rects(fig3, draining, "rgba(255,152,0,0.18)")
    _add_zone_rects(fig3, voiding, "rgba(76,175,80,0.18)")
    fig3.update_layout(title="KZ-Filtered Flow Rate", xaxis_title="Time (s)", yaxis_title="Flow (g/s)", height=350, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

    # Flow vs mass
    fig4 = go.Figure()
    n_fm = min(len(filt_mass), len(kz_flow))
    fig4.add_trace(go.Scatter(x=filt_mass[:n_fm], y=kz_flow[:n_fm], name="Flow vs Mass", line=dict(color="#AB47BC", width=2), fill="tozeroy"))
    fig4.update_layout(title="Flow Rate vs Mass", xaxis_title="Mass (g)", yaxis_title="Flow (g/s)", height=350, showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)

    # Cumulative volume
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=t[:len(cum_volume)], y=cum_volume, name="Volume", line=dict(color="#4CAF50", width=2), fill="tozeroy"))
    fig5.update_layout(title="Cumulative Volume", xaxis_title="Time (s)", yaxis_title="Volume (g)", height=350, showlegend=False)
    st.plotly_chart(fig5, use_container_width=True)


def _render_generic_section(name, df):
    st.subheader(name)
    # Try to auto-plot if there are numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) >= 2:
        fig = go.Figure()
        x_col = num_cols[0]
        for col in num_cols[1:]:
            fig.add_trace(go.Scatter(x=df[x_col], y=df[col], name=col, mode="lines"))
        fig.update_layout(xaxis_title=x_col, height=350)
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("Data table"):
        st.dataframe(df, use_container_width=True)


def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # Case-insensitive fallback
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 2 — RUN TEST
# ══════════════════════════════════════════════════════════════════════════

# INSERT _wait_for_drain FUNCTION HERE: used by the run queue between tests.
def _wait_for_drain(sensor, timeout_s=120, threshold_g=15, stable_s=8):
    """Wait until live mass remains below threshold for a stable interval."""
    progress = st.progress(0.0, text="Waiting for drain...")
    status = st.empty()
    start = time.time()
    stable_start = None

    while True:
        elapsed = time.time() - start
        mass = float(sensor.current_reading)

        if mass < threshold_g:
            if stable_start is None:
                stable_start = time.time()
            stable_elapsed = time.time() - stable_start
        else:
            stable_start = None
            stable_elapsed = 0.0

        progress.progress(
            min(1.0, elapsed / max(timeout_s, 1e-6)),
            text=f"Waiting for drain: {mass:.1f} g, stable {stable_elapsed:.1f}/{stable_s:.1f}s",
        )
        status.info(
            f"Waiting for drain below {threshold_g:.1f} g "
            f"({stable_elapsed:.1f}/{stable_s:.1f}s stable, {elapsed:.1f}/{timeout_s:.1f}s elapsed)"
        )

        if stable_elapsed >= stable_s:
            progress.progress(1.0, text="Drain complete")
            status.success("Drain complete")
            return True
        if elapsed >= timeout_s:
            progress.empty()
            status.warning("Drain wait timed out")
            return False

        time.sleep(0.5)


def page_run():
    st.header("Run Automated Test")

    cfg = st.session_state.cfg
    link = st.session_state.link
    sensor = st.session_state.sensor
    if "pump_link" not in st.session_state:
        st.session_state.pump_link = link

    if not link.is_open():
        st.warning("Connect to the pump in the sidebar first.")

    # ── input: CSV or template ─────────────────────────────────────────
    source = st.radio("Input source", ["Upload CSV curve", "Use template"], horizontal=True)

    source_t = None
    source_q = None
    source_name = None

    if source == "Upload CSV curve":
        f = st.file_uploader("CSV with columns: time_s, flow_ml_s", type=["csv"], key="run_csv")
        if f:
            try:
                df = pd.read_csv(f)
                t_col = st.selectbox("Time column", df.columns, index=0)
                q_col = st.selectbox("Flow column", df.columns, index=min(1, len(df.columns) - 1))
                source_t = df[t_col].to_numpy(dtype=float)
                source_q = df[q_col].to_numpy(dtype=float)
                source_name = f.name
            except Exception as e:
                st.error(f"Parse error: {e}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            shape = st.selectbox("Shape", SHAPES)
        with c2:
            qmax = st.number_input("Peak flow (mL/s)", value=20.0, min_value=0.1, max_value=500.0, step=0.5)
        with c3:
            volume = st.number_input("Volume (mL)", value=250.0, min_value=10.0, max_value=2000.0, step=10.0)
        with c4:
            duration = st.number_input("Duration (s)", value=25.0, min_value=5.0, max_value=600.0, step=1.0)

        profile, profile_err = build_run_profile(shape, qmax, volume, duration, n=200)
        if profile_err:
            st.error(profile_err)
            source_t = None
            source_q = None
        else:
            source_t = np.asarray(profile["u"], dtype=float) * duration
            source_q = np.asarray(profile["q"], dtype=float)
            source_name = f"{shape}_{int(round(volume))}mL_{int(round(duration))}s.csv"

    with st.expander("Test Queue"):
        st.session_state.setdefault("run_queue", [])
        st.session_state.setdefault("run_queue_upload_rev", 0)

        # Key rotates on Clear so the uploader widget resets and doesn't re-add files
        queue_files = st.file_uploader(
            "Queue CSV files with columns: time_s, flow_ml_s",
            type=["csv"],
            accept_multiple_files=True,
            key=f"run_queue_csvs_{st.session_state['run_queue_upload_rev']}",
        )
        if queue_files:
            existing_names = {item["name"] for item in st.session_state["run_queue"]}
            for queue_file in queue_files:
                if queue_file.name in existing_names:
                    continue
                try:
                    queue_df = pd.read_csv(queue_file)
                    queue_t_col = _find_col(queue_df, ["time_s", "time"])
                    queue_q_col = _find_col(queue_df, ["flow_ml_s", "flow"])
                    if queue_t_col is None or queue_q_col is None:
                        st.error(f"{queue_file.name}: missing time_s and flow_ml_s columns")
                        continue
                    queue_df = queue_df.rename(columns={queue_t_col: "time_s", queue_q_col: "flow_ml_s"})
                    st.session_state["run_queue"].append({"name": queue_file.name, "df": queue_df})
                    existing_names.add(queue_file.name)
                except Exception as e:
                    st.error(f"{queue_file.name}: parse error: {e}")

        if st.session_state["run_queue"]:
            for i, item in enumerate(st.session_state["run_queue"]):
                row = st.columns([0.08, 0.66, 0.13, 0.13])
                row[0].write(f"{i + 1}.")
                row[1].write(item["name"])
                if row[2].button("Up", key=f"run_queue_up_{i}", disabled=i == 0):
                    st.session_state["run_queue"][i - 1], st.session_state["run_queue"][i] = (
                        st.session_state["run_queue"][i],
                        st.session_state["run_queue"][i - 1],
                    )
                    st.rerun()
                if row[3].button("Down", key=f"run_queue_down_{i}", disabled=i == len(st.session_state["run_queue"]) - 1):
                    st.session_state["run_queue"][i + 1], st.session_state["run_queue"][i] = (
                        st.session_state["run_queue"][i],
                        st.session_state["run_queue"][i + 1],
                    )
                    st.rerun()
        else:
            st.info("Queue is empty")

        speed = st.slider("Queue playback speed", 0.25, 4.0, 1.0, 0.25, key="run_queue_speed")
        qc1, qc2 = st.columns(2)
        if qc1.button("Clear Queue", use_container_width=True):
            st.session_state["run_queue"] = []
            st.session_state["queue_all_results"] = []
            st.session_state["run_queue_upload_rev"] += 1  # resets file uploader widget
            st.rerun()

        effective_max_rpm = min(float(cfg.get("pump_max_rpm", PUMP_MAX_RPM)), float(PUMP_MAX_RPM))
        pump_ready = st.session_state.pump_link.is_open()
        _age = sensor.last_packet_age
        sensor_ready = sensor.is_open() and _age is not None and _age < 3.0
        run_queue_disabled = not pump_ready or not sensor_ready or not st.session_state["run_queue"]
        if qc2.button("Run Queue", type="primary", use_container_width=True, disabled=run_queue_disabled):
            link = st.session_state.pump_link
            sensor = st.session_state.sensor
            cfg = st.session_state.cfg
            total = len(st.session_state["run_queue"])
            queue_all_results = list(st.session_state.get("queue_all_results", []))
            for idx, item in enumerate(st.session_state["run_queue"], start=1):
                st.info(f"Running test {idx} of {total}: {item['name']}")
                try:
                    queue_source_t = item["df"]["time_s"].to_numpy(dtype=float)
                    queue_source_q = np.clip(item["df"]["flow_ml_s"].to_numpy(dtype=float), 0.0, None)
                    queue_peak_rpm = float(np.max(queue_source_q)) / cfg["cal_factor"] if cfg["cal_factor"] > 0 and len(queue_source_q) else 0.0
                    queue_profile_scale = 1.0
                    if queue_peak_rpm > effective_max_rpm + 1e-6 and queue_peak_rpm > 0:
                        queue_profile_scale = effective_max_rpm / queue_peak_rpm
                    queue_scaled_q = queue_source_q * queue_profile_scale
                    queue_source_duration = float(queue_source_t[-1] - queue_source_t[0]) if len(queue_source_t) > 1 else 0.0
                    st.session_state["last_run_expected_volume"] = float(np.trapezoid(queue_scaled_q, queue_source_t)) / max(speed, 1e-6)
                    st.session_state["last_run_source_duration"] = queue_source_duration / max(speed, 1e-6)
                    # Clear stale CSV so a failed run never poisons the result list
                    st.session_state["last_run_csv"] = None
                    st.session_state["pending_multi_fit_run_name"] = (
                        item["name"] if st.session_state.get("capture_run_for_multi_fit") else None
                    )
                    # _run_exact already calls _display_sensor_analysis internally
                    _run_exact(link, queue_source_t, queue_scaled_q, cfg["cal_factor"], speed, effective_max_rpm)
                    # Collect result only if analysis succeeded
                    if st.session_state.get("run_analysis_status") == "Graphs ready":
                        run_csv = st.session_state.get("last_run_csv")
                        if run_csv:
                            queue_all_results.append({"name": item["name"], "csv": run_csv})
                    # Persist incrementally so partial results survive an abort
                    st.session_state["queue_all_results"] = queue_all_results
                except Exception as exc:
                    st.error(f"Test {idx} ({item['name']}) failed: {exc}")
                    queue_all_results.append({"name": item["name"], "csv": None, "error": str(exc)})
                    st.session_state["queue_all_results"] = queue_all_results
                if idx < total:
                    if not _wait_for_drain(sensor):
                        st.error("Drain timeout — queue stopped after test {idx}.")
                        break
            st.success(f"Queue complete — {len(queue_all_results)} of {total} runs finished.")

        # Show per-run download buttons if queue results are available
        if st.session_state.get("queue_all_results"):
            st.subheader("Queue Results")
            for res in st.session_state["queue_all_results"]:
                ts = int(time.time())
                qr1, qr2 = st.columns([2, 1])
                qr1.download_button(
                    f"Download {res['name']} results",
                    data=res["csv"],
                    file_name=res["name"].replace(".csv", f"_results_{ts}.csv"),
                    mime="text/csv",
                    key=f"qres_{res['name']}_{ts}",
                )
                if qr2.button(f"Add {res['name']} To Multi-Run Fit", key=f"add_qres_multi_{res['name']}_{ts}", use_container_width=True):
                    ok, payload = _append_multi_cal_run_from_csv_bytes(res["name"], res.get("csv"))
                    if ok:
                        st.success(
                            f"Added {res['name']} to multi-run calibration"
                            + (f" ({payload:.0f} mL expected)." if payload else ".")
                        )
                    else:
                        st.error(f"Could not add {res['name']} to multi-run calibration: {payload}")

    if source_t is None or source_q is None:
        return

    source_t = np.asarray(source_t, dtype=float)
    source_q = np.clip(np.asarray(source_q, dtype=float), 0.0, None)

    # ── settings ───────────────────────────────────────────────────────
    st.divider()

    st.checkbox(
        "Auto-save finished runs to Multi-Run Calibration",
        key="capture_run_for_multi_fit",
        help="When enabled, each successful run is saved directly into the multi-run calibration dataset.",
    )
    capture_result = st.session_state.get("last_multi_fit_capture_result")
    if capture_result:
        if capture_result["ok"]:
            st.caption(
                f"Most recent fit capture: {capture_result['name']}"
                + (f" ({capture_result['payload']:.0f} mL expected)." if capture_result["payload"] else ".")
            )
        else:
            st.caption(f"Most recent fit capture failed: {capture_result['name']} ({capture_result['payload']})")

    c1, c2 = st.columns(2)
    with c1:
        mode = st.radio("Execution mode", ["Exact playback", "Template fit"], horizontal=True)
    with c2:
        speed = st.slider("Playback speed", 0.25, 4.0, 1.0, 0.25)

    source_duration = float(source_t[-1] - source_t[0]) if len(source_t) > 1 else 0
    source_qmax = float(np.max(source_q)) if len(source_q) else 0
    source_volume = float(np.trapezoid(source_q, source_t)) if len(source_t) > 1 else 0
    playback_duration = source_duration / max(speed, 1e-6)
    peak_rpm = source_qmax / cfg["cal_factor"] if cfg["cal_factor"] > 0 else 0

    # Effective RPM ceiling = the lower of the firmware hard-stop and the
    # user-configured hardware cap (e.g. pump motor limited to 350 RPM).
    effective_max_rpm = min(float(cfg.get("pump_max_rpm", PUMP_MAX_RPM)), float(PUMP_MAX_RPM))

    # Stats
    cols = st.columns(4)
    cols[0].metric("Duration", f"{playback_duration:.1f} s")
    cols[1].metric("Peak flow", f"{source_qmax:.2f} mL/s")
    cols[2].metric("Volume", f"{source_volume:.0f} mL")
    cols[3].metric("Peak RPM", f"{peak_rpm:.0f}")

    # Auto-scale the profile if peak RPM exceeds hardware cap (preserves shape)
    profile_scale = 1.0
    if peak_rpm > effective_max_rpm + 1e-6 and peak_rpm > 0:
        profile_scale = effective_max_rpm / peak_rpm

    scaled_q = source_q * profile_scale
    scaled_qmax = source_qmax * profile_scale
    scaled_peak_rpm = peak_rpm * profile_scale

    # Warnings / gate checks
    run_disabled = not link.is_open()
    if not sensor.is_open():
        st.error("Sensor is not connected. Connect the sensor in the sidebar before starting a test.")
        run_disabled = True
    if profile_scale < 1.0 - 1e-6:
        effective_volume = float(np.trapezoid(scaled_q, source_t))
        st.warning(
            f"Profile peak ({peak_rpm:.0f} RPM) exceeds pump cap ({effective_max_rpm:.0f} RPM). "
            f"Auto-scaling to {profile_scale:.0%} — peak flow {scaled_qmax:.2f} mL/s, "
            f"effective volume {effective_volume:.0f} mL. Shape is preserved."
        )
    elif 0 < scaled_peak_rpm < MIN_RPM_THRESHOLD:
        st.error(f"Peak RPM ({scaled_peak_rpm:.1f}) below minimum ({MIN_RPM_THRESHOLD}). Increase flow or adjust cal_factor.")
        run_disabled = True

    # Preview chart — show scaled profile so user sees what will actually run
    fig = go.Figure()
    preview_t = (source_t - source_t[0]) / max(speed, 1e-6) if speed != 1.0 else source_t - source_t[0]
    if profile_scale < 1.0 - 1e-6:
        fig.add_trace(go.Scatter(x=preview_t, y=source_q, name="Original", line=dict(color="#9E9E9E", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=preview_t, y=scaled_q, name=f"Scaled ({profile_scale:.0%})", line=dict(color="#E91E63", width=2)))
    else:
        fig.add_trace(go.Scatter(x=preview_t, y=source_q, name="Planned output", line=dict(color="#E91E63", width=2)))
    fig.update_layout(xaxis_title="Time (s)", yaxis_title="Flow (mL/s)", height=350)
    st.plotly_chart(fig, use_container_width=True)

    status_text = st.session_state.get("run_analysis_status", "Idle")
    if status_text == "Idle":
        st.caption("Run status: idle")
    elif status_text == "Running automated test...":
        st.info("Run status: automated test is currently running.")
    elif status_text == "Collecting and analyzing sensor data...":
        st.warning("Run status: pump run finished, building analysis graphs now. Please wait...")
    elif status_text == "Graphs ready":
        st.success("Run status: graphs are ready below.")
    elif status_text == "No sensor analysis available":
        st.warning("Run status: the pump run finished, but no usable sensor data was captured for analysis graphs.")
    else:
        st.info(f"Run status: {status_text}")

    # ── run buttons ────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    run_btn = c1.button("Run", type="primary", use_container_width=True, disabled=run_disabled)
    abort_btn = c2.button("Abort", use_container_width=True)
    clear_results_btn = c3.button("Clear Last Results", use_container_width=True)

    if clear_results_btn:
        st.session_state["last_run_analysis"] = None
        st.session_state["run_analysis_status"] = "Idle"
        st.info("Cleared saved run results")

    if abort_btn and link.is_open():
        stopped = link.hard_stop("manual abort button")
        st.session_state["run_analysis_status"] = "Idle"
        if stopped:
            st.warning("Abort/stop sequence sent")
        else:
            st.error("Abort failed to send")

    if run_btn:
        # Expected volume accounts for auto-scaling and speed
        scaled_volume = float(np.trapezoid(scaled_q, source_t))
        st.session_state["last_run_expected_volume"] = scaled_volume / max(speed, 1e-6)
        # Playback wall-clock duration — used to fix the overlay X-axis range
        st.session_state["last_run_source_duration"] = source_duration / max(speed, 1e-6)
        st.session_state["pending_multi_fit_run_name"] = (
            source_name or f"run_{int(time.time())}.csv"
        ) if st.session_state.get("capture_run_for_multi_fit") else None
        if mode == "Exact playback":
            _run_exact(link, source_t, scaled_q, cfg["cal_factor"], speed, effective_max_rpm)
        else:
            # Detect shape for template run
            fit = analyze_curve(source_t, source_q)
            sh = fit["shape"] if fit else "bell"
            dur = source_duration / speed
            _run_template(link, sh, source_qmax, source_volume, dur)

    saved_results = st.session_state.get("last_run_analysis")
    if saved_results is not None:
        st.divider()
        st.subheader("Last Automated Test Results")
        st.caption("These are the most recently captured sensor-analysis graphs from the automated test page.")
        add_last_fit_col, _ = st.columns([1, 3])
        with add_last_fit_col:
            if st.button("Add Last Run To Multi-Run Fit", key="add_last_run_multi_fit", use_container_width=True):
                run_name = f"last_run_{int(saved_results.get('saved_at', 0))}.csv"
                ok, payload = _append_multi_cal_run_from_csv_bytes(
                    run_name,
                    st.session_state.get("last_run_csv"),
                )
                if ok:
                    st.success(
                        f"Added last run to multi-run calibration"
                        + (f" ({payload:.0f} mL expected)." if payload else ".")
                    )
                else:
                    st.error(f"Could not add last run to multi-run calibration: {payload}")
        _render_saved_run_analysis(saved_results)


def _run_exact(link, t, q, cal_factor, speed_mult, max_rpm=None, manage_sensor_collection=True, analyze_after=True, auto_tare=True):
    """Stream the source curve point by point to the pump, with simultaneous sensor collection."""
    if max_rpm is None:
        max_rpm = PUMP_MAX_RPM

    if not link.is_open():
        st.error("Pump is not connected")
        st.session_state["run_analysis_status"] = "Idle"
        return

    st.session_state["run_analysis_status"] = "Running automated test..."
    st.session_state["last_run_analysis"] = None
    st.session_state["last_run_pdf"] = None
    st.session_state["last_run_pdf_for"] = None
    sensor = st.session_state.sensor
    progress = st.progress(0.0, text="Streaming...")
    chart_slot = st.empty()
    status = st.empty()

    t = np.asarray(t, dtype=float)
    q = np.clip(np.asarray(q, dtype=float), 0.0, None)
    order = np.argsort(t)
    t, q = t[order], q[order]
    t = t - t[0]

    if len(t) < 2 or t[-1] <= 0 or cal_factor <= 0:
        st.error("Invalid source data or cal_factor")
        return

    duration = float(t[-1])
    playback_duration = duration / max(speed_mult, 1e-6)
    control_dt = min(0.05, max(0.01, playback_duration / 1000.0))

    cmd_t, cmd_q, cmd_rpm = [], [], []
    last_rpm = None
    link.drain()

    # Verify the sensor stream is live before we start.  If the last packet
    # arrived more than 3 s ago the BLE link may be stalled — warn the user
    # but still proceed (start_collecting will attempt a restart in that case).
    age = sensor.last_packet_age
    if age is None or age > 3.0:
        st.warning(
            f"Sensor stream appears stale (last packet: {'never' if age is None else f'{age:.1f}s ago'}). "
            "If no data is recorded after the run, disconnect and reconnect the sensor."
        )

    if auto_tare:
        # Auto-tare before every run so the mass baseline starts at zero.
        tare_val = sensor.tare()
        status.info(f"Tared sensor (baseline {tare_val:.1f} g). Starting run...")
        time.sleep(0.3)
    else:
        status.info("Starting run with existing sensor tare/recording state...")

    if manage_sensor_collection:
        # Start sensor collection — always call stop in finally regardless of is_open()
        # to handle BLE drops mid-run (is_open() may become False but _collecting stays True)
        sensor.start_collecting()

    start = time.time()

    try:
        while True:
            elapsed = time.time() - start
            src_elapsed = min(duration, elapsed * speed_mult)
            flow = float(np.interp(src_elapsed, t, q))
            rpm = max(0.0, min(max_rpm, flow / cal_factor))
            if rpm < MIN_RPM_THRESHOLD:
                rpm = 0.0

            if last_rpm is None or abs(rpm - last_rpm) >= RPM_WRITE_EPSILON:
                link.write_line("0" if rpm <= 0 else f"{rpm:.3f}")
                link.drain()
                last_rpm = rpm

            cmd_t.append(src_elapsed)
            cmd_q.append(flow if rpm > 0 else 0.0)
            cmd_rpm.append(rpm)

            frac = min(1.0, src_elapsed / max(duration, 1e-6))
            samp = sensor.sample_count
            progress.progress(frac, text=f"{src_elapsed:.1f}/{duration:.1f}s  flow={flow:.2f} mL/s  rpm={rpm:.0f}  samp={samp}")

            if src_elapsed >= duration:
                break
            time.sleep(control_dt)
    finally:
        if link.is_open():
            link.hard_stop("exact playback complete")
        if manage_sensor_collection:
            sensor.stop_collecting()  # Always stop — guards against BLE drop leaving _collecting=True
        progress.empty()

    status.success(f"Playback complete ({playback_duration:.1f}s wall time)")
    st.session_state["run_analysis_status"] = "Collecting and analyzing sensor data..."

    if cmd_t:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cmd_t, y=cmd_q, name="Commanded", line=dict(color="#E91E63", width=2)))
        fig.update_layout(xaxis_title="Time (s)", yaxis_title="Flow (mL/s)", height=350, title="Playback Result")
        chart_slot.plotly_chart(fig, use_container_width=True)

        df = pd.DataFrame({"time_s": cmd_t, "flow_ml_s": cmd_q, "rpm": cmd_rpm})
        csv = df.to_csv(index=False).encode()
        _ts = int(time.time())
        st.download_button("Download log CSV", csv, file_name=f"playback_{_ts}.csv", key=f"dl_log_csv_{_ts}")

    # Display pump TX/RX log
    if link.is_open():
        with st.expander("Show pump TX/RX log"):
            log_text = link.export_log_text()
            if log_text:
                st.code(log_text, language="text")
                log_bytes = log_text.encode()
                _ts2 = int(time.time())
                st.download_button("Download pump log", log_bytes, file_name=f"pump_log_{_ts2}.txt", key=f"pump_log_exact_{_ts2}")
            else:
                st.info("No pump log available")

    # Analyze collected sensor data
    if analyze_after:
        _display_sensor_analysis()


def _run_template(link, shape, qmax, volume, duration):
    """Send shape/qmax/volume/duration commands, then 'run', and poll, with simultaneous sensor collection."""
    if not link.is_open():
        st.error("Pump is not connected")
        st.session_state["run_analysis_status"] = "Idle"
        return

    st.session_state["run_analysis_status"] = "Running automated test..."
    st.session_state["last_run_analysis"] = None
    st.session_state["last_run_pdf"] = None
    st.session_state["last_run_pdf_for"] = None
    sensor = st.session_state.sensor
    link.send(f"shape {shape}")
    link.send(f"qmax {qmax}")
    link.send(f"volume {volume}")
    link.send(f"duration {duration}")

    # Auto-tare before every run so the mass baseline starts at zero.
    sensor.tare()
    time.sleep(0.3)

    # Start sensor collection before pump run — stop unconditionally in finally
    sensor.start_collecting()

    if link.is_open() and not link.write_line("run"):
        sensor.stop_collecting()
        st.error("Failed to send run command to pump")
        st.session_state["run_analysis_status"] = "Idle"
        return

    progress = st.progress(0.0, text="Running...")
    chart_slot = st.empty()
    status = st.empty()

    t_start = time.time()
    t_samples, q_samples, v_samples = [], [], []
    finished = False
    timeout = max(duration * 2.0, 30.0)

    try:
        while time.time() - t_start < timeout:
            if not link.is_open():
                break
            for line in link.drain():
                if line.startswith("t=") and "delivered=" in line:
                    vals = _parse_telemetry(line)
                    if "t" in vals and "flow" in vals:
                        t_samples.append(vals["t"])
                        q_samples.append(vals["flow"])
                        v_samples.append(vals.get("delivered", 0.0))
                        frac = min(1.0, vals.get("delivered", 0) / max(volume, 1e-6))
                        progress.progress(frac, text=f"{vals.get('delivered', 0):.1f}/{volume:.0f} mL")
                for kw, fn in [("COMPLETE", status.success), ("FAILED", status.error), ("STOPPED", status.warning)]:
                    if kw in line:
                        fn(line)
                        finished = True
                if finished:
                    break
            if finished:
                break
            time.sleep(0.2)
    finally:
        if link.is_open():
            link.hard_stop("template run finalize")
        sensor.stop_collecting()  # Always stop — guards against BLE drop leaving _collecting=True

    progress.empty()
    st.session_state["run_analysis_status"] = "Collecting and analyzing sensor data..."

    if t_samples:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_samples, y=q_samples, name="Measured", line=dict(color="#E91E63", width=2)))
        fig.update_layout(xaxis_title="Time (s)", yaxis_title="Flow (mL/s)", height=350, title="Run Result")
        chart_slot.plotly_chart(fig, use_container_width=True)

        df = pd.DataFrame({"time_s": t_samples, "flow_ml_s": q_samples, "delivered_mL": v_samples})
        csv = df.to_csv(index=False).encode()
        st.download_button("Download log CSV", csv, file_name=f"run_{int(time.time())}.csv")

    # Display pump TX/RX log
    if link.is_open():
        with st.expander("Show pump TX/RX log"):
            log_text = link.export_log_text()
            if log_text:
                st.code(log_text, language="text")
                log_bytes = log_text.encode()
                st.download_button("Download pump log", log_bytes, file_name=f"pump_log_{int(time.time())}.txt", key="pump_log_template")
            else:
                st.info("No pump log available")

    # Analyze collected sensor data
    _display_sensor_analysis()


def _parse_telemetry(line):
    parts = line.replace("=", " ").split()
    vals = {}
    i = 0
    while i < len(parts) - 1:
        key = parts[i]
        if key in ("t", "u", "flow", "rpm", "delivered"):
            tok = parts[i + 1].rstrip("smLs/")
            try:
                vals[key] = float(tok)
            except ValueError:
                pass
        i += 1
    return vals


def _build_overlay_figure(results):
    """Single overlaid chart matching Flutter's GraphWidget layout."""
    t_arr = np.asarray(results["t_arr"], dtype=float)
    filt_mass = np.asarray(results["filt_mass"], dtype=float)
    kz_flow = np.asarray(results["kz_flow"], dtype=float)
    empty = results["empty"]
    voiding = results["voiding"]
    draining = results["draining"]
    roi = results.get("roi", [0, max(0, len(t_arr) - 1)])

    fig = go.Figure()

    def _vrect(zones, color):
        for z in zones:
            s = max(0, min(int(z[0]), len(t_arr) - 1))
            e = max(0, min(int(z[1]), len(t_arr) - 1))
            if len(t_arr) and e >= s:
                fig.add_vrect(x0=t_arr[s], x1=t_arr[e], fillcolor=color, line_width=0)

    # ROI highlight
    if len(roi) == 2 and len(t_arr):
        rs = max(0, min(int(roi[0]), len(t_arr) - 1))
        re = max(0, min(int(roi[1]), len(t_arr) - 1))
        fig.add_vrect(x0=t_arr[rs], x1=t_arr[re], fillcolor="rgba(135,206,250,0.15)", line_width=0)

    _vrect(empty,    "rgba(158,158,158,0.12)")
    _vrect(draining, "rgba(255,152,0,0.22)")
    _vrect(voiding,  "rgba(76,175,80,0.28)")

    # ROI boundary dashed lines
    if len(roi) == 2 and len(t_arr):
        rs = max(0, min(int(roi[0]), len(t_arr) - 1))
        re = max(0, min(int(roi[1]), len(t_arr) - 1))
        fig.add_vline(x=t_arr[rs], line_dash="dash", line_color="#9E9E9E", line_width=1)
        fig.add_vline(x=t_arr[re], line_dash="dash", line_color="#9E9E9E", line_width=1)

    # Mass trace
    n_mass = min(len(t_arr), len(filt_mass))
    fig.add_trace(go.Scatter(
        x=t_arr[:n_mass], y=filt_mass[:n_mass],
        name="Mass (g)", line=dict(color="#2196F3", width=2.5),
    ))

    # KZ flow — auto-scaled so peak = 40% of mass peak (matches Flutter)
    if len(kz_flow) > 0 and np.max(kz_flow) > 0:
        mass_peak = float(np.max(filt_mass)) if len(filt_mass) > 0 and np.max(filt_mass) > 0 else 1.0
        flow_peak = float(np.max(kz_flow))
        scale = (mass_peak * 0.4) / flow_peak
        n_flow = min(len(t_arr), len(kz_flow))
        fig.add_trace(go.Scatter(
            x=t_arr[:n_flow], y=kz_flow[:n_flow] * scale,
            name=f"KZ Flow (×{scale:.2f})", line=dict(color="#E91E63", width=2),
        ))

    x_min = float(t_arr[0]) if len(t_arr) else 0.0
    x_max = float(t_arr[-1]) if len(t_arr) else 1.0
    # Override with stored source duration if available (e.g. from CSV profile)
    src_dur = results.get("source_duration")
    if src_dur and src_dur > 0:
        x_max = float(src_dur)

    fig.update_layout(
        xaxis_title="Time (s)", yaxis_title="Mass (g)",
        xaxis_range=[x_min, x_max],
        height=500,
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.05)", borderwidth=1),
        margin=dict(t=10, b=40),
    )
    return fig


def _build_run_analysis_figures(results):
    t_arr = np.asarray(results["t_arr"], dtype=float)
    raw_mass = np.asarray(results["raw_mass"], dtype=float)
    filt_mass = np.asarray(results["filt_mass"], dtype=float)
    filt_inflow = np.asarray(results.get("filt_inflow", []), dtype=float)
    kz_flow = np.asarray(results["kz_flow"], dtype=float)
    cum_volume = np.asarray(results["cum_volume"], dtype=float)
    cal_map = results.get("calibration_map", [[], []])
    empty = results["empty"]
    voiding = results["voiding"]
    draining = results["draining"]
    roi = results.get("roi", [0, max(0, len(t_arr) - 1)])

    def _add_zone_rects(fig, zones, color):
        for z in zones:
            s = max(0, min(int(z[0]), len(t_arr) - 1))
            e = max(0, min(int(z[1]), len(t_arr) - 1))
            if len(t_arr) and e >= s:
                fig.add_vrect(x0=t_arr[s], x1=t_arr[e], fillcolor=color, line_width=0)

    def _add_roi_lines(fig):
        if len(roi) == 2 and len(t_arr):
            rs = max(0, min(int(roi[0]), len(t_arr) - 1))
            re = max(0, min(int(roi[1]), len(t_arr) - 1))
            fig.add_vline(x=t_arr[rs], line_dash="dash", line_color="#616161")
            fig.add_vline(x=t_arr[re], line_dash="dash", line_color="#616161")

    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(x=t_arr[:len(raw_mass)], y=raw_mass, name="Raw Weight", line=dict(color="#1E88E5", width=2)))
    fig_raw.update_layout(title="Raw Weight vs Time", xaxis_title="Time (s)", yaxis_title="Weight (g)", height=340, showlegend=False)

    fig_kz = go.Figure()
    fig_kz.add_trace(go.Scatter(x=t_arr[:len(kz_flow)], y=kz_flow, name="KZ Flow", line=dict(color="#8E24AA", width=2.5)))
    _add_zone_rects(fig_kz, empty, "rgba(158,158,158,0.10)")
    _add_zone_rects(fig_kz, draining, "rgba(255,152,0,0.18)")
    _add_zone_rects(fig_kz, voiding, "rgba(76,175,80,0.18)")
    _add_roi_lines(fig_kz)
    fig_kz.update_layout(title="Flow Rate vs Time (KZ smoothed)", xaxis_title="Time (s)", yaxis_title="Flow Rate (g/s)", height=340, showlegend=False)

    n_fm = min(len(filt_mass), len(kz_flow))
    fig_fm = go.Figure()
    fig_fm.add_trace(go.Scatter(x=filt_mass[:n_fm], y=kz_flow[:n_fm], name="Flow vs Mass", mode="lines", line=dict(color="#AB47BC", width=2), fill="tozeroy"))
    fig_fm.update_layout(title="Flow Rate vs Mass", xaxis_title="Mass (g)", yaxis_title="Flow Rate (g/s)", height=340, showlegend=False)

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=t_arr[:len(cum_volume)], y=cum_volume, name="Cumulative Volume", line=dict(color="#43A047", width=2), fill="tozeroy"))
    fig_vol.update_layout(title="Cumulative Volume vs Time", xaxis_title="Time (s)", yaxis_title="Cumulative Volume (mL)", height=340, showlegend=False)

    fig_cal = go.Figure()
    if len(cal_map) >= 2 and cal_map[0] and cal_map[1]:
        fig_cal.add_trace(go.Scatter(x=cal_map[0], y=cal_map[1], name="Calibration Map", mode="lines", line=dict(color="#1E88E5", width=2), fill="tozeroy"))
    fig_cal.update_layout(title="Calibration Map (Drain Rate vs Mass)", xaxis_title="Mass (g)", yaxis_title="Drain Rate (g/s)", height=340, showlegend=False)

    return [
        ("Raw Weight vs Time", fig_raw),
        ("Flow Rate vs Time (KZ smoothed)", fig_kz),
        ("Flow Rate vs Mass", fig_fm),
        ("Cumulative Volume vs Time", fig_vol),
        ("Calibration Map", fig_cal),
    ]


def _build_results_csv(results):
    t = np.asarray(results["t_arr"], dtype=float)
    n = len(t)
    def _pad(arr): return np.asarray(arr, dtype=float)[:n] if len(arr) >= n else np.pad(np.asarray(arr, dtype=float), (0, n - len(arr)))
    cum_g = _pad(results.get("cum_volume", []))
    # Build zone column
    empty_zones = results.get("empty", [])
    voiding_zones = results.get("voiding", [])
    draining_zones = results.get("draining", [])
    roi = results.get("roi", [0, max(0, len(t) - 1)])
    zone_col = np.full(len(t), "unknown", dtype=object)
    for seg in empty_zones:
        zone_col[seg[0]:seg[1]] = "empty"
    for seg in draining_zones:
        zone_col[seg[0]:seg[1]] = "draining"
    for seg in voiding_zones:
        zone_col[seg[0]:seg[1]] = "voiding"
    is_roi = np.zeros(len(t), dtype=int)
    is_roi[roi[0]:roi[1] + 1] = 1
    df = pd.DataFrame({
        "time_s":           t,
        "mass_g":           _pad(results.get("raw_mass", [])),   # matches page_results reader
        "filt_mass_g":      _pad(results.get("filt_mass", [])),
        "filt_inflow_g_s":  _pad(results.get("filt_inflow", [])),
        "kz_flow_g_s":      _pad(results.get("kz_flow", [])),
        "cum_volume_g":     cum_g,
        "cum_volume_mL":    cum_g / 0.9982,
        "zone":             zone_col,
        "is_roi":           is_roi,
    })
    # Scalar metadata as a comment row at the top
    meta = (
        f"# expected_volume_mL={results.get('expected_volume_mL', 0.0):.2f},"
        f"sensor_cal_factor={results.get('sensor_cal_factor_used', 0.0):.8f},"
        f"source_duration_s={results.get('source_duration', 0.0):.2f}\n"
    )
    return (meta + df.to_csv(index=False)).encode()


def _build_run_analysis_pdf(results):
    figures = [("Overview", _build_overlay_figure(results))] + _build_run_analysis_figures(results)
    images = []
    for _, fig in figures:
        png_bytes = pio.to_image(fig, format="png", width=1600, height=900, scale=1)
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        images.append(img)
    if not images:
        return None
    pdf_buffer = io.BytesIO()
    images[0].save(pdf_buffer, format="PDF", save_all=True, append_images=images[1:])
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


def _render_saved_run_analysis(results):
    t_arr = np.asarray(results["t_arr"], dtype=float)
    filt_mass = np.asarray(results["filt_mass"], dtype=float)
    kz_flow = np.asarray(results["kz_flow"], dtype=float)
    cum_volume = np.asarray(results["cum_volume"], dtype=float)
    empty = results["empty"]
    voiding = results["voiding"]
    draining = results["draining"]
    roi = results.get("roi", [0, max(0, len(t_arr) - 1)])

    # ── summary metrics ────────────────────────────────────────────────
    duration = float(t_arr[-1] - t_arr[0]) if len(t_arr) > 1 else 0.0
    n_samples = len(t_arr)
    inferred_fs = (n_samples - 1) / duration if duration > 0 else 0.0
    peak_flow = float(np.max(kz_flow)) if len(kz_flow) else 0.0
    total_vol = float(cum_volume[-1]) if len(cum_volume) > 0 else 0.0
    mass_change = float(filt_mass[-1] - filt_mass[0]) if len(filt_mass) > 1 else 0.0

    cols = st.columns(5)
    cols[0].metric("Duration", f"{duration:.1f} s")
    cols[1].metric("Samples / Rate", f"{n_samples} / {inferred_fs:.1f} Hz")
    cols[2].metric("Peak Flow", f"{peak_flow:.2f} g/s")
    cols[3].metric("Cumul. Volume", f"{total_vol:.0f} g  ({total_vol/0.9982:.0f} mL)")
    cols[4].metric("Mass change", f"{mass_change:+.0f} g")

    # ── volume / sensor calibration diagnostic ─────────────────────────
    expected_vol = float(results.get("expected_volume_mL", 0.0))
    cal_used = float(results.get("sensor_cal_factor_used", DEFAULT_CALIBRATION_FACTOR))
    if expected_vol > 0 and total_vol > 0.5:
        ratio = expected_vol / total_vol
        implied_factor = cal_used * ratio
        pct_error = abs(expected_vol - total_vol) / expected_vol * 100.0
        colour = "green" if pct_error < 10 else ("orange" if pct_error < 50 else "red")
        st.markdown(
            f"**Volume accuracy:** sensor reported **{total_vol:.0f} mL**, "
            f"expected **{expected_vol:.0f} mL** "
            f"— :{colour}[**{pct_error:.1f}% error**]"
        )
        if abs(ratio - 1.0) > 0.05:
            with st.expander("Fix sensor calibration factor", expanded=(abs(ratio - 1.0) > 0.20)):
                st.markdown(
                    f"The sensor cal factor **{cal_used:.8f}** needs to be scaled by **{ratio:.3f}×** "
                    f"to match the expected volume.\n\n"
                    f"**Implied correct factor: `{implied_factor:.8f}`**\n\n"
                    f"This is computed as `current_factor × (expected_volume / measured_volume)` "
                    f"= `{cal_used:.8f} × ({expected_vol:.0f} / {total_vol:.0f})`.\n\n"
                    f"You can also override the expected volume below if you measured the actual output."
                )
                override_vol = st.number_input(
                    "Override expected volume (mL) — leave as-is to use CSV value",
                    value=float(expected_vol), min_value=1.0, step=10.0,
                    key=f"override_vol_{int(results.get('saved_at', 0))}",
                )
                if override_vol != expected_vol:
                    implied_factor = cal_used * (override_vol / max(total_vol, 0.01))
                    st.info(f"With override: implied factor = {implied_factor:.8f}")

                cfg = st.session_state.cfg
                sensor_obj = st.session_state.sensor
                if st.button("Apply corrected sensor cal factor", key=f"apply_cal_{int(results.get('saved_at', 0))}"):
                    cfg["sensor_cal_factor"] = implied_factor
                    sensor_obj.calibration_factor = implied_factor
                    _save_cfg(cfg)
                    st.success(f"Sensor cal factor updated to {implied_factor:.8f}. Re-run the test to verify.")
                    st.rerun()

    saved_at = int(results.get("saved_at", 0))

    # ── CSV download (instant) ──────────────────────────────────────────
    csv_bytes = st.session_state.get("last_run_csv")
    if st.session_state.get("last_run_csv_for") != saved_at or csv_bytes is None:
        csv_bytes = _build_results_csv(results)
        st.session_state["last_run_csv"] = csv_bytes
        st.session_state["last_run_csv_for"] = saved_at

    dl_col, pdf_col = st.columns([2, 1])
    with dl_col:
        st.download_button(
            "Download results CSV",
            data=csv_bytes,
            file_name=f"autoflow_results_{saved_at}.csv",
            mime="text/csv",
            key=f"download_csv_{saved_at}",
            type="primary",
            use_container_width=True,
        )
    with pdf_col:
        if st.button("Export PDF", key=f"prepare_pdf_{saved_at}", use_container_width=True):
            try:
                with st.spinner("Rendering PDF from graphs..."):
                    pdf_bytes = _build_run_analysis_pdf(results)
                    st.session_state["last_run_pdf"] = pdf_bytes
                    st.session_state["last_run_pdf_for"] = saved_at
                st.rerun()
            except Exception as e:
                st.warning(f"PDF export unavailable: {e}")

    pdf_bytes = st.session_state.get("last_run_pdf")
    if pdf_bytes is not None and st.session_state.get("last_run_pdf_for") == saved_at:
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=f"autoflow_graphs_{saved_at}.pdf",
            mime="application/pdf",
            key=f"download_pdf_{saved_at}",
            use_container_width=True,
        )

    # Auto-trigger CSV download in browser when queue auto-downloads
    if st.session_state.get("auto_download_csv"):
        csv_b64 = base64.b64encode(csv_bytes).decode()
        fname = f"autoflow_results_{saved_at}.csv"
        st.components.v1.html(
            f"""<script>
            (function(){{
                var a = document.createElement('a');
                a.href = 'data:text/csv;base64,{csv_b64}';
                a.download = '{fname}';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            }})();
            </script>""",
            height=0,
        )
        st.session_state["auto_download_csv"] = False

    # ── primary overlay chart (Flutter-style: all traces on one plot) ──
    st.plotly_chart(_build_overlay_figure(results), use_container_width=True)

    # ── legend ─────────────────────────────────────────────────────────
    leg_cols = st.columns(6)
    for col, (color, label) in zip(leg_cols, [
        ("#2196F3", "Mass"),
        ("#E91E63", "KZ Flow (scaled)"),
        ("rgba(76,175,80,0.5)", "Voiding"),
        ("rgba(255,152,0,0.5)", "Draining"),
        ("rgba(158,158,158,0.5)", "Empty"),
        ("rgba(135,206,250,0.5)", "ROI"),
    ]):
        col.markdown(
            f'<span style="display:inline-block;width:14px;height:10px;background:{color};'
            f'border-radius:2px;margin-right:4px"></span>{label}',
            unsafe_allow_html=True,
        )

    # ── detailed individual charts ─────────────────────────────────────
    with st.expander("Detailed individual charts"):
        for _, fig in _build_run_analysis_figures(results):
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Zone Analysis Results**")
        st.code(
            "\n".join([
                f"Empty Zones: {empty}",
                f"Voiding Zones: {voiding}",
                f"Draining Zones: {draining}",
                f"ROI: {roi}",
            ]),
            language="text",
        )


def _display_sensor_analysis():
    """Analyze and display collected sensor data after pump run."""
    sensor = st.session_state.sensor
    t_data, m_data, _ = sensor.get_data()

    if not t_data or len(t_data) < 3:
        st.session_state["run_analysis_status"] = "No sensor analysis available"
        pending_name = st.session_state.get("pending_multi_fit_run_name")
        if pending_name:
            st.session_state["last_multi_fit_capture_result"] = {
                "name": pending_name,
                "ok": False,
                "payload": "No usable sensor data captured.",
            }
            st.session_state["pending_multi_fit_run_name"] = None
        age = sensor.last_packet_age
        age_str = f"{age:.1f}s ago" if age is not None else "never"
        st.error(
            f"No sensor data was collected during the run ({len(t_data)} samples). "
            f"Last packet from sensor: **{age_str}**.\n\n"
            "**Fix:** the sensor BLE stream stalled. Disconnect the sensor in the sidebar, "
            "reconnect, wait until you see live readings updating, then run again."
        )
        return

    # Get calibration map
    cal_map = st.session_state.get("calibration_map", [[], []])

    # Run the full analysis pipeline
    t_arr = np.asarray(t_data, dtype=float)
    m_arr = np.asarray(m_data, dtype=float)
    filt_mass, filt_inflow, kz_flow, cum_volume, empty, voiding, draining, roi = compute_flow_from_mass(t_arr, m_arr, cal_map)

    saved_at = int(time.time())
    analysis = {
        "t_arr": t_arr.tolist(),
        "raw_mass": m_arr.tolist(),
        "filt_mass": np.asarray(filt_mass, dtype=float).tolist(),
        "filt_inflow": np.asarray(filt_inflow, dtype=float).tolist(),
        "kz_flow": np.asarray(kz_flow, dtype=float).tolist(),
        "cum_volume": np.asarray(cum_volume, dtype=float).tolist(),
        "calibration_map": cal_map,
        "empty": empty,
        "voiding": voiding,
        "draining": draining,
        "roi": roi,
        "saved_at": saved_at,
        "expected_volume_mL": st.session_state.get("last_run_expected_volume", 0.0),
        "sensor_cal_factor_used": sensor.calibration_factor,
        "source_duration": st.session_state.get("last_run_source_duration", 0.0),
    }
    st.session_state["last_run_analysis"] = analysis
    st.session_state["run_analysis_status"] = "Graphs ready"

    # Build CSV immediately (instant) and store for auto-download
    st.session_state["last_run_csv"] = _build_results_csv(analysis)
    st.session_state["last_run_csv_for"] = saved_at
    st.session_state["auto_download_csv"] = True

    pending_name = st.session_state.get("pending_multi_fit_run_name")
    if pending_name:
        ok, payload = _append_multi_cal_run_from_csv_bytes(
            pending_name,
            st.session_state["last_run_csv"],
        )
        st.session_state["last_multi_fit_capture_result"] = {
            "name": pending_name,
            "ok": ok,
            "payload": payload,
        }
        st.session_state["pending_multi_fit_run_name"] = None

    # Clear any stale PDF state
    st.session_state["last_run_pdf"] = None
    st.session_state["last_run_pdf_for"] = None

    return


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="AutoFlow Dashboard", page_icon="🔬", layout="wide")
    _init()
    _sidebar()

    page = st.radio(
        "Page", ["Sensor & Calibration", "Test Results", "Run Automated Test"],
        horizontal=True, label_visibility="collapsed",
    )

    if page == "Sensor & Calibration":
        page_sensor()
    elif page == "Test Results":
        page_results()
    else:
        page_run()


main()

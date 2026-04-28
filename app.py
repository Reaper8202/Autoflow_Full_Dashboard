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

# ── config persistence ─────────────────────────────────────────────────────

CONFIG_PATH = Path.home() / ".autoflow_dashboard_config.json"
DEFAULT_CONFIG = {
    "cal_factor": 0.030,
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

        # Auto-refresh while collecting
        if is_collecting:
            time.sleep(0.5)
            st.rerun()

    # ── calibration workflow ───────────────────────────────────────────
    with tab_calib:
        st.subheader("Calibration Workflow")
        st.markdown("""
        1. Place **250 mL of water** (250 g) on the sensor
        2. Press **Start Calibration** — data will be collected for ~20 seconds
        3. The system auto-stops when readings stabilize, or press **Stop**
        4. A calibration map (mass → drain rate) is computed and saved
        """)

        cc1, cc2, cc3 = st.columns(3)
        cal_start = cc1.button("Start Calibration", use_container_width=True, disabled=not sensor.is_open())
        cal_stop = cc2.button("Stop Calibration", use_container_width=True)
        cal_save = cc3.button("Save Map to CSV", use_container_width=True)

        if cal_start:
            sensor.start_collecting()
            st.session_state["calibrating"] = True
            st.session_state["cal_start_time"] = time.time()

        if cal_stop and st.session_state.get("calibrating"):
            sensor.stop_collecting()
            st.session_state["calibrating"] = False
            _build_calibration_map()

        is_calibrating = st.session_state.get("calibrating", False)

        if is_calibrating:
            elapsed = time.time() - st.session_state.get("cal_start_time", time.time())
            t_data, m_data, _ = sensor.get_data()

            st.progress(min(1.0, elapsed / 20.0), text=f"Collecting... {elapsed:.1f}s ({len(t_data)} samples)")

            if t_data:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=t_data, y=m_data, name="Calibration Data", line=dict(color="#4CAF50", width=2)))
                fig.update_layout(xaxis_title="Time (s)", yaxis_title="Mass (g)", height=300, margin=dict(t=30))
                st.plotly_chart(fig, use_container_width=True)

            # Auto-stop: check stability (last 30 readings within 1g of each other)
            if len(m_data) >= 40 and elapsed > 5:
                recent = m_data[-30:]
                spread = max(recent) - min(recent)
                if spread < 1.0:
                    sensor.stop_collecting()
                    st.session_state["calibrating"] = False
                    _build_calibration_map()
                    st.success(f"Auto-stopped: readings stable (spread={spread:.2f}g)")
                else:
                    time.sleep(0.5)
                    st.rerun()
            elif is_calibrating:
                # Safety timeout at 60s
                if elapsed > 60:
                    sensor.stop_collecting()
                    st.session_state["calibrating"] = False
                    _build_calibration_map()
                    st.warning("Calibration timed out at 60s")
                else:
                    time.sleep(0.5)
                    st.rerun()

        if cal_save:
            _save_calibration_map_csv()

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


def _build_calibration_map():
    """Build a calibration map from the most recent sensor data."""
    sensor = st.session_state.sensor
    t_data, m_data, _ = sensor.get_data()
    if len(t_data) < 20:
        st.warning("Not enough data to build calibration map (need at least 20 samples)")
        return

    t = np.array(t_data)
    y = np.array(m_data)

    from analysis import lowpass_filter, get_derivative

    # Low-pass filter the mass
    filt_mass = lowpass_filter(y)

    # Find the peak, then look at the descending portion
    max_idx = int(np.argmax(filt_mass))
    max_idx = min(max_idx + 20, len(filt_mass) - 2)  # +20 samples offset

    if max_idx >= len(filt_mass) - 1:
        max_idx = len(filt_mass) - 2

    # Find min after peak
    search_end = len(filt_mass) - 1
    if max_idx >= search_end:
        st.warning("Could not find descending portion for calibration")
        return

    min_idx = max_idx + int(np.argmin(filt_mass[max_idx:search_end]))
    if min_idx <= max_idx:
        st.warning("No proper descending portion found")
        return

    # Derivative and filter
    deriv = get_derivative(t, filt_mass, ss=2)
    filt_rate = lowpass_filter(deriv)

    # Extract the windowed portion
    cal_masses = filt_mass[max_idx:min_idx].tolist()
    cal_rates = filt_rate[max_idx:min_idx].tolist()

    st.session_state.calibration_map = [cal_masses, cal_rates]
    st.success(f"Calibration map built with {len(cal_masses)} points")


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

def page_run():
    st.header("Run Automated Test")

    cfg = st.session_state.cfg
    link = st.session_state.link
    sensor = st.session_state.sensor

    if not link.is_open():
        st.warning("Connect to the pump in the sidebar first.")
    if not sensor.is_open():
        st.warning("Sensor is not connected. The pump can still run, but no post-run flow/mass analysis graphs will be produced.")

    # ── input: CSV or template ─────────────────────────────────────────
    source = st.radio("Input source", ["Upload CSV curve", "Use template"], horizontal=True)

    source_t = None
    source_q = None

    if source == "Upload CSV curve":
        f = st.file_uploader("CSV with columns: time_s, flow_ml_s", type=["csv"], key="run_csv")
        if f:
            try:
                df = pd.read_csv(f)
                t_col = st.selectbox("Time column", df.columns, index=0)
                q_col = st.selectbox("Flow column", df.columns, index=min(1, len(df.columns) - 1))
                source_t = df[t_col].to_numpy(dtype=float)
                source_q = df[q_col].to_numpy(dtype=float)
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

    if source_t is None or source_q is None:
        return

    source_t = np.asarray(source_t, dtype=float)
    source_q = np.clip(np.asarray(source_q, dtype=float), 0.0, None)

    # ── settings ───────────────────────────────────────────────────────
    st.divider()

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

    # Stats
    cols = st.columns(4)
    cols[0].metric("Duration", f"{playback_duration:.1f} s")
    cols[1].metric("Peak flow", f"{source_qmax:.2f} mL/s")
    cols[2].metric("Volume", f"{source_volume:.0f} mL")
    cols[3].metric("Peak RPM", f"{peak_rpm:.0f}")

    # Warnings
    run_disabled = not link.is_open()
    if peak_rpm > PUMP_MAX_RPM:
        st.error(f"Peak RPM ({peak_rpm:.0f}) exceeds max ({PUMP_MAX_RPM:.0f}). Reduce flow or adjust cal_factor.")
        run_disabled = True
    elif 0 < peak_rpm < MIN_RPM_THRESHOLD:
        st.error(f"Peak RPM ({peak_rpm:.1f}) below minimum ({MIN_RPM_THRESHOLD}). Increase flow or adjust cal_factor.")
        run_disabled = True

    # Preview chart
    fig = go.Figure()
    preview_t = (source_t - source_t[0]) / max(speed, 1e-6) if speed != 1.0 else source_t - source_t[0]
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
        if mode == "Exact playback":
            _run_exact(link, source_t, source_q, cfg["cal_factor"], speed)
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
        _render_saved_run_analysis(saved_results)


def _run_exact(link, t, q, cal_factor, speed_mult):
    """Stream the source curve point by point to the pump, with simultaneous sensor collection."""
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

    # Start sensor collection
    if sensor.is_open():
        sensor.start_collecting()

    start = time.time()

    try:
        while True:
            elapsed = time.time() - start
            src_elapsed = min(duration, elapsed * speed_mult)
            flow = float(np.interp(src_elapsed, t, q))
            rpm = max(0.0, min(PUMP_MAX_RPM, flow / cal_factor))
            if rpm < MIN_RPM_THRESHOLD:
                rpm = 0.0

            if last_rpm is None or abs(rpm - last_rpm) >= RPM_WRITE_EPSILON:
                link.write_line("0" if rpm <= 0 else f"{rpm:.3f}")
                last_rpm = rpm

            cmd_t.append(src_elapsed)
            cmd_q.append(flow if rpm > 0 else 0.0)
            cmd_rpm.append(rpm)

            frac = min(1.0, src_elapsed / max(duration, 1e-6))
            progress.progress(frac, text=f"{src_elapsed:.1f}/{duration:.1f}s  flow={flow:.2f} mL/s  rpm={rpm:.0f}")

            if src_elapsed >= duration:
                break
            time.sleep(control_dt)
    finally:
        if link.is_open():
            link.hard_stop("exact playback complete")
        # Stop sensor collection
        if sensor.is_open():
            sensor.stop_collecting()
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
        st.download_button("Download log CSV", csv, file_name=f"playback_{int(time.time())}.csv")

    # Analyze collected sensor data
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

    # Start sensor collection before pump run
    if sensor.is_open():
        sensor.start_collecting()

    if link.is_open() and not link.write_line("run"):
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
        # Stop sensor collection
        if sensor.is_open():
            sensor.stop_collecting()

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

    fig_mass = go.Figure()
    fig_mass.add_trace(go.Scatter(x=t_arr[:len(filt_mass)], y=filt_mass, name="Filtered Mass", line=dict(color="#2196F3", width=2)))
    _add_zone_rects(fig_mass, empty, "rgba(158,158,158,0.10)")
    _add_zone_rects(fig_mass, draining, "rgba(255,152,0,0.18)")
    _add_zone_rects(fig_mass, voiding, "rgba(76,175,80,0.18)")
    _add_roi_lines(fig_mass)
    fig_mass.update_layout(title="Mass vs Time (with zones)", xaxis_title="Time (s)", yaxis_title="Mass (g)", height=380, showlegend=False)

    fig_inflow = go.Figure()
    if len(filt_inflow):
        fig_inflow.add_trace(go.Scatter(x=t_arr[:len(filt_inflow)], y=filt_inflow, name="Filtered Inflow", line=dict(color="#5E35B1", width=2)))
    _add_zone_rects(fig_inflow, empty, "rgba(158,158,158,0.10)")
    _add_zone_rects(fig_inflow, draining, "rgba(255,152,0,0.18)")
    _add_zone_rects(fig_inflow, voiding, "rgba(76,175,80,0.18)")
    _add_roi_lines(fig_inflow)
    fig_inflow.update_layout(title="Filtered Inflow vs Time", xaxis_title="Time (s)", yaxis_title="Flow Rate (g/s)", height=340, showlegend=False)

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

    fig_cal = go.Figure()
    if len(cal_map) >= 2 and cal_map[0] and cal_map[1]:
        fig_cal.add_trace(go.Scatter(x=cal_map[0], y=cal_map[1], name="Calibration Map", mode="lines", line=dict(color="#1E88E5", width=2), fill="tozeroy"))
    fig_cal.update_layout(title="Calibration Map", xaxis_title="Mass (g)", yaxis_title="Drain Rate (g/s)", height=340, showlegend=False)

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=t_arr[:len(cum_volume)], y=cum_volume, name="Cumulative Volume", line=dict(color="#43A047", width=2), fill="tozeroy"))
    fig_vol.update_layout(title="Cumulative Volume vs Time", xaxis_title="Time (s)", yaxis_title="Cumulative Volume (g)", height=340, showlegend=False)

    return [
        ("Raw Weight vs Time", fig_raw),
        ("Mass vs Time (with zones)", fig_mass),
        ("Filtered Inflow vs Time", fig_inflow),
        ("Flow Rate vs Time (KZ smoothed)", fig_kz),
        ("Flow Rate vs Mass", fig_fm),
        ("Calibration Map", fig_cal),
        ("Cumulative Volume vs Time", fig_vol),
    ]


def _build_run_analysis_pdf(results):
    figures = _build_run_analysis_figures(results)
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
    kz_flow = np.asarray(results["kz_flow"], dtype=float)
    cum_volume = np.asarray(results["cum_volume"], dtype=float)
    empty = results["empty"]
    voiding = results["voiding"]
    draining = results["draining"]
    roi = results.get("roi", [0, max(0, len(t_arr) - 1)])

    cols = st.columns(4)
    duration = t_arr[-1] - t_arr[0] if len(t_arr) > 1 else 0
    cols[0].metric("Duration", f"{duration:.1f} s")
    cols[1].metric("Peak Flow", f"{np.max(kz_flow):.2f} g/s" if len(kz_flow) else "0.00 g/s")
    total_vol = cum_volume[-1] if len(cum_volume) > 0 else 0
    cols[2].metric("Total Volume", f"{total_vol:.1f} g ({total_vol / 0.9982:.0f} mL)")
    cols[3].metric("Zones", f"V:{len(voiding)} D:{len(draining)} E:{len(empty)}")

    saved_at = int(results.get("saved_at", 0))
    if st.session_state.get("last_run_pdf_for") != saved_at:
        st.session_state["last_run_pdf"] = None
        st.session_state["last_run_pdf_for"] = saved_at

    pdf_col1, pdf_col2 = st.columns([1, 2])
    with pdf_col1:
        if st.button("Prepare PDF", key=f"prepare_pdf_{saved_at}"):
            try:
                with st.spinner("Rendering PDF from graphs..."):
                    st.session_state["last_run_pdf"] = _build_run_analysis_pdf(results)
            except Exception as e:
                st.session_state["last_run_pdf"] = None
                st.warning(f"PDF export unavailable right now: {e}")
    with pdf_col2:
        if st.session_state.get("last_run_pdf") is not None:
            st.download_button(
                "Download graphs as PDF",
                data=st.session_state["last_run_pdf"],
                file_name=f"autoflow_graphs_{saved_at}.pdf",
                mime="application/pdf",
                key=f"download_pdf_{saved_at}",
            )

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
        st.info("No sensor data collected during pump run.")
        return

    # Get calibration map
    cal_map = st.session_state.get("calibration_map", [[], []])

    # Run the full analysis pipeline
    t_arr = np.asarray(t_data, dtype=float)
    m_arr = np.asarray(m_data, dtype=float)
    filt_mass, filt_inflow, kz_flow, cum_volume, empty, voiding, draining, roi = compute_flow_from_mass(t_arr, m_arr, cal_map)

    st.session_state["last_run_pdf"] = None
    st.session_state["last_run_pdf_for"] = None
    st.session_state["last_run_analysis"] = {
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
        "saved_at": time.time(),
    }
    st.session_state["run_analysis_status"] = "Graphs ready"

    return


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="AutoFlow Dashboard", page_icon="🔬", layout="wide")
    _init()
    _sidebar()

    page = st.radio(
        "", ["Sensor & Calibration", "Test Results", "Run Automated Test"],
        horizontal=True, label_visibility="collapsed",
    )

    if page == "Sensor & Calibration":
        page_sensor()
    elif page == "Test Results":
        page_results()
    else:
        page_run()


main()

#!/usr/bin/env python3
"""Command-line operator tools for AutoFlow pump validation.

This script intentionally avoids importing dashboard/Streamlit code. It uses
the lower-level modules directly so automated runs can be launched, inspected,
and analyzed from a terminal or by another agent.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = Path.home() / ".autoflow_dashboard_config.json"


def _die(message: str, code: int = 1) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def _load_science():
    try:
        import numpy as np
        import pandas as pd
        import plotly.graph_objects as go
        from analysis import (
            SHAPES,
            analyze_curve,
            build_run_profile,
            compute_flow_from_mass,
        )
    except ModuleNotFoundError as exc:
        missing = exc.name or "dependency"
        _die(
            f"missing Python dependency '{missing}'. Install the project environment with "
            "`python3 -m pip install -r requirements.txt` from automated_dashboard."
        )
    return np, pd, go, SHAPES, analyze_curve, build_run_profile, compute_flow_from_mass


def _load_hardware():
    try:
        from pump_link import (
            MIN_RPM_THRESHOLD,
            RPM_WRITE_EPSILON,
            PUMP_MAX_RPM,
            PumpLink,
            list_serial_ports,
        )
        from sensor_link import (
            DEFAULT_CALIBRATION_FACTOR,
            SensorLink,
            discover_ble_devices,
            list_sensor_serial_ports,
        )
    except ModuleNotFoundError as exc:
        missing = exc.name or "dependency"
        _die(
            f"missing Python dependency '{missing}'. Install the project environment with "
            "`python3 -m pip install -r requirements.txt` from automated_dashboard."
        )
    return {
        "MIN_RPM_THRESHOLD": MIN_RPM_THRESHOLD,
        "RPM_WRITE_EPSILON": RPM_WRITE_EPSILON,
        "PUMP_MAX_RPM": PUMP_MAX_RPM,
        "PumpLink": PumpLink,
        "SensorLink": SensorLink,
        "DEFAULT_CALIBRATION_FACTOR": DEFAULT_CALIBRATION_FACTOR,
        "discover_ble_devices": discover_ble_devices,
        "list_serial_ports": list_serial_ports,
        "list_sensor_serial_ports": list_sensor_serial_ports,
    }


def _load_dashboard_config() -> dict[str, Any]:
    if not DEFAULT_CONFIG.exists():
        return {}
    try:
        return json.loads(DEFAULT_CONFIG.read_text())
    except Exception:
        return {}


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_outdir(path: str | Path | None, prefix: str = "run") -> Path:
    if path:
        out = Path(path)
    else:
        out = ROOT / "operator_runs" / f"{prefix}_{_timestamp()}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _read_csv(path: str | Path):
    _, pd, _, _, _, _, _ = _load_science()
    return pd.read_csv(path, comment="#")


def _find_col(df, candidates: list[str]) -> str | None:
    columns = list(df.columns)
    for candidate in candidates:
        if candidate in columns:
            return candidate
    lower = {str(col).lower(): col for col in columns}
    for candidate in candidates:
        match = lower.get(candidate.lower())
        if match is not None:
            return match
    return None


def _load_profile_csv(path: str | Path):
    np, _, _, _, _, _, _ = _load_science()
    df = _read_csv(path)
    t_col = _find_col(df, ["time_s", "time", "t"])
    q_col = _find_col(df, ["flow_ml_s", "flow_mL_s", "flow", "q", "kz_flow_g_s"])
    if t_col is None or q_col is None:
        _die(f"{path}: expected time_s and flow_ml_s columns")
    t = df[t_col].to_numpy(dtype=float)
    q = np.clip(df[q_col].to_numpy(dtype=float), 0.0, None)
    order = np.argsort(t)
    t = t[order]
    q = q[order]
    if len(t) < 2 or float(t[-1] - t[0]) <= 0:
        _die(f"{path}: profile needs at least two increasing time points")
    return t - t[0], q


def _validate_positive(name: str, value: float) -> float:
    value = float(value)
    if value <= 0:
        _die(f"{name} must be positive")
    return value


def _load_calibration_map(path: str | Path | None) -> list[list[float]]:
    if path is None:
        return [[], []]
    df = _read_csv(path)
    m_col = _find_col(df, ["mass_g", "mass"])
    r_col = _find_col(df, ["drain_rate_g_per_s", "drain_rate", "rate"])
    if m_col is None or r_col is None:
        _die(f"{path}: expected mass_g and drain_rate_g_per_s columns")
    return [
        df[m_col].dropna().astype(float).tolist(),
        df[r_col].dropna().astype(float).tolist(),
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _build_results(t, mass, calibration_map, metadata: dict[str, Any]) -> dict[str, Any]:
    np, _, _, _, _, _, compute_flow_from_mass = _load_science()
    filt_mass, filt_inflow, kz_flow, cum_volume, empty, voiding, draining, roi = compute_flow_from_mass(
        np.asarray(t, dtype=float),
        np.asarray(mass, dtype=float),
        calibration_map,
    )
    results = {
        "t_arr": np.asarray(t, dtype=float).tolist(),
        "raw_mass": np.asarray(mass, dtype=float).tolist(),
        "filt_mass": np.asarray(filt_mass, dtype=float).tolist(),
        "filt_inflow": np.asarray(filt_inflow, dtype=float).tolist(),
        "kz_flow": np.asarray(kz_flow, dtype=float).tolist(),
        "cum_volume": np.asarray(cum_volume, dtype=float).tolist(),
        "calibration_map": calibration_map,
        "empty": empty,
        "voiding": voiding,
        "draining": draining,
        "roi": roi,
        "saved_at": int(time.time()),
    }
    results.update(metadata)
    return results


def _results_summary(results: dict[str, Any]) -> dict[str, Any]:
    np, _, _, _, _, _, _ = _load_science()
    t = np.asarray(results.get("t_arr", []), dtype=float)
    mass = np.asarray(results.get("filt_mass", []), dtype=float)
    flow = np.asarray(results.get("kz_flow", []), dtype=float)
    volume = np.asarray(results.get("cum_volume", []), dtype=float)
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    samples = int(len(t))
    measured_ml = float(volume[-1] / 0.9982) if len(volume) else 0.0
    expected_ml = float(results.get("expected_volume_mL", 0.0) or 0.0)
    tolerance_pct = float(results.get("volume_tolerance_pct", 20.0) or 20.0)
    sensor_factor = float(results.get("sensor_cal_factor_used", 0.0) or 0.0)
    pct_error = None
    volume_pass = None
    calibration_scale = None
    suggested_sensor_factor = None
    if expected_ml > 0:
        pct_error = abs(measured_ml - expected_ml) / expected_ml * 100.0
        volume_pass = pct_error <= tolerance_pct
    if measured_ml > 0 and expected_ml > 0:
        calibration_scale = expected_ml / measured_ml
        if sensor_factor > 0:
            suggested_sensor_factor = sensor_factor * calibration_scale
    return {
        "status": "pass" if volume_pass is True else ("fail" if volume_pass is False else "unknown"),
        "ground_truth": "input_csv_integrated_volume",
        "duration_s": duration,
        "samples": samples,
        "sample_rate_hz": (samples - 1) / duration if duration > 0 and samples > 1 else 0.0,
        "peak_flow_g_s": float(np.max(flow)) if len(flow) else 0.0,
        "measured_volume_mL": measured_ml,
        "expected_volume_mL": expected_ml,
        "volume_tolerance_pct": tolerance_pct,
        "volume_error_pct": pct_error,
        "sensor_calibration_scale": calibration_scale,
        "suggested_sensor_cal_factor": suggested_sensor_factor,
        "mass_change_g": float(mass[-1] - mass[0]) if len(mass) > 1 else 0.0,
        "zone_counts": {
            "empty": len(results.get("empty", [])),
            "voiding": len(results.get("voiding", [])),
            "draining": len(results.get("draining", [])),
        },
        "roi": results.get("roi", []),
    }


def _write_analysis_csv(path: Path, results: dict[str, Any]) -> None:
    np, pd, _, _, _, _, _ = _load_science()
    t = np.asarray(results["t_arr"], dtype=float)
    n = len(t)

    def pad(values):
        arr = np.asarray(values, dtype=float)
        if len(arr) >= n:
            return arr[:n]
        return np.pad(arr, (0, n - len(arr)))

    zone = np.full(n, "unknown", dtype=object)
    for label in ("empty", "draining", "voiding"):
        for start, end in results.get(label, []):
            zone[int(start):int(end)] = label
    is_roi = np.zeros(n, dtype=int)
    roi = results.get("roi", [0, max(0, n - 1)])
    if len(roi) == 2 and n:
        is_roi[int(roi[0]):int(roi[1]) + 1] = 1

    cum_g = pad(results.get("cum_volume", []))
    df = pd.DataFrame(
        {
            "time_s": t,
            "mass_g": pad(results.get("raw_mass", [])),
            "filt_mass_g": pad(results.get("filt_mass", [])),
            "filt_inflow_g_s": pad(results.get("filt_inflow", [])),
            "kz_flow_g_s": pad(results.get("kz_flow", [])),
            "cum_volume_g": cum_g,
            "cum_volume_mL": cum_g / 0.9982,
            "zone": zone,
            "is_roi": is_roi,
        }
    )
    metadata = (
        f"# expected_volume_mL={float(results.get('expected_volume_mL', 0.0) or 0.0):.2f},"
        f"sensor_cal_factor={float(results.get('sensor_cal_factor_used', 0.0) or 0.0):.8f},"
        f"source_duration_s={float(results.get('source_duration', 0.0) or 0.0):.2f}\n"
    )
    path.write_text(metadata + df.to_csv(index=False))


def _write_plots(path: Path, results: dict[str, Any]) -> None:
    np, _, go, _, _, _, _ = _load_science()
    t = np.asarray(results["t_arr"], dtype=float)
    raw = np.asarray(results.get("raw_mass", []), dtype=float)
    mass = np.asarray(results.get("filt_mass", []), dtype=float)
    flow = np.asarray(results.get("kz_flow", []), dtype=float)
    volume = np.asarray(results.get("cum_volume", []), dtype=float) / 0.9982

    fig = go.Figure()
    if len(raw):
        fig.add_trace(go.Scatter(x=t[:len(raw)], y=raw, name="raw mass g", line=dict(width=1)))
    if len(mass):
        fig.add_trace(go.Scatter(x=t[:len(mass)], y=mass, name="filtered mass g", line=dict(width=2)))
    if len(flow):
        fig.add_trace(go.Scatter(x=t[:len(flow)], y=flow, name="kz flow g/s", yaxis="y2", line=dict(width=2)))
    if len(volume):
        fig.add_trace(go.Scatter(x=t[:len(volume)], y=volume, name="volume mL", yaxis="y3", line=dict(width=2)))
    fig.update_layout(
        title="AutoFlow Operator Analysis",
        xaxis_title="Time (s)",
        yaxis=dict(title="Mass (g)"),
        yaxis2=dict(title="Flow (g/s)", overlaying="y", side="right"),
        yaxis3=dict(title="Volume (mL)", anchor="free", overlaying="y", side="right", position=0.94),
        height=720,
    )
    fig.write_html(path, include_plotlyjs="cdn")


def cmd_list_ports(args: argparse.Namespace) -> None:
    hw = _load_hardware()
    payload: dict[str, Any] = {
        "pump_ports": hw["list_serial_ports"](),
        "sensor_serial_ports": hw["list_sensor_serial_ports"](),
    }
    if args.ble:
        try:
            devices = hw["discover_ble_devices"](timeout=args.timeout)
            payload["ble_devices"] = [
                {"name": d.name, "address": d.address, "rssi": d.rssi, "token": d.token, "label": d.label}
                for d in devices
            ]
        except Exception as exc:
            payload["ble_error"] = str(exc)
    print(json.dumps(payload, indent=2))


def cmd_make_profile(args: argparse.Namespace) -> None:
    np, pd, _, SHAPES, _, build_run_profile, _ = _load_science()
    if args.shape not in SHAPES:
        _die(f"unknown shape {args.shape!r}; choose one of {', '.join(SHAPES)}")
    profile, err = build_run_profile(args.shape, args.qmax, args.volume, args.duration, n=args.samples)
    if err:
        _die(err)
    t = np.asarray(profile["u"], dtype=float) * args.duration
    q = np.asarray(profile["q"], dtype=float)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time_s": t, "flow_ml_s": q}).to_csv(out, index=False)
    summary = {
        "output": str(out),
        "shape": args.shape,
        "duration_s": args.duration,
        "peak_flow_ml_s": float(np.max(q)),
        "volume_mL": float(np.trapezoid(q, t)),
        "samples": int(len(t)),
    }
    print(json.dumps(summary, indent=2))


def cmd_make_mass_fixture(args: argparse.Namespace) -> None:
    np, pd, _, _, _, _, _ = _load_science()
    source_t, source_q = _load_profile_csv(args.input)
    sample_rate = _validate_positive("--sample-rate", args.sample_rate)
    dt = 1.0 / sample_rate
    end = float(source_t[-1]) + max(0.0, args.post_s)
    t = np.arange(0.0, end + dt / 2.0, dt)
    effective_t = t - args.delay_s
    flow = np.interp(effective_t, source_t, source_q, left=0.0, right=0.0)
    flow = np.clip(flow * args.flow_scale + args.flow_bias, 0.0, None)

    rng = np.random.default_rng(args.seed)
    if args.flow_noise_sd > 0:
        flow = np.clip(flow + rng.normal(0.0, args.flow_noise_sd, size=len(flow)), 0.0, None)

    mass = np.zeros(len(t), dtype=float)
    mass[0] = args.baseline_g
    for i in range(1, len(t)):
        drain = args.drain_rate_g_s if mass[i - 1] > args.drain_floor_g else 0.0
        mass[i] = mass[i - 1] + (flow[i - 1] - drain) * (t[i] - t[i - 1])
        if mass[i] < args.drain_floor_g:
            mass[i] = args.drain_floor_g
    if args.mass_noise_sd > 0:
        mass = mass + rng.normal(0.0, args.mass_noise_sd, size=len(mass))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time_s": t, "mass_g": mass, "true_flow_ml_s": flow}).to_csv(out, index=False)
    print(
        json.dumps(
            {
                "output": str(out),
                "samples": int(len(t)),
                "duration_s": float(t[-1] - t[0]),
                "expected_volume_mL": float(np.trapezoid(source_q, source_t)),
                "simulated_final_mass_g": float(mass[-1]),
            },
            indent=2,
        )
    )


def cmd_analyze(args: argparse.Namespace) -> None:
    np, pd, _, _, analyze_curve, _, _ = _load_science()
    out = _ensure_outdir(args.outdir, prefix="analysis")
    df = _read_csv(args.input)
    t_col = _find_col(df, ["time_s", "time", "t"])
    if t_col is None:
        _die(f"{args.input}: expected a time_s column")

    mass_col = _find_col(df, ["mass_g", "rawMass_g", "rawMass", "mass"])
    flow_col = _find_col(df, ["flow_ml_s", "flow", "kz_flow_g_s", "filt_inflow_g_s"])
    t = df[t_col].to_numpy(dtype=float)

    if mass_col is not None:
        mass = df[mass_col].to_numpy(dtype=float)
        cal_map = _load_calibration_map(args.calibration_map)
        results = _build_results(
            t,
            mass,
            cal_map,
            {
                "expected_volume_mL": float(args.expected_volume or 0.0),
                "sensor_cal_factor_used": float(args.sensor_cal_factor or 0.0),
                "source_duration": float(args.source_duration or 0.0),
                "volume_tolerance_pct": float(args.volume_tolerance_pct),
            },
        )
        summary = _results_summary(results)
        _write_json(out / "summary.json", summary)
        _write_json(out / "analysis.json", results)
        _write_analysis_csv(out / "analysis.csv", results)
        if not args.no_plots:
            _write_plots(out / "analysis.html", results)
        print(json.dumps({"outdir": str(out), "summary": summary}, indent=2))
        return

    if flow_col is None:
        _die(f"{args.input}: expected either mass_g or flow_ml_s columns")

    q = np.clip(df[flow_col].to_numpy(dtype=float), 0.0, None)
    fit = analyze_curve(t, q)
    summary = {
        "duration_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
        "peak_flow": float(np.max(q)) if len(q) else 0.0,
        "volume": float(np.trapezoid(q, t)) if len(t) > 1 else 0.0,
        "fit": fit and {k: v for k, v in fit.items() if k not in ("t", "q")},
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps({"outdir": str(out), "summary": summary}, indent=2))


def cmd_build_calibration_map(args: argparse.Namespace) -> None:
    np, pd, _, _, _, _, _ = _load_science()
    try:
        from analysis import get_derivative, lowpass_filter
    except ModuleNotFoundError as exc:
        _die(f"missing Python dependency '{exc.name}'")

    df = _read_csv(args.input)
    t_col = _find_col(df, ["time_s", "time", "t"])
    mass_col = _find_col(df, ["mass_g", "rawMass_g", "rawMass", "mass"])
    if t_col is None or mass_col is None:
        _die(f"{args.input}: expected time_s and mass_g columns")

    t = df[t_col].to_numpy(dtype=float)
    y = df[mass_col].to_numpy(dtype=float)
    if len(t) < 20:
        _die("not enough samples to build a calibration map; need at least 20")

    filt_mass = lowpass_filter(y)
    total_drop = float(filt_mass.max() - filt_mass.min())
    if total_drop < args.min_drop_g:
        _die(
            f"mass only changed {total_drop:.1f} g; need at least {args.min_drop_g:.1f} g. "
            "For pump-only/no-passive-drain setups use a zero-drain map."
        )

    max_idx = int(np.argmax(filt_mass))
    max_idx = min(max_idx + args.peak_offset_samples, len(filt_mass) - 2)
    search_end = len(filt_mass) - 1
    if max_idx >= search_end:
        _die("peak is at the end of the recording; start capture before opening the drain")

    min_idx = max_idx + int(np.argmin(filt_mass[max_idx:search_end]))
    if min_idx <= max_idx + 5:
        _die("drain window too short; let water drain for several seconds")

    deriv = get_derivative(t, filt_mass, ss=2)
    filt_rate = lowpass_filter(deriv)
    pairs = sorted(
        zip(filt_mass[max_idx:min_idx].tolist(), filt_rate[max_idx:min_idx].tolist()),
        key=lambda item: item[0],
        reverse=True,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "mass_g": [p[0] for p in pairs],
            "drain_rate_g_per_s": [p[1] for p in pairs],
        }
    ).to_csv(out, index=False)
    print(
        json.dumps(
            {
                "output": str(out),
                "points": len(pairs),
                "mass_drop_g": total_drop,
                "mass_range_g": [float(min(p[0] for p in pairs)), float(max(p[0] for p in pairs))],
                "rate_range_g_s": [float(min(p[1] for p in pairs)), float(max(p[1] for p in pairs))],
            },
            indent=2,
        )
    )


def cmd_make_zero_calibration(args: argparse.Namespace) -> None:
    _, pd, _, _, _, _, _ = _load_science()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"mass_g": [], "drain_rate_g_per_s": []}).to_csv(out, index=False)
    print(json.dumps({"output": str(out), "points": 0, "mode": "zero_drain"}, indent=2))


def _connect_hardware(args: argparse.Namespace):
    hw = _load_hardware()
    cfg = _load_dashboard_config()
    pump_port = args.pump_port or cfg.get("last_com_port")
    sensor_target = args.sensor_target or cfg.get("last_sensor_port")
    if not pump_port:
        _die("pump port required; pass --pump-port or set it in the dashboard config")
    if not sensor_target:
        _die("sensor target required; pass --sensor-target or set it in the dashboard config")

    pump = hw["PumpLink"]()
    sensor = hw["SensorLink"]()
    sensor.calibration_factor = float(
        args.sensor_cal_factor
        or cfg.get("sensor_cal_factor")
        or hw["DEFAULT_CALIBRATION_FACTOR"]
    )

    print(f"connecting pump: {pump_port}")
    if not pump.connect(pump_port):
        _die("pump connect failed; see pump log for details")
    print(f"connecting sensor: {sensor_target}")
    if not sensor.connect(sensor_target):
        pump.close()
        _die(sensor.last_error or "sensor connect failed")
    return hw, pump, sensor


def _effective_cal_factor(args: argparse.Namespace) -> float:
    if args.cal_factor is not None:
        return _validate_positive("--cal-factor", args.cal_factor)
    return _validate_positive("dashboard cal_factor", _load_dashboard_config().get("cal_factor", 0.030))


def _effective_max_rpm(args: argparse.Namespace, firmware_max_rpm: float) -> float:
    cfg = _load_dashboard_config()
    configured = args.max_rpm if args.max_rpm is not None else cfg.get("pump_max_rpm", firmware_max_rpm)
    return min(_validate_positive("--max-rpm", configured), float(firmware_max_rpm))


def _run_exact_once(
    *,
    source_t,
    source_q,
    pump,
    sensor,
    cal_factor: float,
    speed: float,
    max_rpm: float,
    calibration_map: list[list[float]],
    outdir: Path,
    min_rpm: float,
    rpm_epsilon: float,
    expected_volume_mL: float,
    volume_tolerance_pct: float,
    no_tare: bool,
) -> dict[str, Any]:
    np, pd, _, _, _, _, _ = _load_science()
    speed = _validate_positive("--speed", speed)
    cal_factor = _validate_positive("--cal-factor", cal_factor)
    max_rpm = _validate_positive("--max-rpm", max_rpm)
    t = np.asarray(source_t, dtype=float)
    q = np.clip(np.asarray(source_q, dtype=float), 0.0, None)
    order = np.argsort(t)
    t = t[order] - t[order][0]
    q = q[order]
    duration = float(t[-1])
    playback_duration = duration / max(speed, 1e-6)
    control_dt = min(0.05, max(0.01, playback_duration / 1000.0))
    commands: list[dict[str, float]] = []

    if not no_tare:
        baseline = sensor.tare()
        print(f"sensor tared at {baseline:.2f} g")
        time.sleep(0.3)

    pump.drain()
    sensor.start_collecting()
    start = time.time()
    last_rpm = None
    try:
        while True:
            elapsed = time.time() - start
            src_elapsed = min(duration, elapsed * speed)
            flow = float(np.interp(src_elapsed, t, q))
            rpm = max(0.0, min(max_rpm, flow / cal_factor))
            if rpm < min_rpm:
                rpm = 0.0
            if last_rpm is None or abs(rpm - last_rpm) >= rpm_epsilon:
                pump.write_line("0" if rpm <= 0 else f"{rpm:.3f}")
                pump.drain()
                last_rpm = rpm
            commands.append({"wall_s": elapsed, "profile_s": src_elapsed, "flow_ml_s": flow if rpm > 0 else 0.0, "rpm": rpm})
            if src_elapsed >= duration:
                break
            time.sleep(control_dt)
    finally:
        if pump.is_open():
            pump.hard_stop("autoflow_ops exact run complete")
        sensor.stop_collecting()

    raw_t, raw_mass, raw_values = sensor.get_data()
    pd.DataFrame(commands).to_csv(outdir / "commands.csv", index=False)
    pd.DataFrame({"time_s": raw_t, "mass_g": raw_mass, "raw_sensor": raw_values}).to_csv(outdir / "sensor_raw.csv", index=False)
    (outdir / "pump_log.txt").write_text(pump.export_log_text())

    if len(raw_t) < 3:
        summary = {
            "status": "no_sensor_analysis",
            "samples": len(raw_t),
            "last_packet_age_s": sensor.last_packet_age,
        }
        _write_json(outdir / "summary.json", summary)
        return summary

    results = _build_results(
        raw_t,
        raw_mass,
        calibration_map,
        {
            "expected_volume_mL": expected_volume_mL,
            "sensor_cal_factor_used": sensor.calibration_factor,
            "source_duration": playback_duration,
            "volume_tolerance_pct": volume_tolerance_pct,
        },
    )
    summary = _results_summary(results)
    _write_json(outdir / "analysis.json", results)
    _write_json(outdir / "summary.json", summary)
    _write_analysis_csv(outdir / "analysis.csv", results)
    _write_plots(outdir / "analysis.html", results)
    return summary


def cmd_run_exact(args: argparse.Namespace) -> None:
    np, _, _, _, _, _, _ = _load_science()
    hw, pump, sensor = _connect_hardware(args)
    out = _ensure_outdir(args.outdir, prefix="hardware")
    try:
        args.speed = _validate_positive("--speed", args.speed)
        t, q = _load_profile_csv(args.input)
        cal_factor = _effective_cal_factor(args)
        max_rpm = _effective_max_rpm(args, hw["PUMP_MAX_RPM"])
        calibration_map = _load_calibration_map(args.calibration_map)
        peak_rpm = float(np.max(q)) / cal_factor if cal_factor > 0 and len(q) else 0.0
        if peak_rpm > max_rpm and peak_rpm > 0:
            scale = max_rpm / peak_rpm
            print(f"profile exceeds max RPM; scaling flow by {scale:.4f}")
            q = q * scale
        expected = float(np.trapezoid(q, t)) / max(args.speed, 1e-6)
        summary = _run_exact_once(
            source_t=t,
            source_q=q,
            pump=pump,
            sensor=sensor,
            cal_factor=cal_factor,
            speed=args.speed,
            max_rpm=max_rpm,
            calibration_map=calibration_map,
            outdir=out,
            min_rpm=hw["MIN_RPM_THRESHOLD"],
            rpm_epsilon=hw["RPM_WRITE_EPSILON"],
            expected_volume_mL=expected,
            volume_tolerance_pct=float(args.volume_tolerance_pct),
            no_tare=args.no_tare,
        )
        print(json.dumps({"outdir": str(out), "summary": summary}, indent=2))
    finally:
        try:
            pump.close()
        finally:
            sensor.close()


def _wait_for_drain(sensor, timeout_s: float, threshold_g: float, stable_s: float) -> bool:
    start = time.time()
    stable_start = None
    while True:
        mass = float(sensor.current_reading)
        now = time.time()
        if mass < threshold_g:
            if stable_start is None:
                stable_start = now
            stable_elapsed = now - stable_start
        else:
            stable_start = None
            stable_elapsed = 0.0
        print(f"drain wait mass={mass:.2f}g stable={stable_elapsed:.1f}/{stable_s:.1f}s", end="\r")
        if stable_elapsed >= stable_s:
            print()
            return True
        if now - start >= timeout_s:
            print()
            return False
        time.sleep(0.5)


def cmd_queue(args: argparse.Namespace) -> None:
    np, _, _, _, _, _, _ = _load_science()
    hw, pump, sensor = _connect_hardware(args)
    root = _ensure_outdir(args.outdir, prefix="queue")
    summaries = []
    try:
        args.speed = _validate_positive("--speed", args.speed)
        cal_factor = _effective_cal_factor(args)
        max_rpm = _effective_max_rpm(args, hw["PUMP_MAX_RPM"])
        calibration_map = _load_calibration_map(args.calibration_map)
        for idx, profile in enumerate(args.inputs, start=1):
            run_out = root / f"{idx:02d}_{Path(profile).stem}"
            run_out.mkdir(parents=True, exist_ok=True)
            print(f"running {idx}/{len(args.inputs)}: {profile}")
            t, q = _load_profile_csv(profile)
            peak_rpm = float(np.max(q)) / cal_factor if cal_factor > 0 and len(q) else 0.0
            if peak_rpm > max_rpm and peak_rpm > 0:
                q = q * (max_rpm / peak_rpm)
            expected = float(np.trapezoid(q, t)) / max(args.speed, 1e-6)
            summary = _run_exact_once(
                source_t=t,
                source_q=q,
                pump=pump,
                sensor=sensor,
                cal_factor=cal_factor,
                speed=args.speed,
                max_rpm=max_rpm,
                calibration_map=calibration_map,
                outdir=run_out,
                min_rpm=hw["MIN_RPM_THRESHOLD"],
                rpm_epsilon=hw["RPM_WRITE_EPSILON"],
                expected_volume_mL=expected,
                volume_tolerance_pct=float(args.volume_tolerance_pct),
                no_tare=args.no_tare,
            )
            summary["profile"] = str(profile)
            summary["outdir"] = str(run_out)
            summaries.append(summary)
            _write_json(root / "queue_summary.json", {"runs": summaries})
            if idx < len(args.inputs):
                ok = _wait_for_drain(sensor, args.drain_timeout, args.drain_threshold, args.drain_stable)
                if not ok:
                    summaries.append({"status": "stopped", "reason": "drain_timeout"})
                    _write_json(root / "queue_summary.json", {"runs": summaries})
                    break
        print(json.dumps({"outdir": str(root), "runs": summaries}, indent=2))
    finally:
        try:
            pump.close()
        finally:
            sensor.close()


def cmd_check_hardware(args: argparse.Namespace) -> None:
    hw, pump, sensor = _connect_hardware(args)
    out = _ensure_outdir(args.outdir, prefix="probe") if args.outdir else None
    try:
        collect_s = max(0.0, float(args.collect_s))
        pump_state = pump.get_state() if pump.is_open() else None
        pump_factor = pump.get_factor() if pump.is_open() else None

        sensor.start_collecting()
        start = time.time()
        while time.time() - start < collect_s:
            time.sleep(0.2)
        sensor.stop_collecting()
        t_data, m_data, raw_data = sensor.get_data()

        summary = {
            "pump_connected": pump.is_open(),
            "sensor_connected": sensor.is_open(),
            "sensor_status": sensor.status_text,
            "sensor_last_error": sensor.last_error,
            "sensor_last_packet_age_s": sensor.last_packet_age,
            "sensor_samples_collected": len(t_data),
            "sensor_live_mass_g": sensor.current_reading,
            "sensor_cal_factor": sensor.calibration_factor,
            "pump_state": pump_state,
            "pump_factor": pump_factor,
        }
        if out is not None:
            _, pd, _, _, _, _, _ = _load_science()
            _write_json(out / "summary.json", summary)
            pd.DataFrame({"time_s": t_data, "mass_g": m_data, "raw_sensor": raw_data}).to_csv(out / "sensor_probe.csv", index=False)
            (out / "pump_log.txt").write_text(pump.export_log_text())
        print(json.dumps({"outdir": str(out) if out is not None else None, "summary": summary}, indent=2))
    finally:
        try:
            pump.close()
        finally:
            sensor.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoFlow pump validation operator tools")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list-ports", help="List pump/sensor serial ports and optionally BLE devices")
    p.add_argument("--ble", action="store_true", help="also scan BLE devices")
    p.add_argument("--timeout", type=float, default=6.0)
    p.set_defaults(func=cmd_list_ports)

    p = sub.add_parser("make-profile", help="Generate a pump input profile CSV")
    p.add_argument("--shape", default="bell")
    p.add_argument("--qmax", type=float, default=15.0)
    p.add_argument("--volume", type=float, default=250.0)
    p.add_argument("--duration", type=float, default=25.0)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_make_profile)

    p = sub.add_parser("make-mass-fixture", help="Generate synthetic sensor mass data from a flow profile")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--sample-rate", type=float, default=40.0)
    p.add_argument("--post-s", type=float, default=5.0)
    p.add_argument("--baseline-g", type=float, default=0.0)
    p.add_argument("--flow-scale", type=float, default=1.0)
    p.add_argument("--flow-bias", type=float, default=0.0)
    p.add_argument("--delay-s", type=float, default=0.0)
    p.add_argument("--flow-noise-sd", type=float, default=0.0)
    p.add_argument("--mass-noise-sd", type=float, default=0.0)
    p.add_argument("--drain-rate-g-s", type=float, default=0.0)
    p.add_argument("--drain-floor-g", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_make_mass_fixture)

    p = sub.add_parser("analyze", help="Analyze a mass or flow CSV and write machine-readable artifacts")
    p.add_argument("--input", required=True)
    p.add_argument("--outdir")
    p.add_argument("--calibration-map")
    p.add_argument("--expected-volume", type=float)
    p.add_argument("--sensor-cal-factor", type=float)
    p.add_argument("--source-duration", type=float)
    p.add_argument("--volume-tolerance-pct", type=float, default=20.0)
    p.add_argument("--no-plots", action="store_true")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("build-calibration-map", help="Build a drain-rate map from a fill-then-drain mass CSV")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-drop-g", type=float, default=10.0)
    p.add_argument("--peak-offset-samples", type=int, default=20)
    p.set_defaults(func=cmd_build_calibration_map)

    p = sub.add_parser("make-zero-calibration", help="Write an empty zero-drain calibration map CSV")
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_make_zero_calibration)

    hardware_parent = argparse.ArgumentParser(add_help=False)
    hardware_parent.add_argument("--pump-port")
    hardware_parent.add_argument("--sensor-target")
    hardware_parent.add_argument("--cal-factor", type=float)
    hardware_parent.add_argument("--sensor-cal-factor", type=float)
    hardware_parent.add_argument("--speed", type=float, default=1.0)
    hardware_parent.add_argument("--max-rpm", type=float)
    hardware_parent.add_argument("--calibration-map")
    hardware_parent.add_argument("--volume-tolerance-pct", type=float, default=20.0)
    hardware_parent.add_argument("--no-tare", action="store_true")
    hardware_parent.add_argument("--outdir")

    p = sub.add_parser("check-hardware", parents=[hardware_parent], help="Connect pump and sensor, then collect a short sensor probe")
    p.add_argument("--collect-s", type=float, default=3.0)
    p.set_defaults(func=cmd_check_hardware)

    p = sub.add_parser("run-exact", parents=[hardware_parent], help="Run one exact pump playback with sensor capture")
    p.add_argument("--input", required=True)
    p.set_defaults(func=cmd_run_exact)

    p = sub.add_parser("queue", parents=[hardware_parent], help="Run multiple exact profiles with drain waits")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--drain-timeout", type=float, default=120.0)
    p.add_argument("--drain-threshold", type=float, default=15.0)
    p.add_argument("--drain-stable", type=float, default=8.0)
    p.set_defaults(func=cmd_queue)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

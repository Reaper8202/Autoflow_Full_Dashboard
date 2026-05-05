# AutoFlow Autonomous Validation Context

## Goal

Build and use tooling so Codex can operate the `automated_dashboard` pump validation system with minimal supervision:

- generate targeted CSV playback profiles,
- run CSV playback against the pump,
- capture load-cell/sensor output,
- save all raw and analyzed artifacts,
- compare sensor-derived cumulative volume against the CSV profile ground truth,
- suggest calibration or code changes based on the results,
- avoid requiring the user to watch each run.

The user can help with physical setup steps, especially calibration-map collection and hardware connection if the terminal cannot access BLE directly.

## Validation Rule

CSV playback is the canonical mode. The integrated volume of the input CSV profile is the ground truth.

Current pass/fail criterion:

- Pass if sensor-derived cumulative volume is within `20%` of the CSV-integrated expected volume.
- Fail otherwise.

The current suspected issue is load-cell/sensor miscalibration, not the generated CSV ground truth. Each analysis should compute:

```text
sensor_calibration_scale = expected_csv_volume_mL / measured_sensor_volume_mL
suggested_sensor_cal_factor = current_sensor_cal_factor * sensor_calibration_scale
```

If measured volume is low, the scale will be above `1.0` and the sensor calibration factor should increase. If measured volume is high, the scale will be below `1.0`.

## Files Added

- `autoflow_ops.py`: non-Streamlit operator CLI for autonomous testing.
- `AUTOFLOW_OPS.md`: human-readable runbook.
- `context.md`: this memory file.

Important: before these files were added, the repo already had uncommitted changes in `app.py`, `analysis.py`, and tracked `__pycache__` files. Do not assume those were created by this operator-tooling work.

## CLI Capabilities

`autoflow_ops.py` currently supports:

- `list-ports`: list pump/sensor serial ports; optional BLE scan.
- `check-hardware`: connect pump and sensor, collect a short sensor probe, save `summary.json`, `sensor_probe.csv`, and `pump_log.txt`.
- `make-profile`: generate a valid pump profile CSV from `analysis.py` templates.
- `make-mass-fixture`: synthesize sensor mass data from a profile for offline debugging.
- `analyze`: analyze mass or flow CSV output and write `summary.json`, `analysis.json`, `analysis.csv`, and optional `analysis.html`.
- `build-calibration-map`: convert a fill-then-drain mass recording into `mass_g,drain_rate_g_per_s`.
- `make-zero-calibration`: create an empty zero-drain calibration map.
- `run-exact`: run one exact CSV playback task with sensor capture.
- `queue`: run multiple exact CSV playback tasks with drain waits between runs.

Hardware commands use the dashboard config at `~/.autoflow_dashboard_config.json` when `--pump-port`, `--sensor-target`, `--cal-factor`, or `--max-rpm` are omitted.

## Calibration Map Plan

The CLI can use `--calibration-map calibration_map.csv` for runs and analysis.

For a passive-drain setup:

1. Start recording mass before filling.
2. Fill to roughly the largest validation volume plus margin.
3. For current 250 mL tests, use about 300 mL / 300 g.
4. Open the drain and let it run down for several seconds.
5. Build the map with:

```bash
python3 autoflow_ops.py build-calibration-map \
  --input calibration_recording.csv \
  --output calibration_map.csv
```

For pump-only/no-passive-drain testing:

```bash
python3 autoflow_ops.py make-zero-calibration \
  --output calibration_map.csv
```

## Commands Tried And Results

Environment:

- Default `/opt/homebrew/bin/python3` initially lacked project dependencies such as `numpy`, `pandas`, `scipy`, `plotly`, and `pyserial`.
- A temporary venv was created at `/tmp/autoflow_ops_venv`.
- `pip install -r requirements.txt` initially failed under restricted network, then succeeded after approval for network access.

Syntax and help checks:

```bash
/tmp/autoflow_ops_venv/bin/python -m py_compile autoflow_ops.py
/tmp/autoflow_ops_venv/bin/python autoflow_ops.py --help
/tmp/autoflow_ops_venv/bin/python autoflow_ops.py check-hardware --help
```

Results:

- Passed.
- Help lists all expected subcommands, including `check-hardware`.

Profile generation:

```bash
/tmp/autoflow_ops_venv/bin/python autoflow_ops.py make-profile \
  --shape bell --qmax 15 --volume 250 --duration 25 \
  --samples 200 --output /tmp/autoflow_profile.csv
```

Result:

- Passed.
- Output volume was 250 mL.
- Earlier attempts with `bell`, `qmax 20` or `qmax 30`, `volume 250`, `duration 25` failed with `Target volume is too low for this shape, peak flow, and duration`; this is expected from the existing `build_run_profile` constraints. Do not loop on that combination as a bug in the CLI.

Synthetic fixture generation:

```bash
/tmp/autoflow_ops_venv/bin/python autoflow_ops.py make-mass-fixture \
  --input /tmp/autoflow_profile.csv \
  --output /tmp/autoflow_mass_review.csv \
  --sample-rate 40 --mass-noise-sd 0.2 --delay-s 0.15
```

Result:

- Passed.
- Generated 1201 samples over 30 s.
- Simulated final mass was about 249.93 g.
- A bug was fixed during review: the synthetic fixture originally clipped delayed time into the profile end and kept applying final flow during post-run tail. It now uses `np.interp(..., left=0, right=0)`.

Synthetic analysis:

```bash
/tmp/autoflow_ops_venv/bin/python autoflow_ops.py analyze \
  --input /tmp/autoflow_mass_review.csv \
  --expected-volume 250 \
  --sensor-cal-factor 0.00052587 \
  --outdir /tmp/autoflow_analysis_review \
  --no-plots
```

Result:

- Passed.
- Summary status: `pass`.
- Measured volume: about 250.99 mL.
- Volume error: about 0.39%.
- Suggested sensor factor: about `0.0005238035`.

Calibration map builder:

```bash
/tmp/autoflow_ops_venv/bin/python autoflow_ops.py build-calibration-map \
  --input /tmp/autoflow_calibration_recording.csv \
  --output /tmp/autoflow_calibration_map_review.csv
```

Result on synthetic fill/drain trace:

- Passed.
- 1576 map points.
- Mass drop about 299.55 g.
- Drain rates about -8.13 to -6.62 g/s.

Zero-drain map:

```bash
/tmp/autoflow_ops_venv/bin/python autoflow_ops.py make-zero-calibration \
  --output /tmp/zero_calibration_review.csv
```

Result:

- Passed.
- Created an empty `mass_g,drain_rate_g_per_s` CSV.

Hardware discovery:

```bash
/tmp/autoflow_ops_venv/bin/python autoflow_ops.py list-ports --ble --timeout 2
```

Result:

- Serial ports found:
  - `/dev/cu.debug-console`
  - `/dev/cu.Bluetooth-Incoming-Port`
- BLE scan failed from this terminal with:
  - `BLE scan failed: ('Bluetooth is unsupported', <BleakBluetoothNotAvailableReason.NO_BLUETOOTH: 1>)`
- The CLI now reports this as JSON instead of crashing.

Interpretation:

- The terminal process may not have BLE/CoreBluetooth access.
- If BLE remains unavailable, ask the user to connect the sensor in the Streamlit dashboard once or provide the `BLE::...` token from `~/.autoflow_dashboard_config.json`.
- Still try `check-hardware` when the pump/sensor ports or token are known.

## Current Gaps

- No real pump/sensor run has been executed yet in this session.
- The default Python environment still lacks project dependencies; the verified environment was `/tmp/autoflow_ops_venv`.
- BLE scan is blocked/unavailable from the terminal environment used here.
- The CLI can connect to hardware if given a usable pump port and sensor target, but that path has not been hardware-verified.
- Tracked `__pycache__` files are dirty because Python compilation imports touched them. Avoid treating that as meaningful source work.

## Next Best Steps

1. Get a real sensor target:
   - Try Streamlit dashboard BLE scan/connect, then read `~/.autoflow_dashboard_config.json`, or ask the user for the `BLE::...` token.
2. Run:

```bash
/tmp/autoflow_ops_venv/bin/python autoflow_ops.py check-hardware \
  --pump-port <pump_port> \
  --sensor-target <sensor_target> \
  --collect-s 3 \
  --outdir operator_runs/probe
```

3. If probe succeeds, collect/build a calibration map or create zero-drain map depending on the physical setup.
4. Generate one conservative CSV profile first, then run:

```bash
/tmp/autoflow_ops_venv/bin/python autoflow_ops.py run-exact \
  --pump-port <pump_port> \
  --sensor-target <sensor_target> \
  --input <profile.csv> \
  --calibration-map <calibration_map.csv> \
  --outdir operator_runs/<run_name>
```

5. Inspect `summary.json`, `analysis.html`, `sensor_raw.csv`, `commands.csv`, and `pump_log.txt`.
6. If volume error fails the 20% threshold, use `sensor_calibration_scale` and `suggested_sensor_cal_factor` before trying algorithmic changes.

## Do Not Revisit Unless Evidence Changes

- Do not treat invalid `bell qmax=20 volume=250 duration=25` or `qmax=30` as a CLI failure; it is rejected by the existing profile solver.
- Do not assume Streamlit interaction is required for offline analysis; `autoflow_ops.py analyze` already exercises the shared analysis pipeline.
- Do not compare against poured physical volume as the primary truth for these validation runs; the user specified the CSV integrated volume as ground truth.
- Do not rely on BLE scanning from terminal until CoreBluetooth access is resolved; use dashboard-provided sensor token if needed.

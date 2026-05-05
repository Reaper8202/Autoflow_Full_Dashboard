# AutoFlow Operator CLI

`autoflow_ops.py` is a non-Streamlit control surface for pump validation. It is intended for autonomous operation: generate profiles, synthesize troubleshooting sensor data, run hardware tasks, queue tasks, save raw outputs, and analyze results.

The validation ground truth is the integrated volume of the input CSV profile. By default a run passes when the sensor-derived cumulative volume is within `20%` of that CSV volume.

## Environment

Install the dashboard dependencies first:

```bash
python3 -m pip install -r requirements.txt
```

All commands below run from `automated_dashboard/`.

## Inspect Hardware

```bash
python3 autoflow_ops.py list-ports
python3 autoflow_ops.py list-ports --ble
```

If `--pump-port` or `--sensor-target` are omitted during hardware runs, the CLI tries the dashboard config at `~/.autoflow_dashboard_config.json`.

Before running a validation profile, probe the hardware:

```bash
python3 autoflow_ops.py check-hardware \
  --pump-port /dev/cu.usbmodemXXXX \
  --sensor-target BLE::AA:BB:CC:DD:EE:FF::AutoFlow \
  --collect-s 3 \
  --outdir operator_runs/probe
```

This writes `summary.json`, `sensor_probe.csv`, and `pump_log.txt` when `--outdir` is provided.

## Generate Input Profiles

```bash
python3 autoflow_ops.py make-profile \
  --shape bell \
  --qmax 15 \
  --volume 250 \
  --duration 25 \
  --samples 200 \
  --output operator_profiles/bell_250ml.csv
```

Supported shapes come from `analysis.SHAPES`: `bell`, `plateau`, `sawtooth`, `sinusoidal`, and `constant`.

## Create Troubleshooting Fixtures

This creates synthetic sensor mass data from a pump profile, useful for debugging analysis without hardware:

```bash
python3 autoflow_ops.py make-mass-fixture \
  --input operator_profiles/bell_250ml.csv \
  --output operator_fixtures/bell_250ml_mass.csv \
  --sample-rate 40 \
  --mass-noise-sd 0.2 \
  --delay-s 0.15
```

Useful fault knobs:

- `--flow-scale 0.8` simulates under-delivery.
- `--flow-bias 1.0` simulates a constant positive flow offset.
- `--delay-s 0.5` simulates delayed sensor response.
- `--drain-rate-g-s 2.0` simulates passive drain while filling.
- `--mass-noise-sd 1.0` simulates noisy load-cell readings.

## Analyze Existing Outputs

```bash
python3 autoflow_ops.py analyze \
  --input operator_fixtures/bell_250ml_mass.csv \
  --expected-volume 250 \
  --outdir operator_runs/debug_bell
```

Artifacts written:

- `summary.json`: compact metrics for quick inspection.
- `analysis.json`: complete arrays and zone details.
- `analysis.csv`: dashboard-compatible time series output.
- `analysis.html`: interactive Plotly plot unless `--no-plots` is passed.

`summary.json` includes `status`, `volume_error_pct`, `sensor_calibration_scale`, and `suggested_sensor_cal_factor`. The scale is computed as:

```text
expected_csv_volume_mL / measured_sensor_volume_mL
```

For example, if the CSV says 250 mL and the sensor reports 200 mL, the scale is `1.25`, so the load-cell calibration factor should increase by 25%.

## Build A Calibration Map

For a passive-drain setup, collect a mass CSV while the container is filled and then drains. Start recording first, fill to roughly the largest volume we expect to test, then open the drain and let it run down for at least several seconds. For the current 250 mL validation profiles, a 300 mL fill is a reasonable calibration target.

Then build the map:

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

## Run One Hardware Task

```bash
python3 autoflow_ops.py run-exact \
  --pump-port /dev/cu.usbmodemXXXX \
  --sensor-target BLE::AA:BB:CC:DD:EE:FF::AutoFlow \
  --input operator_profiles/bell_250ml.csv \
  --max-rpm 350 \
  --calibration-map calibration_map.csv \
  --outdir operator_runs/hardware_bell
```

Artifacts written:

- `commands.csv`: commanded flow and RPM timeline.
- `sensor_raw.csv`: captured sensor mass/raw data.
- `pump_log.txt`: pump TX/RX log.
- `summary.json`: outcome metrics.
- `analysis.csv`, `analysis.json`, `analysis.html`: generated if enough sensor samples were captured.

## Run A Queue

```bash
python3 autoflow_ops.py queue \
  --pump-port /dev/cu.usbmodemXXXX \
  --sensor-target BLE::AA:BB:CC:DD:EE:FF::AutoFlow \
  --max-rpm 350 \
  --calibration-map calibration_map.csv \
  --drain-threshold 15 \
  --drain-stable 8 \
  operator_profiles/bell_250ml.csv \
  operator_profiles/plateau_250ml.csv
```

Each run receives its own subdirectory, and `queue_summary.json` is updated after every task so partial results survive an interrupted queue.

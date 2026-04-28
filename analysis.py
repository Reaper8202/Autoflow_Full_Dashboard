"""Flow analysis: shape templates, curve fitting, KZ filter, zone detection."""

import numpy as np
from scipy.signal import lfilter


# ---------------------------------------------------------------------------
#  Shape templates (match the Feather firmware)
# ---------------------------------------------------------------------------

def tmpl_bell(u):
    if u <= 0 or u >= 1:
        return 0.0
    peak, s1, s2 = 0.35, 0.18, 0.28
    sig = s1 if u < peak else s2
    return float(np.exp(-0.5 * ((u - peak) / sig) ** 2))


def tmpl_plateau(u):
    if u <= 0 or u >= 1:
        return 0.0
    if u < 0.25:
        env = np.sin((u / 0.25) * (np.pi / 2)) ** 2
    elif u < 0.75:
        env = 1.0
    else:
        env = np.sin(((1 - u) / 0.25) * (np.pi / 2)) ** 2
    return 0.55 * float(env)


def tmpl_sawtooth(u):
    env = tmpl_bell(u)
    phase = (u * 6.0) % 1.0
    return env * (0.25 + 0.75 * phase)


def tmpl_sinusoidal(u):
    env = tmpl_bell(u)
    return env * (0.60 + 0.40 * np.sin(2 * np.pi * 5.0 * u))


def tmpl_constant(u):
    return 0.0 if u <= 0 or u >= 1 else 1.0


SHAPES = ["bell", "plateau", "sawtooth", "sinusoidal", "constant"]
TEMPLATES = {
    "bell": tmpl_bell,
    "plateau": tmpl_plateau,
    "sawtooth": tmpl_sawtooth,
    "sinusoidal": tmpl_sinusoidal,
    "constant": tmpl_constant,
}


def shape_curve(name, n=200):
    fn = TEMPLATES[name]
    u = np.linspace(0, 1, n)
    return u, np.array([fn(float(x)) for x in u])


def build_run_profile(shape, qmax, volume, duration, n=200):
    u, raw = shape_curve(shape, n=n)
    peak = float(np.max(raw))
    if peak <= 0 or qmax <= 0 or volume <= 0 or duration <= 0:
        return None, "qmax, volume, and duration must be positive"

    norm = raw / peak
    mean_shape = float(np.trapezoid(norm, u))
    avg_flow = float(volume) / float(duration)

    if avg_flow > qmax + 1e-9:
        return None, "Target volume requires average flow above peak flow"
    if mean_shape >= 1.0 - 1e-9:
        if abs(avg_flow - qmax) > 1e-6:
            return None, "Constant profile requires volume = peak flow x duration"
        q = np.full_like(u, qmax)
        return {"u": u, "q": q, "avg_flow": avg_flow, "base_flow": qmax, "peak_flow": qmax, "mean_shape": 1.0}, None

    amp_flow = (qmax - avg_flow) / (1.0 - mean_shape)
    base_flow = qmax - amp_flow
    if amp_flow < -1e-6:
        return None, "Target volume requires average flow above peak flow"
    if base_flow < -1e-6:
        return None, "Target volume is too low for this shape, peak flow, and duration"

    amp_flow = max(0.0, float(amp_flow))
    base_flow = max(0.0, float(base_flow))
    q = base_flow + amp_flow * norm
    return {"u": u, "q": q, "avg_flow": avg_flow, "base_flow": base_flow, "peak_flow": float(np.max(q)), "mean_shape": mean_shape}, None


# ---------------------------------------------------------------------------
#  Curve analysis / shape detection
# ---------------------------------------------------------------------------

def analyze_curve(t, q):
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    q = np.clip(q, 0.0, None)
    if len(t) < 3 or t[-1] <= t[0]:
        return None

    duration = float(t[-1] - t[0])
    qmax = float(np.max(q))
    volume = float(np.trapezoid(q, t))

    u = (t - t[0]) / duration
    u_uniform = np.linspace(0, 1, 200)
    q_uniform = np.interp(u_uniform, u, q)
    if qmax <= 0:
        return None
    q_norm = q_uniform / qmax

    scores = {}
    for name in SHAPES:
        _, tmpl = shape_curve(name, n=200)
        tmpl_norm = tmpl / (np.max(tmpl) + 1e-9)
        num = np.sum((q_norm - q_norm.mean()) * (tmpl_norm - tmpl_norm.mean()))
        den = np.sqrt(np.sum((q_norm - q_norm.mean()) ** 2) * np.sum((tmpl_norm - tmpl_norm.mean()) ** 2))
        scores[name] = float(num / (den + 1e-9))

    best_shape = max(scores, key=scores.get)
    return {
        "shape": best_shape,
        "qmax": qmax,
        "volume_integrated": volume,
        "duration": duration,
        "score": scores[best_shape],
        "shape_scores": scores,
        "t": t,
        "q": q,
    }


# ---------------------------------------------------------------------------
#  Filtering (matches the Flutter app's uroflowFunctions.dart)
# ---------------------------------------------------------------------------

FILTER_CUTOFF = 2.0
PRE_FLOW_TIME = 3
POST_FLOW_TIME = 5
SAMPLING_RATE = 40

# Hardcoded Butterworth coefficients used by the Flutter app.
_B = np.array([0.02008337, 0.04016673, 0.02008337])
_A = np.array([1.0, -1.56101808, 0.64135154])


def lowpass_filter(data):
    if len(data) < 6:
        return np.array(data, dtype=float)
    return np.array(lfilter(_B, _A, data), dtype=float)


def get_derivative(t, data, ss=2):
    n = len(data)
    dd = np.zeros(n)
    for i in range(ss):
        dd[i] = (data[i + ss] - data[i]) / (t[i + ss] - t[i]) if t[i + ss] != t[i] else 0.0
    for i in range(ss, n - ss):
        dd[i] = (data[i + ss] - data[i - ss]) / (t[i + ss] - t[i - ss]) if t[i + ss] != t[i - ss] else 0.0
    for i in range(n - ss, n):
        dd[i] = (data[i] - data[i - ss]) / (t[i] - t[i - ss]) if t[i] != t[i - ss] else 0.0
    return dd


def kz_filter(data, window=9, iterations=3):
    if len(data) == 0:
        return np.array([])
    filtered = np.array(data, dtype=float)
    radius = window // 2
    for _ in range(iterations):
        result = np.zeros_like(filtered)
        for i in range(len(filtered)):
            lo = max(0, i - radius)
            hi = min(len(filtered) - 1, i + radius)
            vals = filtered[lo:hi + 1]
            finite = vals[np.isfinite(vals)]
            result[i] = np.mean(finite) if len(finite) > 0 else 0.0
        filtered = result
    return filtered


# ---------------------------------------------------------------------------
#  Zone detection (matches the Flutter app)
# ---------------------------------------------------------------------------

def identify_zones(raw_data, filt_mass_rate, calibration_map, filt_mass):
    mass_thresh = 3.0
    flow_rate_thresh = 2.0
    window_len = 5
    voiding, draining, empty = [], [], []

    for zone in range(0, len(raw_data), window_len):
        end = min(zone + window_len, len(raw_data))
        temp_data = raw_data[zone:end]
        temp_rate = filt_mass_rate[zone:end]
        avg_temp = np.mean(temp_data)
        drain = _corresponding_drain_rate(calibration_map, avg_temp)

        all_above_mass = all(x > mass_thresh for x in temp_data)
        if all_above_mass and all((r - drain) > flow_rate_thresh for r in temp_rate):
            voiding.append([zone, end])
        elif all_above_mass and all((r - drain) < -flow_rate_thresh for r in temp_rate):
            draining.append([zone, end])
        elif all_above_mass:
            draining.append([zone, end])
        else:
            empty.append([zone, end])

    return empty, voiding, draining


def _corresponding_drain_rate(mapping, mass):
    if len(mapping) < 2:
        return 0.0
    masses, rates = mapping[0], mapping[1]
    if len(masses) == 0 or len(rates) == 0 or len(masses) != len(rates):
        return 0.0
    is_desc = len(masses) < 2 or masses[0] >= masses[-1]
    m = masses if is_desc else masses[::-1]
    r = rates if is_desc else rates[::-1]
    if mass <= m[-1]:
        return 0.0
    if mass >= m[0]:
        return r[0]
    for i in range(len(m)):
        if mass >= m[i]:
            return r[i]
    return 0.0


def compute_flow_from_mass(t, y, calibration_map):
    """Full flow-rate pipeline matching the Flutter app."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) < 10:
        return y, np.zeros_like(y), np.zeros_like(y), np.zeros_like(y), [], [], [], [0, max(0, len(y) - 1)]

    filt_mass = lowpass_filter(y)
    derivative = get_derivative(t, filt_mass, ss=2)
    filt_mass_rate = lowpass_filter(derivative)

    empty, voiding, draining = identify_zones(y, filt_mass_rate, calibration_map, filt_mass)

    n = min(len(y), len(filt_mass_rate), len(filt_mass))
    inflow = np.zeros(n)
    for i in range(n):
        drain = _corresponding_drain_rate(calibration_map, filt_mass[i])
        inflow[i] = max(0.0, filt_mass_rate[i] - drain)

    filt_inflow = lowpass_filter(inflow)
    kz_flow = kz_filter(filt_inflow, window=9, iterations=3)

    cum_volume = np.zeros(n)
    acc = 0.0
    for i in range(n - 1):
        dt = t[i + 1] - t[i]
        if dt > 0:
            acc += 0.5 * (filt_inflow[i + 1] + filt_inflow[i]) * dt
        cum_volume[i] = acc
    if n >= 2:
        cum_volume[-1] = cum_volume[-2]

    if voiding:
        start = max(0, voiding[0][0] - PRE_FLOW_TIME * SAMPLING_RATE)
        end = min(n - 1, voiding[-1][1] + POST_FLOW_TIME * SAMPLING_RATE)
    else:
        start = 0
        end = min(n - 1, empty[-1][1] if empty else max(0, n - 1))
    roi = [int(start), int(end)]

    return filt_mass, filt_inflow, kz_flow, cum_volume, empty, voiding, draining, roi

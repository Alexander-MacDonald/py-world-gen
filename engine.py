from __future__ import annotations
import math
from typing import Callable, Dict, List, Optional, Tuple, Any

# This module provides a small, data-driven terrain engine scaffold:
# - Noise registry + unified sampling API
# - Modifiers chain
# - Combiners/selectors
# - Advanced domain warping (multi-stage)
# - Erosion/Stamps/Biome graph/Distributions stubs

# Consumers are expected to provide/forward a permutation table `perm`
# compatible with Perlin hashing (length 512, ints 0..255).

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v

def lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)

def remap(x: float, in0: float, in1: float, out0: float, out1: float) -> float:
    if abs(in1 - in0) < 1e-12:
        return out0
    t = (x - in0) / (in1 - in0)
    return out0 + t * (out1 - out0)

def smoothmin(a: float, b: float, k: float) -> float:
    # polynomial smooth min
    if k <= 0:
        return min(a, b)
    h = clamp(0.5 + 0.5*(b - a)/k, 0.0, 1.0)
    return lerp(b, a, h) - k*h*(1.0 - h)

# ------------------------------------------------------------
# Base Perlin 2D and helpers
# (Callers in repo already have Perlin2D/make_perm; we redefine here for isolation.)
# ------------------------------------------------------------

_GRADS_2D = [
    (1,0), (-1,0), (0,1), (0,-1),
    (1,1), (-1,1), (1,-1), (-1,-1)
]

def _fade(t: float) -> float:
    return t * t * t * (t * (t * 6 - 15) + 10)

def _grad(hashv: int, x: float, y: float) -> float:
    g = _GRADS_2D[hashv & 7]
    return g[0]*x + g[1]*y

def perlin2d(x: float, y: float, perm: List[int]) -> float:
    xi = int(math.floor(x)) & 255
    yi = int(math.floor(y)) & 255
    xf = x - math.floor(x)
    yf = y - math.floor(y)
    u = _fade(xf)
    v = _fade(yf)

    aa = perm[(perm[xi] + yi) & 255]
    ab = perm[(perm[xi] + yi + 1) & 255]
    ba = perm[(perm[xi + 1] + yi) & 255]
    bb = perm[(perm[xi + 1] + yi + 1) & 255]

    x1 = (1-u)*_grad(aa, xf, yf) + u*_grad(ba, xf - 1, yf)
    x2 = (1-u)*_grad(ab, xf, yf - 1) + u*_grad(bb, xf - 1, yf - 1)
    return (1-v)*x1 + v*x2

def fbm_perlin(x: float, y: float, base_freq: float, octaves: int, lacunarity: float, gain: float, perm: List[int]) -> float:
    amp = 1.0
    freq = base_freq
    total = 0.0
    norm = 0.0
    for _ in range(max(1, octaves)):
        n = perlin2d(x * freq, y * freq, perm)
        total += n * amp
        norm += amp
        freq *= lacunarity
        amp *= gain
    return total / norm if norm else 0.0

# ------------------------------------------------------------
# Additional noise families
# ------------------------------------------------------------

def billow(x: float, y: float, base_freq: float, octaves: int, lacunarity: float, gain: float, perm: List[int]) -> float:
    # Soft rounded hills: use |Perlin| with fBm structure
    amp = 1.0
    freq = base_freq
    total = 0.0
    norm = 0.0
    for _ in range(max(1, octaves)):
        n = perlin2d(x * freq, y * freq, perm)
        n = 2.0*abs(n) - 1.0
        total += n * amp
        norm += amp
        freq *= lacunarity
        amp *= gain
    return total / norm if norm else 0.0

def ridged_mf(x: float, y: float, base_freq: float, octaves: int, lacunarity: float, gain: float, perm: List[int], offset: float = 1.0) -> float:
    # Ridged multifractal (simple): 1 - |noise| with gain/lacunarity
    # Based on Musgrave's ridged fractal; weight emphasizes sharp ridges
    signal = 0.0
    weight = 1.0
    freq = base_freq
    result = 0.0
    for _ in range(max(1, octaves)):
        n = perlin2d(x * freq, y * freq, perm)
        n = abs(n)
        n = offset - n
        n *= n
        n *= weight
        result += n
        weight = clamp(n * gain * 2.0, 0.0, 1.0)
        freq *= lacunarity
    # normalize roughly to [-1,1] by mapping result ~[0..~something] to [-1..1]
    return result * 2.0 - 1.0

def hybrid_mf(x: float, y: float, base_freq: float, octaves: int, lacunarity: float, gain: float, perm: List[int], offset: float = 0.7) -> float:
    # Musgrave Hybrid Multifractal: value += weight * signal; weight = clamp(signal,0,1)
    value = 0.0
    weight = 1.0
    freq = base_freq
    for _ in range(max(1, octaves)):
        signal = perlin2d(x * freq, y * freq, perm)
        signal = (signal + offset) * gain
        value += weight * signal
        weight = clamp(signal, 0.0, 1.0)
        freq *= lacunarity
    # value is roughly 0..some positive; remap to [-1,1]
    return value * 0.5  # heuristic scaling; consumers may remap

def worley2d(x: float, y: float, base_freq: float, jitter: float, perm: List[int]) -> float:
    # Basic cellular/Worley distance-to-nearest feature in 2D grid
    # Evaluate in cell of (x*freq,y*freq) and neighbors
    xf = x * base_freq
    yf = y * base_freq
    ix = math.floor(xf)
    iy = math.floor(yf)
    d2_min = 1e9
    # Examine 3x3 neighborhood
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cx = ix + dx
            cy = iy + dy
            # hash cell to pseudo-random offset in [0,1)
            h = perm[(cx & 255)] ^ perm[(cy & 255)]
            rx = ((perm[(h + 37) & 255] & 255) / 255.0) - 0.5
            ry = ((perm[(h + 113) & 255] & 255) / 255.0) - 0.5
            fx = cx + 0.5 + rx * jitter
            fy = cy + 0.5 + ry * jitter
            dxp = xf - fx
            dyp = yf - fy
            d2 = dxp*dxp + dyp*dyp
            if d2 < d2_min:
                d2_min = d2
    d = math.sqrt(d2_min)
    # normalize distance roughly to [0,1] by clamping radius ~sqrt(2)
    d_norm = clamp(d / 1.41421356, 0.0, 1.0)
    return d_norm * 2.0 - 1.0

def fbm_over_worley(x: float, y: float, base_freq: float, octaves: int, lacunarity: float, gain: float, perm: List[int], worley_freq: Optional[float] = None, jitter: float = 0.9) -> float:
    n = fbm_perlin(x, y, base_freq, octaves, lacunarity, gain, perm)  # [-1,1]
    c = worley2d(x, y, worley_freq if worley_freq is not None else base_freq, jitter, perm)  # [-1,1]
    # convert to [0,1]
    n01 = 0.5 * (n + 1.0)
    c01 = 0.5 * (c + 1.0)
    h = n01 - 0.5 * c01
    return h * 2.0 - 1.0

# ------------------------------------------------------------
# Noise registry and sampling
# ------------------------------------------------------------

NoiseFn = Callable[..., float]

NOISE_REGISTRY: Dict[str, Callable[..., float]] = {
    "perlin": perlin2d,
    "fbm": fbm_perlin,
    "billow": billow,
    "ridged": ridged_mf,
    "cellular": worley2d,
    "hybrid": hybrid_mf,
    "fbm_over_cell": fbm_over_worley,
}

def SampleNoise(profile: Dict[str, Any], x: float, y: float, perm: List[int], override_freq: Optional[float] = None) -> float:
    kind = profile.get("noise", "fbm")
    fn = NOISE_REGISTRY.get(kind)
    if fn is None:
        raise KeyError(f"Unknown noise kind: {kind}")
    # Common params with defaults
    freq = override_freq if override_freq is not None else profile.get("freq", 1.0)
    octaves = int(profile.get("octaves", 4))
    lac = float(profile.get("lacunarity", 2.0))
    gain = float(profile.get("gain", 0.5))
    offset = float(profile.get("offset", 0.7))
    jitter = float(profile.get("jitter", 0.9))

    if kind == "perlin":
        return fn(x * freq, y * freq, perm)
    if kind == "fbm":
        return fn(x, y, freq, octaves, lac, gain, perm)
    if kind == "billow":
        return fn(x, y, freq, octaves, lac, gain, perm)
    if kind == "ridged":
        return fn(x, y, freq, octaves, lac, gain, perm, offset)
    if kind == "cellular":
        return fn(x, y, freq, jitter, perm)
    if kind == "hybrid":
        return fn(x, y, freq, octaves, lac, gain, perm, offset)
    if kind == "fbm_over_cell":
        wfreq = profile.get("worley_freq")
        return fn(x, y, freq, octaves, lac, gain, perm, wfreq, jitter)
    # fallback
    return fn(x, y, freq, octaves, lac, gain, perm)

# ------------------------------------------------------------
# Modifiers
# ------------------------------------------------------------

def terrace(v: float, steps: int) -> float:
    if steps <= 1:
        return v
    t = (v + 1.0) * 0.5  # to [0,1]
    step = 1.0 / (steps - 1)
    k = round(t / step)
    t2 = k * step
    return t2 * 2.0 - 1.0

def ApplyModifiers(value: float, mods: Optional[List[Dict[str, Any]]], x: Optional[float] = None, y: Optional[float] = None, perm: Optional[List[int]] = None) -> float:
    if not mods:
        return value
    v = value
    for m in mods:
        mtype = m.get("type")
        if mtype == "abs":
            v = abs(v)
        elif mtype == "pow":
            k = float(m.get("k", 1.0))
            v = math.copysign(abs(v) ** k, v)
        elif mtype == "terrace":
            steps = int(m.get("steps", 4))
            v = terrace(v, steps)
        elif mtype == "clamp":
            lo = float(m.get("lo", -1.0))
            hi = float(m.get("hi", 1.0))
            v = clamp(v, lo, hi)
        elif mtype == "remap":
            in0 = float(m.get("in0", -1.0))
            in1 = float(m.get("in1", 1.0))
            out0 = float(m.get("out0", -1.0))
            out1 = float(m.get("out1", 1.0))
            v = remap(v, in0, in1, out0, out1)
        elif mtype == "add_noise":
            if x is None or y is None or perm is None:
                continue
            prof = m.get("profile", {"noise": "fbm", "freq": 8.0, "octaves": 2, "gain": 0.5})
            amt = float(m.get("amount", 0.1))
            v += amt * SampleNoise(prof, x, y, perm)
        # ignore unknown types for forward-compat
    return v

# ------------------------------------------------------------
# Combiners / selectors
# ------------------------------------------------------------

def Combine(a: float, b: float, mode: str, mask: Optional[float] = None) -> float:
    if mode == "add":
        return a + b
    if mode == "mul":
        return a * b
    if mode == "max":
        return a if a >= b else b
    if mode == "min":
        return a if a <= b else b
    if mode == "lerp":
        m = 0.0 if mask is None else mask
        return a * (1.0 - m) + b * m
    return b

# ------------------------------------------------------------
# Advanced domain warping (multi-stage)
# ------------------------------------------------------------

def ApplyWarp(nx: float, ny: float, warp_stages: Optional[List[Dict[str, Any]]], perm: List[int]) -> Tuple[float, float]:
    if not warp_stages:
        return nx, ny
    x, y = nx, ny
    for w in warp_stages:
        prof = {"noise": w.get("noise", "fbm"),
                "freq": float(w.get("freq", 1.0)),
                "octaves": int(w.get("octaves", 3)),
                "gain": float(w.get("gain", 0.5)),
                "lacunarity": float(w.get("lacunarity", 2.0))}
        strength = float(w.get("strength", 0.25))
        dvx = SampleNoise(prof, x, y, perm)
        dvy = SampleNoise(prof, x + 100.0, y - 100.0, perm)
        x += dvx * strength
        y += dvy * strength
    return x, y

# ------------------------------------------------------------
# Erosion (stubs)
# ------------------------------------------------------------

def ThermalRelax(heightfield: List[List[float]], talus: float) -> None:
    # Simple slope-limiting relax (very naive)
    w = len(heightfield)
    h = len(heightfield[0]) if w else 0
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    changes: List[Tuple[int,int,float]] = []
    for x in range(w):
        for y in range(h):
            h0 = heightfield[x][y]
            for dx,dy in dirs:
                nx, ny = x+dx, y+dy
                if 0 <= nx < w and 0 <= ny < h:
                    diff = h0 - heightfield[nx][ny]
                    if diff > talus:
                        move = 0.5 * (diff - talus)
                        changes.append((x,y,-move))
                        changes.append((nx,ny,move))
    for x,y,dh in changes:
        heightfield[x][y] += dh

def HydraulicStep(heightfield: List[List[float]], rain: float) -> None:
    # Placeholder: light diffusion as a cheap stand-in
    w = len(heightfield)
    h = len(heightfield[0]) if w else 0
    out = [[heightfield[x][y] for y in range(h)] for x in range(w)]
    for x in range(w):
        for y in range(h):
            acc = heightfield[x][y] * 4.0
            cnt = 4.0
            for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < w and 0 <= ny < h:
                    acc += heightfield[nx][ny]
                    cnt += 1.0
            out[x][y] = (1.0 - rain) * heightfield[x][y] + rain * (acc / cnt)
    for x in range(w):
        for y in range(h):
            heightfield[x][y] = out[x][y]

# ------------------------------------------------------------
# Feature stamps (minimal)
# ------------------------------------------------------------

def StampShape(heightfield: List[List[float]], cx: int, cy: int, radius: int, strength: float = 0.25, shape: str = "cone") -> None:
    w = len(heightfield)
    h = len(heightfield[0]) if w else 0
    r2 = radius * radius
    for x in range(max(0, cx - radius), min(w, cx + radius + 1)):
        for y in range(max(0, cy - radius), min(h, cy + radius + 1)):
            dx = x - cx
            dy = y - cy
            d2 = dx*dx + dy*dy
            if d2 > r2:
                continue
            d = math.sqrt(d2) / radius
            if shape == "cone":
                s = 1.0 - d
            elif shape == "crater":
                s = (1.0 - d)
                s = s - 0.5  # rim/bowl hint
            else:
                s = 1.0 - d
            heightfield[x][y] += s * strength

# ------------------------------------------------------------
# Biome graph classification (minimal)
# ------------------------------------------------------------

def BiomeFromGrid(temp01: float, moist01: float, table: List[List[str]]) -> str:
    ti = int(clamp(temp01, 0.0, 1.0) * max(0, len(table) - 1))
    row = table[ti]
    if not row:
        return "unknown"
    mi = int(clamp(moist01, 0.0, 1.0) * max(0, len(row) - 1))
    return row[mi]

# ------------------------------------------------------------
# Distributions
# ------------------------------------------------------------

def ApplyDistribution(val: float, dist: Optional[Dict[str, Any]]) -> float:
    if not dist:
        return val
    t = dist.get("type")
    if t == "pow":
        k = float(dist.get("k", 1.0))
        return val ** k
    if t == "sigmoid":
        mid = float(dist.get("mid", 0.5))
        steep = float(dist.get("steep", 6.0))
        return 1.0 / (1.0 + math.exp(-steep * (val - mid)))
    return val

# ------------------------------------------------------------
# Node-graph (skeleton)
# ------------------------------------------------------------

def eval_node(name: str, nodes: Dict[str, Dict[str, Any]], cache: Dict[str, Any], perm: List[int], sampler: Callable[..., float]) -> Any:
    if name in cache:
        return cache[name]
    node = nodes[name]
    op = node.get("op")
    if op == "noise":
        # produce a sampled field lazily; here we return the node spec for caller to rasterize
        field = ("noise", node)
    elif op == "warp":
        field = ("warp", node)
    elif op == "blend":
        field = ("blend", node)
    else:
        field = ("unknown", node)
    cache[name] = field
    return field


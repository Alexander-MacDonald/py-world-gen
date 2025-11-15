#!/usr/bin/env python3
"""
Improved worldgen → isometric renderer

Changes from previous version:
- strong island/radial falloff so we get a clear landmass
- better height → color mapping (beach, grass, rock, snow)
- biomes no longer paint everything white
- iso renderer draws top + two side faces for height illusion
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Any, Set, Tuple
from PIL import Image, ImageDraw
from typing import Any as _Any  # local alias to avoid clash

# New engine scaffold (noise registry, modifiers, combiners, warp, etc.)
from engine import (
    NOISE_REGISTRY,
    SampleNoise,
    ApplyModifiers,
    Combine,
    ApplyWarp,
    ThermalRelax,
    HydraulicStep,
    StampShape,
    BiomeFromGrid,
    ApplyDistribution,
)

# ============================================================
# 1. Deterministic RNG: SplitMix64
# ============================================================

class SplitMix64:
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
        z = z ^ (z >> 31)
        return z & 0xFFFFFFFFFFFFFFFF

    def rand_float(self) -> float:
        return (self.next() >> 11) * (1.0 / (1 << 53))

def stable_hash(s: str) -> int:
    h = 0xcbf29ce484222325
    for c in s.encode("utf-8"):
        h ^= c
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h

def PRNG(seed: int, stream_id: str) -> SplitMix64:
    return SplitMix64(seed ^ stable_hash(stream_id))


# ============================================================
# 2. Perlin + fBm
# ============================================================

_GRADS_2D = [
    (1,0), (-1,0), (0,1), (0,-1),
    (1,1), (-1,1), (1,-1), (-1,-1)
]

def _fade(t: float) -> float:
    return t * t * t * (t * (t * 6 - 15) + 10)

def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)

def _grad(hashv: int, x: float, y: float) -> float:
    g = _GRADS_2D[hashv & 7]
    return g[0]*x + g[1]*y

def Perlin2D(x: float, y: float, perm: List[int]) -> float:
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

    x1 = _lerp(_grad(aa, xf, yf), _grad(ba, xf - 1, yf), u)
    x2 = _lerp(_grad(ab, xf, yf - 1), _grad(bb, xf - 1, yf - 1), u)
    return _lerp(x1, x2, v)

def make_perm(rng: SplitMix64) -> List[int]:
    p = list(range(256))
    for i in range(255, 0, -1):
        j = int(rng.rand_float() * (i+1))
        p[i], p[j] = p[j], p[i]
    return p * 2

def fBm(x: float, y: float, base_freq: float, octaves: int, lacunarity: float, gain: float, perm: List[int]) -> float:
    amp = 1.0
    freq = base_freq
    total = 0.0
    norm = 0.0
    for _ in range(octaves):
        n = Perlin2D(x * freq, y * freq, perm)
        total += n * amp
        norm += amp
        freq *= lacunarity
        amp *= gain
    return total / norm if norm else 0.0

def DomainWarp(x: float, y: float, warp_freq: float, warp_strength: float, perm: List[int]) -> Tuple[float, float]:
    dx = fBm(x, y, warp_freq, 3, 2.0, 0.5, perm)
    dy = fBm(x+100.0, y-100.0, warp_freq, 3, 2.0, 0.5, perm)
    return (x + warp_strength * dx, y + warp_strength * dy)


# ============================================================
# 3. Data structures
# ============================================================

from dataclasses import dataclass, field

@dataclass
class Tile:
    x: int
    y: int
    flags: Set[str] = field(default_factory=set)
    props: Dict[str, float] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)

@dataclass
class World:
    width: int
    height: int
    tiles: List[List[Tile]] = field(init=False)
    cache: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.tiles = [[Tile(x, y) for y in range(self.height)] for x in range(self.width)]

    def tile(self, x: int, y: int) -> Tile:
        return self.tiles[x][y]

@dataclass
class WorldGenParams:
    width: int
    height: int
    seed: int = 42
    force_island: bool = True
    noise_domain_scale: dict = field(default_factory=lambda: {
        "x": 1.0,   # >1.0 = zoom IN horizontally (more detail, thinner features)
        "y": 1.0,   # <1.0 = zoom OUT vertically (wider N-S bands)
    })
    scales: Dict[str, float] = field(default_factory=lambda: {
        "continent": 0.3,
        "mountain_range": 0.04,
        "hill": 0.015,
        "detail": 0.006,
    })
    warp_params: Dict[str, float] = field(default_factory=lambda: {
        "strength": 0.25,
        "frequency": 1.3,
    })
    climate_params: Dict[str, float] = field(default_factory=lambda: {
        "sea_level": -0.08,        # lower than before → more land
        "lapse_rate": 6.5,
        "base_equator_temp": 30.0,
        "pole_temp": -15.0,
        "humidity_base": 0.55,
    })
    river_params: Dict[str, float] = field(default_factory=lambda: {
        "min_accum_flow": 0.45,
        "carve_strength": 0.035,
    })
    # New: configurable noise profiles (data-driven)
    noise_profiles: Dict[str, Dict[str, _Any]] = field(default_factory=lambda: {
        # larger continents, mild warp built-in
        "continent": {"noise": "fbm", "freq": 2.0, "octaves": 5, "gain": 0.5, "lacunarity": 2.0,
                       "warp_stages": [
                           {"noise": "fbm", "freq": 1.2, "strength": 0.06, "octaves": 3}
                       ]},
        # higher-frequency ranges with stronger warp for rugged mountains
        "mountains": {"noise": "ridged", "freq": 1.0/0.04, "octaves": 5, "gain": 0.5, "lacunarity": 2.0,
                       "warp_stages": [
                           {"noise": "fbm", "freq": 1.6, "strength": 0.08, "octaves": 3},
                           {"noise": "ridged", "freq": 3.5, "strength": 0.03, "octaves": 3}
                       ]},
        "hills":     {"noise": "billow", "freq": 1.0/0.015, "octaves": 3, "gain": 0.5, "lacunarity": 2.0},
        "detail":    {"noise": "fbm", "freq": 1.0/0.006, "octaves": 2, "gain": 0.5, "lacunarity": 2.0},
        # Example: fbm over cellular for badlands:
        "badlands":  {"noise": "fbm_over_cell", "freq": 2.2, "octaves": 4, "gain": 0.5, "lacunarity": 2.0, "jitter": 0.9, "worley_freq": 2.2},
        # Fine dunes for deserts
        "dunes":     {"noise": "billow", "freq": 12.0, "octaves": 3, "gain": 0.55, "lacunarity": 2.2},
        # Small-scale ridges to enhance mountains
        "ridges_detail": {"noise": "ridged", "freq": 6.0, "octaves": 3, "gain": 0.6, "lacunarity": 2.0},
        # Subtle temperature variation (degrees C scale later)
        "temperature_noise": {"noise": "fbm", "freq": 0.8, "octaves": 2, "gain": 0.5, "lacunarity": 2.0},
    })
    # New: global (optional) warp stages for advanced domain warping
    # softer global warp; per-profile warp can override this
    warp_stages: List[Dict[str, _Any]] = field(default_factory=lambda: [
        {"noise": "fbm", "freq": 1.3, "strength": 0.08, "octaves": 3},
        {"noise": "ridged", "freq": 3.5, "strength": 0.03, "octaves": 3},
    ])
    # New: erosion profile (disabled by default)
    erosion_profile: Dict[str, _Any] = field(default_factory=lambda: {
        "enabled": False,
        "type": "thermal_then_hydro",
        "iterations": 0,
        "talus": 0.6,
        "rain": 0.2,
    })
    # New: feature stamps parameters (empty by default)
    feature_params: List[Dict[str, _Any]] = field(default_factory=list)
    # New: biome grid (optional). If provided, used by BiomeClassifyGrid.
    biome_grid: Dict[str, _Any] = field(default_factory=lambda: {
        "temp_count": 6,
        "moist_count": 6,
        # temperature rows (cold -> hot), moisture cols (dry -> wet)
        "table": [
            ["polar_ice","polar_desert","tundra","tundra","boreal_forest","boreal_forest"],
            ["cold_desert","tundra","boreal_forest","boreal_forest","boreal_forest","mixed_forest"],
            ["temperate_desert","steppe","grassland","shrubland","temperate_forest","temperate_rainforest"],
            ["temperate_desert","steppe","grassland","savanna","temperate_forest","temperate_rainforest"],
            ["subtropical_desert","semi_arid","savanna","seasonal_forest","tropical_dry_forest","tropical_rainforest"],
            ["sand_desert","semi_arid","savanna","monsoon_forest","tropical_forest","tropical_rainforest"],
        ],
    })
    # New: distributions
    distribution_profiles: Dict[str, Dict[str, _Any]] = field(default_factory=lambda: {
        # examples
        # "land_height": {"type": "pow", "k": 1.4},
        # "forest_density": {"type": "sigmoid", "mid": 0.5, "steep": 6},
        "moisture": {"type": "sigmoid", "mid": 0.55, "steep": 4.0},
    })


# ============================================================
# 4. Helpers
# ============================================================

def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v

def smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3 - 2 * t)

def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    idx = int(p * (len(vs) - 1))
    return vs[idx]

def LatFromY(y: int, height: int) -> float:
    ny = (y / (height - 1)) * 2 - 1
    return -ny * 90.0

def noise_coords(x: int, y: int, world: World, p: WorldGenParams) -> tuple[float, float]:
    # base normalized coords
    nx = x / world.width
    ny = y / world.height

    sx = p.noise_domain_scale.get("x", 1.0)
    sy = p.noise_domain_scale.get("y", 1.0)

    # scale around center so the map doesn’t “slide” off edges
    # (0.5,0.5) stays fixed
    nx = (nx - 0.5) * sx + 0.5
    ny = (ny - 0.5) * sy + 0.5
    return nx, ny

# Helper: per-profile warp override
def warp_coords_for_profile(nx: float, ny: float, profile: Dict[str, _Any] | None, p: WorldGenParams, perm: List[int]) -> tuple[float,float]:
    if profile and isinstance(profile.get("warp_stages"), list) and profile["warp_stages"]:
        return ApplyWarp(nx, ny, profile["warp_stages"], perm)
    return ApplyWarp(nx, ny, p.warp_stages, perm)

def HypsometricRemap(x: float) -> float:
    # more water, sharper rise
    if x < 0.5:
        return (x / 0.5) ** 2 * 0.4     # 0..0.4
    else:
        t = (x - 0.5) / 0.5
        return 0.4 + 0.6 * (t ** 0.6)   # 0.4..1.0

def island_falloff(nx: float, ny: float, strength: float = 2.2) -> float:
    # radial: 1 at center, 0 at edge
    # nx,ny in [0..1]
    dx = nx - 0.5
    dy = ny - 0.5
    d = math.sqrt(dx*dx + dy*dy) * 2  # 0..~1.4
    fall = clamp(1 - d ** strength, 0.0, 1.0)
    return fall


# ============================================================
# 5. Layers (simplified)
# ============================================================

class Layer:
    def name(self) -> str:
        return self.__class__.__name__
    def dependencies(self) -> List[str]:
        return []
    def compute(self, world: World, p: WorldGenParams, perm: List[int]):
        raise NotImplementedError


class ContinentMask(Layer):
    def compute(self, world, p, perm):
        for x in range(world.width):
            for y in range(world.height):
                nx, ny = noise_coords(x, y, world, p)
                # Per-profile mild warp for continents
                cprof = p.noise_profiles["continent"]
                wx, wy = warp_coords_for_profile(nx, ny, cprof, p, perm)
                # Sample via registry profile
                base = SampleNoise(cprof, wx, wy, perm)
                # Bias toward land a bit to reduce vast oceans
                base = clamp(base + 0.08, -1.0, 1.0)
                if p.force_island:
                    radial = island_falloff(x / world.width, y / world.height, strength=2.4)
                    mask = 0.6 * (base * 0.5 + 0.5) + 0.4 * radial
                    mask = mask * 2 - 1
                else:
                    mask = base
                world.tile(x, y).props["continent_mask"] = mask


class BaseHeight(Layer):
    def dependencies(self): return ["ContinentMask"]

    def compute(self, world, p, perm):
        for x in range(world.width):
            for y in range(world.height):
                t = world.tile(x, y)
                nx, ny = noise_coords(x, y, world, p)
                # Use per-profile warps where provided
                mountains_prof = p.noise_profiles["mountains"]
                hills_prof = p.noise_profiles["hills"]
                detail_prof = p.noise_profiles["detail"]
                wxm, wym = warp_coords_for_profile(nx, ny, mountains_prof, p, perm)
                wxh, wyh = warp_coords_for_profile(nx, ny, hills_prof, p, perm)
                wxd, wyd = warp_coords_for_profile(nx, ny, detail_prof, p, perm)
                mountains = SampleNoise(mountains_prof, wxm, wym, perm)
                hills     = SampleNoise(hills_prof,     wxh, wyh, perm)
                detail    = SampleNoise(detail_prof,    wxd, wyd, perm)
                mask      = t.props["continent_mask"]
                # slightly easier to become land
                landness  = smoothstep(-0.35, 0.1, mask)
                base_h    = 0.5*mountains + 0.35*hills + 0.15*detail
                ocean_h   = -0.9 + 0.3*detail
                height    = ocean_h*(1 - landness) + base_h*landness
                # Modifiers chain including microdetail add_noise using detail profile
                mods = [
                    {"type": "pow", "k": 0.95},
                    {"type": "add_noise", "profile": p.noise_profiles.get("detail", {"noise":"fbm","freq":8.0}), "amount": 0.04},
                    {"type": "clamp", "lo": -1.0, "hi": 1.0},
                ]
                # Use detail-warped coords for microdetail add_noise seeding
                height = ApplyModifiers(height, mods, x=wxd, y=wyd, perm=perm)
                t.props["height_raw"] = height


class NormalizeHeight(Layer):
    def dependencies(self): return ["BaseHeight"]

    def compute(self, world, p, perm):
        vals = [world.tile(x,y).props["height_raw"]
                for x in range(world.width)
                for y in range(world.height)]
        lo = percentile(vals, 0.02)
        hi = percentile(vals, 0.98)
        sea_level = p.climate_params["sea_level"]
        for x in range(world.width):
            for y in range(world.height):
                t = world.tile(x, y)
                h = t.props["height_raw"]
                if hi - lo < 1e-5:
                    h = 0.0
                else:
                    h = (h - lo) / (hi - lo)
                    h = HypsometricRemap(h)
                    h = h * 2 - 1
                    h = math.copysign(abs(h) ** 0.75, h)
                # Optional distribution on land height
                h = h
                dist_prof = p.distribution_profiles.get("land_height") if hasattr(p, "distribution_profiles") else None
                if dist_prof:
                    h01 = clamp((h + 1.0) * 0.5, 0.0, 1.0)
                    h01 = ApplyDistribution(h01, dist_prof)
                    h = h01 * 2.0 - 1.0
                t.props["height"] = h
                if h < sea_level:
                    t.flags.add("is_ocean")
                # beach band
                if abs(h - sea_level) < 0.015:
                    t.tags.add("beach")


class MoistureLike(Layer):
    def dependencies(self): return ["NormalizeHeight"]

    def compute(self, world, p, perm):
        for x in range(world.width):
            for y in range(world.height):
                t = world.tile(x, y)
                nx = x / world.width
                ny = y / world.height
                # use a simple profile for moisture field
                n = SampleNoise({"noise": "fbm", "freq": 2.4, "octaves": 3, "gain": 0.5, "lacunarity": 2.0}, nx, ny, perm)
                base = p.climate_params["humidity_base"]
                moist = base + 0.35*n
                if "is_ocean" in t.flags:
                    moist += 0.3
                moist = clamp(moist, 0.0, 1.0)
                # Optional distribution shaping
                mprof = p.distribution_profiles.get("moisture") if hasattr(p, "distribution_profiles") else None
                if mprof:
                    moist = ApplyDistribution(moist, mprof)
                t.props["moisture"] = moist


class DesertMask(Layer):
    def dependencies(self): return ["Temperature"]

    def compute(self, world, p, perm):
        sea = p.climate_params["sea_level"]
        for x in range(world.width):
            for y in range(world.height):
                t = world.tile(x, y)
                if "is_ocean" in t.flags:
                    t.props["desert_mask"] = 0.0
                    continue
                temp = t.props["temperature"]
                moist = t.props["moisture"]
                h = t.props["height"]
                # dryness from temp and inverse moisture
                hot = clamp((temp - 10.0) / 20.0, 0.0, 1.0)
                dry = clamp((0.5 - moist) * 2.0, 0.0, 1.0)
                land = 1.0 if h > sea + 0.02 else 0.0
                t.props["desert_mask"] = hot * dry * land


class MountainMask(Layer):
    def dependencies(self): return ["NormalizeHeight"]

    def compute(self, world, p, perm):
        for x in range(world.width):
            for y in range(world.height):
                h = world.tile(x,y).props["height"]
                world.tile(x,y).props["mountain_mask"] = smoothstep(0.55, 0.8, h)


class DunesField(Layer):
    def dependencies(self): return ["DesertMask"]

    def compute(self, world, p, perm):
        prof = p.noise_profiles["dunes"]
        for x in range(world.width):
            for y in range(world.height):
                t = world.tile(x,y)
                nx, ny = noise_coords(x, y, world, p)
                wx, wy = ApplyWarp(nx, ny, p.warp_stages, perm)
                n = SampleNoise(prof, wx, wy, perm)
                n = ApplyModifiers(n, [
                    {"type": "abs"},
                    {"type": "pow", "k": 1.2},
                    {"type": "remap", "in0": 0.0, "in1": 1.0, "out0": 0.0, "out1": 0.08},
                ], x=wx, y=wy, perm=perm)
                dunes_add = n * t.props.get("desert_mask", 0.0)
                # fade dunes near coast if distance map is available
                dist = world.cache.get("distance_to_ocean")
                if dist is not None:
                    d = dist[x][y]
                    # 0 at coast..3 -> 0, 8+ -> 1
                    interior = clamp((d - 3) / max(1.0, (8 - 3)), 0.0, 1.0)
                    dunes_add *= interior
                else:
                    # approximate interior via continent mask
                    interior = smoothstep(0.15, 0.6, t.props.get("continent_mask", 0.0))
                    dunes_add *= interior
                t.props["dunes_add"] = dunes_add


class BadlandsField(Layer):
    def dependencies(self): return ["DesertMask"]

    def compute(self, world, p, perm):
        prof = p.noise_profiles["badlands"]
        for x in range(world.width):
            for y in range(world.height):
                t = world.tile(x,y)
                nx, ny = noise_coords(x, y, world, p)
                wx, wy = ApplyWarp(nx, ny, p.warp_stages, perm)
                n = SampleNoise(prof, wx, wy, perm)
                n = ApplyModifiers(n, [
                    {"type": "terrace", "steps": 6},
                    {"type": "pow", "k": 1.1},
                    # keep contribution positive to avoid carving oceans
                    {"type": "remap", "in0": -1.0, "in1": 1.0, "out0": 0.0, "out1": 0.12},
                ], x=wx, y=wy, perm=perm)
                # Slightly prefer higher/land areas and dry regions
                mask = t.props.get("desert_mask", 0.0) * 0.7 + smoothstep(0.2, 0.6, t.props.get("height", 0.0)) * 0.3
                t.props["badlands_add"] = n * mask


class RidgesField(Layer):
    def dependencies(self): return ["MountainMask"]

    def compute(self, world, p, perm):
        prof = p.noise_profiles["ridges_detail"]
        for x in range(world.width):
            for y in range(world.height):
                t = world.tile(x,y)
                nx, ny = noise_coords(x, y, world, p)
                wx, wy = ApplyWarp(nx, ny, p.warp_stages, perm)
                n = SampleNoise(prof, wx, wy, perm)
                n = ApplyModifiers(n, [
                    {"type": "abs"},
                    {"type": "pow", "k": 1.15},
                    {"type": "remap", "in0": 0.0, "in1": 1.0, "out0": 0.0, "out1": 0.06},
                ], x=wx, y=wy, perm=perm)
                t.props["ridges_add"] = n * t.props.get("mountain_mask", 0.0)


class Temperature(Layer):
    def dependencies(self): return ["NormalizeHeight", "MoistureLike"]

    def compute(self, world, p, perm):
        for x in range(world.width):
            for y in range(world.height):
                t = world.tile(x, y)
                lat_norm = abs(LatFromY(y, world.height)) / 90.0
                base = (p.climate_params["base_equator_temp"] * (1 - lat_norm) +
                        p.climate_params["pole_temp"] * lat_norm)
                alt = max(0.0, t.props["height"] - p.climate_params["sea_level"])
                temp = base - 6.5 * (alt * 4.0)
                temp += (t.props["moisture"] - 0.5) * 4.0
                # small spatial variation using noise profile (degrees scale)
                nx, ny = noise_coords(x, y, world, p)
                tnoise = SampleNoise(p.noise_profiles.get("temperature_noise", {"noise":"fbm","freq":0.8,"octaves":2}), nx, ny, perm)
                temp += tnoise * 1.5
                t.props["temperature"] = temp




class SimpleRivers(Layer):
    def dependencies(self):
        return ["Temperature"]

    def compute(self, world: World, p: WorldGenParams, perm):
        import math
        w, h = world.width, world.height

        heights = [[world.tile(x,y).props["height"] for y in range(h)] for x in range(w)]
        neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

        # 1) flow dir
        flow_dir = [[None for _ in range(h)] for _ in range(w)]
        for x in range(w):
            for y in range(h):
                here = heights[x][y]
                best = None
                best_h = here
                for dx,dy in neighbors:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < w and 0 <= ny < h:
                        nh = heights[nx][ny]
                        if nh < best_h:
                            best_h = nh
                            best = (nx, ny)
                flow_dir[x][y] = best

        # 2) accumulation
        accum = [[0.0 for _ in range(h)] for _ in range(w)]
        cells = [(heights[x][y], x, y) for x in range(w) for y in range(h)]
        cells.sort(reverse=True)

        # 👇 reduce rainfall a bit
        RAIN_SCALE = 0.25   # was 0.5

        for _, x, y in cells:
            tile = world.tile(x, y)
            rain = 0.0 if "is_ocean" in tile.flags else (tile.props["moisture"] * RAIN_SCALE)
            accum[x][y] += rain
            d = flow_dir[x][y]
            if d is not None:
                accum[d[0]][d[1]] += accum[x][y]

        # 3) mark rivers
        min_flow = p.river_params.get("min_accum_flow", 0.45)
        # 👇 make it harder to be a river
        min_flow *= 1.6   # raise threshold 60%

        carve_strength = p.river_params["carve_strength"]
        sea = p.climate_params["sea_level"]

        for x in range(w):
            for y in range(h):
                tile = world.tile(x, y)
                is_high_enough = tile.props["height"] > (sea + 0.05)  # avoid coast
                if accum[x][y] > min_flow and is_high_enough and "is_ocean" not in tile.flags:
                    tile.flags.add("is_river")
                    tile.props["height"] -= carve_strength * math.log1p(accum[x][y])
                    tile.props["river_to"] = flow_dir[x][y]
                else:
                    tile.props["river_to"] = flow_dir[x][y]

        world.cache["flow_dir"] = flow_dir
        world.cache["flow_accum"] = accum




class CoastalShelf(Layer):
    """
    Smooth the transition between land and ocean in the DATA:
    - shallow up ocean near land
    - lower land near ocean
    """
    def __init__(self, shelf_tiles: int = 5, ocean_shallow_tiles: int = 4):
        self.shelf_tiles = shelf_tiles
        self.ocean_shallow_tiles = ocean_shallow_tiles

    def dependencies(self):
        return ["NormalizeHeight"]

    def compute(self, world: World, p: WorldGenParams, perm):
        w, h = world.width, world.height
        sea = p.climate_params["sea_level"]

        # 1) build queue of all ocean cells
        from collections import deque
        q = deque()
        dist = [[9999 for _ in range(h)] for _ in range(w)]
        for x in range(w):
            for y in range(h):
                if "is_ocean" in world.tile(x,y).flags:
                    dist[x][y] = 0
                    q.append((x,y))

        # 2) multi-source BFS
        dirs = [(-1,0),(1,0),(0,-1),(0,1)]
        while q:
            x,y = q.popleft()
            d = dist[x][y]
            nd = d + 1
            for dx,dy in dirs:
                nx, ny = x+dx, y+dy
                if 0 <= nx < w and 0 <= ny < h:
                    if nd < dist[nx][ny]:
                        dist[nx][ny] = nd
                        q.append((nx,ny))

        # 3) apply shelf to land: tiles close to ocean get pulled down
        for x in range(w):
            for y in range(h):
                t = world.tile(x,y)
                d = dist[x][y]              # tiles to nearest ocean
                h_val = t.props["height"]   # current [-1..1]

                if "is_ocean" not in t.flags:
                    # land side
                    if d <= self.shelf_tiles:
                        # t in [0..1], 0=shore, 1=outer shelf
                        tnorm = d / self.shelf_tiles
                        # how much to blend toward sea level (more near shore)
                        # e.g. 1.0 near shore → mostly sea_level, 0 far away
                        strength = 1.0 - tnorm
                        # target land just a bit above water
                        target = sea + 0.08
                        new_h = h_val * (1 - strength) + target * strength
                        t.props["height"] = new_h
                        # mark beach if really close
                        if d <= 2:
                            t.tags.add("beach")
                else:
                    # ocean side: make near-coast ocean shallower
                    if d <= self.ocean_shallow_tiles:
                        tnorm = d / self.ocean_shallow_tiles
                        strength = 1.0 - tnorm
                        # ocean is currently maybe -0.8 .. -0.2
                        # bring it closer to sea, but stay below
                        target = sea - 0.05
                        new_h = h_val * (1 - strength) + target * strength
                        t.props["height"] = new_h

        # done
        # (optionally store dist if later layers want it)
        world.cache["distance_to_ocean"] = dist


class CombineHeight(Layer):
    """Generic height combiner.
    Combines two existing numeric props into one via mode and optional mask.
    """
    def __init__(self, out_key: str = "height_raw", a_key: str = "height_raw", b_key: str = "height_raw", mask_key: str | None = None, mode: str = "lerp"):
        self.out_key = out_key
        self.a_key = a_key
        self.b_key = b_key
        self.mask_key = mask_key
        self.mode = mode

    def dependencies(self):
        return ["NormalizeHeight"]

    def compute(self, world: World, p: WorldGenParams, perm):
        for x in range(world.width):
            for y in range(world.height):
                t = world.tile(x, y)
                a = t.props.get(self.a_key, 0.0)
                b = t.props.get(self.b_key, 0.0)
                m = t.props.get(self.mask_key, 0.0) if self.mask_key else None
                t.props[self.out_key] = Combine(a, b, self.mode, m)


class BiomeClassify(Layer):
    def dependencies(self): return ["SimpleRivers"]

    def compute(self, world, p, perm):
        # If a biome grid is provided, use table-based biomes; else fallback to thresholds.
        use_grid = bool(p.biome_grid and p.biome_grid.get("table"))
        for x in range(world.width):
            for y in range(world.height):
                t = world.tile(x, y)
                h = t.props["height"]
                temp = t.props["temperature"]
                m = t.props["moisture"]

                if "is_ocean" in t.flags:
                    t.tags.add("ocean")
                    continue

                if "is_river" in t.flags:
                    t.tags.add("river")

                if use_grid:
                    # normalize temp/moist to 0..1 with simple heuristics
                    # temperature mapping assumes approx -20..35 C
                    T01 = clamp((temp + 20.0) / 55.0, 0.0, 1.0)
                    M01 = clamp(m, 0.0, 1.0)
                    biome = BiomeFromGrid(T01, M01, p.biome_grid.get("table", []))
                    # Refine based on masks/features
                    if t.props.get("desert_mask", 0.0) > 0.6:
                        # dunes or sand desert
                        if t.props.get("dunes_add", 0.0) > 0.08:
                            t.tags.add("dune_sea")
                        else:
                            t.tags.add("sand_desert" if temp > 18 else "temperate_desert")
                    elif t.props.get("badlands_add", 0.0) > 0.06:
                        t.tags.add("badlands")
                    elif t.props.get("mountain_mask", 0.0) > 0.75 and temp < 5:
                        t.tags.add("alpine")
                    else:
                        # wetlands near sea level and high moisture
                        sea = p.climate_params["sea_level"]
                        if abs(h - sea) < 0.08 and m > 0.65:
                            t.tags.add("wetlands" if temp < 20 else "swamp")
                        else:
                            t.tags.add(biome)
                else:
                    # threshold fallback
                    if h > 0.9 and temp < 2:
                        t.tags.add("snow")
                        continue
                    if "beach" in t.tags:
                        continue
                    if h > 0.75:
                        t.tags.add("rock")
                        continue
                    if m > 0.65 and temp > 6:
                        t.tags.add("forest")
                    elif m < 0.28 and temp > 8:
                        t.tags.add("desert")
                    else:
                        t.tags.add("grass")


class Erosion(Layer):
    def dependencies(self):
        return ["SimpleRivers"]

    def compute(self, world: World, p: WorldGenParams, perm):
        prof = p.erosion_profile or {}
        if not prof.get("enabled") or int(prof.get("iterations", 0)) <= 0:
            return
        # Build heightfield view
        w, h = world.width, world.height
        heightfield = [[world.tile(x,y).props.get("height", 0.0) for y in range(h)] for x in range(w)]
        iters = int(prof.get("iterations", 10))
        talus = float(prof.get("talus", 0.6))
        rain = float(prof.get("rain", 0.2))
        for _ in range(iters):
            ThermalRelax(heightfield, talus)
            HydraulicStep(heightfield, rain)
        # Write back
        for x in range(w):
            for y in range(h):
                world.tile(x,y).props["height"] = heightfield[x][y]


class Stamps(Layer):
    def dependencies(self):
        return ["BiomeClassify"]

    def compute(self, world: World, p: WorldGenParams, perm):
        feats = p.feature_params or []
        if not feats:
            return
        # Deterministic RNG
        rng = PRNG(p.seed, "stamps")
        w,h = world.width, world.height
        # Heightfield view
        heightfield = [[world.tile(x,y).props.get("height", 0.0) for y in range(h)] for x in range(w)]
        for feat in feats:
            ftype = feat.get("type", "cone")
            count = int(feat.get("count", 0))
            radius = int(feat.get("radius", 10))
            strength = float(feat.get("strength", 0.25))
            on = feat.get("on")  # biome name or None
            min_h = feat.get("min_height")
            for _ in range(count):
                # pick random spots until one matches predicate (limited attempts)
                for _attempt in range(200):
                    rx = int(rng.rand_float() * w)
                    ry = int(rng.rand_float() * h)
                    t = world.tile(rx, ry)
                    if on and on not in t.tags:
                        continue
                    if min_h is not None and t.props.get("height", 0.0) < float(min_h):
                        continue
                    StampShape(heightfield, rx, ry, radius=radius, strength=strength, shape=("crater" if ftype=="crater" else "cone"))
                    break
        # Write back
        for x in range(w):
            for y in range(h):
                world.tile(x,y).props["height"] = heightfield[x][y]


# ============================================================
# 6. worldgen orchestration
# ============================================================

def topological_sort_layers(layers: List[Layer]) -> List[Layer]:
    name_to_layer = {l.name(): l for l in layers}
    visited = set()
    order: List[Layer] = []

    def dfs(layer: Layer):
        if layer.name() in visited:
            return
        for dep in layer.dependencies():
            dfs(name_to_layer[dep])
        visited.add(layer.name())
        order.append(layer)

    for l in layers:
        dfs(l)

    return order

def generate_world(p: WorldGenParams) -> World:
    world = World(p.width, p.height)
    root_rng = PRNG(p.seed, "root")
    perm = make_perm(root_rng)

    layers: List[Layer] = [
        ContinentMask(),
        BaseHeight(),
        NormalizeHeight(),
        # Apply shelf early to shape coastline and provide distance_to_ocean for dunes
        CoastalShelf(shelf_tiles=6, ocean_shallow_tiles=4),
        MoistureLike(),
        Temperature(),
        DesertMask(),
        MountainMask(),
        DunesField(),
        BadlandsField(),
        RidgesField(),
        # Combine new fields into current height
        CombineHeight(out_key="height", a_key="height", b_key="dunes_add", mode="add"),
        CombineHeight(out_key="height", a_key="height", b_key="badlands_add", mode="add"),
        CombineHeight(out_key="height", a_key="height", b_key="ridges_add", mode="add"),
        SimpleRivers(),
        Erosion(),
        BiomeClassify(),
        Stamps(),
    ]

    order = topological_sort_layers(layers)
    for L in order:
        L.compute(world, p, perm)

    return world


# ============================================================
# 7. Isometric renderer with side faces
# ============================================================

class IsoRenderer:
    def __init__(
        self,
        world,
        tile_w=32,
        tile_h=16,
        elev=30,
        margin=80,
        base_drop=40,
        vertical_exaggeration=1,  # <--- NEW
    ):
        self.world = world
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.elev = elev
        self.margin = margin
        self.base_drop = base_drop
        self.vertical_exaggeration = vertical_exaggeration

    def tile_screen_pos_raw(self, x, y, h):
        # exaggerate here
        h_ex = clamp(h * self.vertical_exaggeration, -1.0, 1.0)
        sx = (x - y) * (self.tile_w // 2)
        sy0 = (x + y) * (self.tile_h // 2)
        elev01 = (h_ex + 1) * 0.5
        sy = sy0 - int(elev01 * self.elev)
        return sx, sy

    # raw iso ground y (no elevation, no centering)
    def tile_ground_y_raw(self, x: int, y: int) -> int:
        return (x + y) * (self.tile_h // 2)

    def top_color(self, t: Tile) -> tuple[int, int, int]:
        if "ocean" in t.tags or "is_ocean" in t.flags:
            h = t.props["height"]
            depth = clamp(-h, 0, 1)
            return (15, 50 + int(80 * depth), 100 + int(60 * depth))
        if "beach" in t.tags:
            return (221, 210, 160)
        # Expanded biome colors
        if "polar_ice" in t.tags:
            return (230, 240, 250)
        if "tundra" in t.tags:
            return (170, 180, 170)
        if "boreal_forest" in t.tags:
            return (60, 100, 70)
        if "temperate_forest" in t.tags or "mixed_forest" in t.tags:
            return (65, 130, 70)
        if "temperate_rainforest" in t.tags:
            return (40, 120, 75)
        if "tropical_forest" in t.tags or "tropical_rainforest" in t.tags or "monsoon_forest" in t.tags or "seasonal_forest" in t.tags:
            return (30, 110, 60)
        if "savanna" in t.tags:
            return (160, 170, 80)
        if "grassland" in t.tags or "steppe" in t.tags or "shrubland" in t.tags:
            return (140, 175, 100)
        if "wetlands" in t.tags or "swamp" in t.tags:
            return (50, 110, 90)
        if "badlands" in t.tags:
            return (190, 120, 80)
        if "dune_sea" in t.tags:
            return (235, 220, 170)
        if "sand_desert" in t.tags or "temperate_desert" in t.tags or "subtropical_desert" in t.tags or "cold_desert" in t.tags:
            return (220, 200, 120)
        if "snow" in t.tags:
            return (235, 235, 240)
        if "rock" in t.tags:
            return (130, 110, 95)
        if "forest" in t.tags:
            return (50, 120, 55)
        if "alpine" in t.tags:
            return (200, 200, 210)
        if "desert" in t.tags:
            return (220, 200, 120)
        return (120, 165, 95)

    def side_color(self, col: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
        return (int(col[0] * factor), int(col[1] * factor), int(col[2] * factor))

    def render(self) -> Image.Image:
        # 1) gather everything first
        positions = []
        for x in range(self.world.width):
            for y in range(self.world.height):
                t = self.world.tile(x, y)
                h = t.props["height"]
                sx, sy = self.tile_screen_pos_raw(x, y, h)       # with elevation
                gy = self.tile_ground_y_raw(x, y)                # without elevation
                positions.append((x, y, sx, sy, gy, h, t))

        # 2) bounds based on elevated tops
        min_x = min(p[2] for p in positions)
        max_x = max(p[2] for p in positions)
        min_y = min(p[3] for p in positions)
        max_y = max(p[3] for p in positions)

        # also need bounds for ground plane (gy) to center nicely
        min_gy = min(p[4] for p in positions)
        max_gy = max(p[4] for p in positions)

        # 3) image size
        width = (max_x - min_x) + self.margin * 2 + self.tile_w
        # height must accommodate: top (with elevation) AND ground plane (plus drop)
        # ground plane after centering will be aligned too
        height = (max_gy - min_y) + self.margin * 2 + self.tile_h + self.base_drop

        img = Image.new("RGB", (width, height), (35, 36, 55))
        draw = ImageDraw.Draw(img)

        # 4) offsets to center (we center by elevated tops' min_x/min_y)
        off_x = -min_x + self.margin
        off_y = -min_y + self.margin

        # 5) draw in iso order
        cells = sorted(positions, key=lambda p: (p[0] + p[1], p[0], p[1]))

        for (x, y, sx, sy, gy, h, t) in cells:
            sx += off_x
            sy += off_y
            # ground (iso) y for this tile, centered, then drop a bit
            ground_y = (gy - min_y) + self.margin + self.base_drop

            w2 = self.tile_w // 2
            h2 = self.tile_h // 2

            top_col = self.top_color(t)
            left_col = self.side_color(top_col, 0.65)
            right_col = self.side_color(top_col, 0.8)

            # top diamond
            top_poly = [
                (sx, sy),
                (sx + w2, sy + h2),
                (sx, sy + self.tile_h),
                (sx - w2, sy + h2),
            ]

            # bottom of the top diamond:
            top_bottom_y = sy + self.tile_h

            # --- NEW: sides go down to this tile's own ground plane ---
            side_h = ground_y - top_bottom_y
            if side_h < 0:
                side_h = 0

            # right face
            if side_h > 0:
                right_poly = [
                    (sx + w2, sy + h2),
                    (sx + w2, sy + h2 + side_h),
                    (sx,       sy + self.tile_h + side_h),
                    (sx,       sy + self.tile_h),
                ]
                draw.polygon(right_poly, fill=right_col)

                # left face
                left_poly = [
                    (sx - w2, sy + h2),
                    (sx - w2, sy + h2 + side_h),
                    (sx,       sy + self.tile_h + side_h),
                    (sx,       sy + self.tile_h),
                ]
                draw.polygon(left_poly, fill=left_col)

            # draw top last
            draw.polygon(top_poly, fill=top_col, outline=(0, 0, 0))

            # small river accent
            if "is_river" in t.flags:
                river_col = (20, 150, 200)

                # center of THIS tile (middle of diamond)
                curr_center = (sx, sy + h2)

                # try stored direction from data
                to_coord = t.props.get("river_to")

                # fallback to lowest neighbor if missing
                if not to_coord:
                    wmax, hmax = self.world.width, self.world.height
                    best = None
                    best_h = t.props["height"]
                    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < wmax and 0 <= ny < hmax:
                            nh = self.world.tile(nx, ny).props["height"]
                            if nh < best_h:
                                best_h = nh
                                best = (nx, ny)
                    to_coord = best

                if to_coord:
                    nx, ny = to_coord
                    down_tile = self.world.tile(nx, ny)
                    # ⚠ raw iso pos:
                    sx2, sy2 = self.tile_screen_pos_raw(nx, ny, down_tile.props["height"])
                    # ✅ apply SAME offsets as current tile:
                    sx2 += off_x
                    sy2 += off_y
                    dst_center = (sx2, sy2 + (self.tile_h // 2))
                else:
                    # no direction → tiny downward dab
                    dst_center = (sx, sy + h2 + 1)

                # build a narrow quad between centers
                x1, y1 = curr_center
                x2, y2 = dst_center

                vx, vy = (x2 - x1), (y2 - y1)
                L = math.hypot(vx, vy)
                if L < 1e-3:
                    L = 1.0
                vx /= L
                vy /= L

                # perpendicular
                px, py = -vy, vx

                # middle third of tile
                river_half = self.tile_w / 6.0      # full width = tile_w/3
                # also clamp how far we draw so it doesn't shoot past neighbor
                max_len = self.tile_w * 0.85
                if L > max_len:
                    scale = max_len / L
                    x2 = x1 + vx * max_len
                    y2 = y1 + vy * max_len

                p1 = (x1 + px*river_half, y1 + py*river_half)
                p2 = (x1 - px*river_half, y1 - py*river_half)
                p3 = (x2 - px*river_half, y2 - py*river_half)
                p4 = (x2 + px*river_half, y2 + py*river_half)

                draw.polygon([p1, p2, p3, p4], fill=river_col)

        return img




# ============================================================
# 8. main
# ============================================================

def main():
    # Less wispy defaults: neutral domain scale, island mode, lower sea level, milder global warp
    p = WorldGenParams(
        width=333,
        height=333,
        seed=3245,
        force_island=False,
        noise_domain_scale={"x": 0.5, "y": 0.5},
    )
    # Adjust climate/sea level for more land area
    p.climate_params["sea_level"] = -0.22
    world = generate_world(p)
    renderer = IsoRenderer(world, tile_w=34, tile_h=17, elev=150, margin=120, base_drop=20)
    img = renderer.render()
    img.save("world/world_iso.png")
    print("Saved world_iso.png")

if __name__ == "__main__":
    main()

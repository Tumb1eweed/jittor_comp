import math
import random

import numpy as np
import jittor as jt


DEFAULT_NOISE_TYPES = "gaussian,laplace,uniform,nonuniform"


def parse_noise_types(noise_types):
    if isinstance(noise_types, (tuple, list)):
        values = list(noise_types)
    else:
        values = str(noise_types).split(",")
    values = [v.strip().lower() for v in values if v.strip()]
    if not values:
        raise ValueError("noise_types must not be empty")
    supported = {"gaussian", "laplace", "uniform", "nonuniform"}
    unknown = sorted(set(values) - supported)
    if unknown:
        raise ValueError("unsupported noise_types: {}".format(",".join(unknown)))
    return values


def sample_noise_std(noise_std, noise_std_min=None, noise_std_max=None, rng=None):
    if noise_std_min is None or noise_std_max is None:
        return float(noise_std)
    if rng is None:
        return float(np.random.uniform(noise_std_min, noise_std_max))
    return float(rng.uniform(noise_std_min, noise_std_max))


def add_numpy_noise(clean, noise_std, noise_types=DEFAULT_NOISE_TYPES, rng=None):
    rng = rng or np.random
    noise_type = rng.choice(parse_noise_types(noise_types))
    shape = clean.shape
    if noise_type == "gaussian":
        noise = rng.normal(0.0, noise_std, size=shape)
    elif noise_type == "laplace":
        noise = rng.laplace(0.0, noise_std / math.sqrt(2.0), size=shape)
    elif noise_type == "uniform":
        noise = rng.uniform(-math.sqrt(3.0) * noise_std, math.sqrt(3.0) * noise_std, size=shape)
    elif noise_type == "nonuniform":
        base = rng.normal(0.0, 1.0, size=shape)
        point_scale = rng.uniform(0.25, 1.75, size=(shape[0], 1))
        point_scale = point_scale / math.sqrt(float(np.mean(point_scale * point_scale)))
        noise = base * point_scale * noise_std
    else:
        raise ValueError("unsupported noise_type: {}".format(noise_type))
    return (clean + noise.astype(np.float32)).astype(np.float32), noise_type


def add_jittor_noise(clean, noise_std, noise_types=DEFAULT_NOISE_TYPES):
    noise_type = random.choice(parse_noise_types(noise_types))
    shape = clean.shape
    if noise_type == "gaussian":
        noise = jt.randn(shape) * noise_std
    elif noise_type == "laplace":
        # Inverse-CDF Laplace sampling. The scale gives variance == noise_std^2.
        u = jt.random(shape) - 0.5
        sign = (u > 0).float32() * 2.0 - 1.0
        noise = -sign * jt.log(1.0 - 2.0 * jt.abs(u) + 1e-6) * (noise_std / math.sqrt(2.0))
    elif noise_type == "uniform":
        noise = (jt.random(shape) * 2.0 - 1.0) * (math.sqrt(3.0) * noise_std)
    elif noise_type == "nonuniform":
        base = jt.randn(shape)
        point_scale = jt.random((shape[0], 1)) * 1.5 + 0.25
        point_scale = point_scale / jt.sqrt(jt.mean(point_scale * point_scale))
        noise = base * point_scale * noise_std
    else:
        raise ValueError("unsupported noise_type: {}".format(noise_type))
    return clean + noise, noise_type

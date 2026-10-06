"""Portable scientific values: null is unavailable; infinity is an exact tag.

JSON does not have NaN or infinity numbers. Unavailable observations serialize
as null; an extended positive/negative infinite observation serializes as the
string "Infinity"/"-Infinity". observed_float reverses these encodings without
substituting zero. A missing structural crossing already uses null, so readers
must retain the field's declared semantics instead of inferring absence of drive.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np


def observed_float(value: Any) -> float:
    if value is None or (isinstance(value, str) and value == ""):
        return float("nan")
    return float(value)


def scientific_values(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return None
        if math.isinf(number):
            return "Infinity" if number > 0 else "-Infinity"
        return number
    if isinstance(value, dict):
        return {key: scientific_values(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scientific_values(v) for v in value]
    if isinstance(value, np.ndarray):
        return scientific_values(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def scientific_dumps(value: Any, **kwargs: Any) -> str:
    kwargs['allow_nan'] = False
    return json.dumps(scientific_values(value), **kwargs)

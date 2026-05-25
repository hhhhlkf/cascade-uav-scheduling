from __future__ import annotations

import random
from typing import Optional

import numpy as np


def seed_everything(seed: Optional[int]) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


"""Global determinism for reproducible campaigns.

Call ``set_global_determinism(seed)`` at the top of every experiment entry point so
that a fixed seed yields bit-stable numbers (the audit found no global determinism
and a non-seeded commander decode). Basilisk's own RNG is seeded via the env config.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_determinism(seed: int) -> None:
    """Seed Python, NumPy, and (if present) PyTorch, and request deterministic kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:  # noqa: BLE001 - torch optional / flags best-effort
        pass

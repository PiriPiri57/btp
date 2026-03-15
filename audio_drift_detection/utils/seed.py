"""Global seed setter for reproducibility."""

import random
import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """Set seed for all random number generators.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

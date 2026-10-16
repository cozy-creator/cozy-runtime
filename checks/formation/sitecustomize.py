"""Kernel formation's stand-in for the card a build serves (`checks/kernel_formation.py`, whose
builders alone carry this directory on their path). Upstream's `setup.py` asks torch for the
visible GPU's capability to choose its targets; a formation host has no GPU, so torch answers
with the card `kernel_sources.build_setup` names in `COZY_KERNEL_SM`."""

import os

if os.environ.get("COZY_KERNEL_SM"):
    import torch

    _CARD = divmod(int(os.environ["COZY_KERNEL_SM"]), 10)
    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: 1
    torch.cuda.get_device_capability = lambda device=None: _CARD

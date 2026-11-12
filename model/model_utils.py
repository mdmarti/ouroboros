import warnings
from typing import Union

import torch


def resolve_device(device: Union[str, torch.device] = "cuda") -> torch.device:
    """
    turn a requested device into one that actually exists. If a cuda device is
    requested but torch.cuda.is_available() is False, warn and fall back to the cpu
    (so the default of 'cuda' still works on machines without a gpu, just slowly).

    inputs
    -----
        - device: a device string or torch.device, e.g. 'cuda', 'cuda:1', 'cpu'

    returns
    -----
        - a usable torch.device
    """
    device = torch.device(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            warnings.warn(
                "cuda was requested but torch.cuda.is_available() is False; "
                "falling back to cpu. Pass device='cpu' to silence this warning.",
                RuntimeWarning,
                stacklevel=2,
            )
            return torch.device("cpu")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"requested {device}, but only {torch.cuda.device_count()} cuda device(s) found"
            )
    return device


def smooth(data: torch.FloatTensor, smooth_len: float) -> torch.FloatTensor:
    """
    Smooth, using a causal uniform moving average. Uses cumsum
    to speed performance. By default, pre-pads with zeros

    inputs
    -----
        - data: data to be smoothed
        - smooth_len: smooth length, in samples, of smoothing window

    returns
    -----
        - smoothed data
    """

    B, L, D = data.shape
    if smooth_len == 1:
        return data
    pad = torch.zeros((B, smooth_len, D), device=data.device)
    try:
        cumsum = torch.cumsum(torch.cat([pad, data], dim=1), dim=1)
    except:
        print(pad.shape, data.shape)
        raise
    return (cumsum[:, smooth_len:, :] - cumsum[:, :-smooth_len, :]) / float(smooth_len)

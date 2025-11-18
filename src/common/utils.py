import torch

def ego_to_float01(x: torch.Tensor) -> torch.Tensor:
    """
    Convert ego map from uint8 [0,1]to float 32
    """
    if x.dtype == torch.uint8:
        return x.float().div_(255.0)
    if x.is_floating_point():
        x = x.float()
        ## return nan as 0.0 (return tensor as it is)
        xmin = torch.nan_to_num(x.min(), nan=0.0)
        xmax = torch.nan_to_num(x.max(), nan=0.0)
        if float(xmin) < -1e-6 or float(xmax) > 1.0 + 1e-6:
            raise ValueError(f"ego_map float out of [0,1]: min={float(xmin):.6f}, max={float(xmax):.6f}")
        return x
    raise TypeError(f"ego_map must be uint8 or float tensor, got {x.dtype}")

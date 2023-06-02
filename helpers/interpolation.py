import torch
import torch.nn.functional as F

def interpolate(t, prompt1_c, prompt2_c, mode="slerp"):
    """
    Perform interpolation (SLERP or LERP) between two tensors.

    :param t: Interpolation factor (scalar between 0 and 1).
    :param prompt1: Starting tensor.
    :param prompt2: Ending tensor.
    :param mode: Interpolation mode ("slerp" or "lerp").
    :return: Interpolated tensor.
    """
    if mode == "slerp":
        # Normalize the tensors
        prompt1_norm = F.normalize(prompt1_c)
        prompt2_norm = F.normalize(prompt2_c)

        # Compute cos_omega
        cos_omega = torch.sum(prompt1_norm * prompt2_norm)

        # Clamp the values to handle numerical issues
        cos_omega = cos_omega.clamp(-1, 1)

        # Compute the angle between the tensors
        omega = torch.acos(cos_omega)

        sin_omega = torch.sqrt(1.0 - cos_omega ** 2)

        # If sin_omega is small, use linear interpolation
        if sin_omega < 1e-5:
            return (1.0 - t) * prompt1_c + t * prompt2_c
        else:
            return (torch.sin((1.0 - t) * omega) / sin_omega) * prompt1_c + (torch.sin(t * omega) / sin_omega) * prompt2_c
    elif mode == "lerp":
        return (1.0 - t) * prompt1_c + t * prompt2_c
    else:
        raise ValueError(f"Unknown interpolation mode: {mode}")

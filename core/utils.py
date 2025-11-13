"""Image utility functions for tensor/PIL conversions."""

import base64
from io import BytesIO
import torch
import numpy as np
from PIL import Image


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a tensor to PIL Image.

    Args:
        tensor (torch.Tensor): Input tensor of shape (1, 3, H, W) with values in [0, 1]. (Comfy format)

    Returns:
        Image: PIL Image object.
    """
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, "RGB")
    return image


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL Image to tensor.

    Args:
        image (Image): Input PIL Image object.

    Returns:
        torch.Tensor: Tensor of shape (1, 3, H, W) with values in [0, 1]. (Comfy format)
    """
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def pil_to_base64(image: Image.Image) -> str:
    """
    Convert a PIL Image to base64 string.

    Args:
        image (Image): Input PIL Image object.

    Returns:
        str: Base64 string of the image.
    """

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    prefix = "data:image/png;base64,"
    return prefix + str(base64.b64encode(buffered.getvalue()).decode("utf-8"))

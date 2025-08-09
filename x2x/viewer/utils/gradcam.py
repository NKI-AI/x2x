import logging
from typing import Any, Dict, Tuple

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from x2x.utils.logger import get_logger

log = get_logger(__name__, level=logging.getLogger().getEffectiveLevel())


class OnlyTensorModelOutput(torch.nn.Module):
    """
    Class to get actual output from model output which is in dictionary format.
    Required for the gradcam forward.

    This will return the output of the model that still contains output of multiple classes.
    The gradcam class selects the label of interest using a targets=[ClassifierOutputTarget(7)] argument, which
    simply selects the 7th index of the output.
    """

    def __init__(self, model: torch.nn.Module, invert=False):
        super().__init__()
        self.model = model
        self.invert = invert

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        model_out = self.model(x)
        model_out_tensor = self.get_value(model_out)
        # Reshape from (batch, num_labels, 2) to (batch, num_labels * 2)
        # This flattens the logits for each label into a single dimension
        model_out_tensor = model_out_tensor.reshape(
            model_out_tensor.shape[0],
            model_out_tensor.shape[1] * model_out_tensor.shape[2],
        )
        return model_out_tensor

    def get_value(self, model_out: Dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError


class ClassScoreModelOutput(OnlyTensorModelOutput):
    """Return a class logit"""

    def get_value(self, model_out: Dict[str, Any]) -> torch.Tensor:
        return (
            model_out["out"]["logits"]
            if not self.invert
            else -1 * model_out["out"]["logits"]
        )


class AttentionScoreModelOutput(OnlyTensorModelOutput):
    """Return an attention logit"""

    def get_value(self, model_out: Dict[str, Any]) -> torch.Tensor:
        return (
            model_out["meta"]["attention_logits"]
            if not self.invert
            else -1 * model_out["meta"]["attention_logits"]
        )


class AttentionTimesScoreModelOutput(OnlyTensorModelOutput):
    """Return an attention weight times a score"""

    def get_value(self, model_out: Dict[str, Any]) -> torch.Tensor:
        return model_out["meta"]["attention_logits"] * model_out["out"]["logits"]


class Unsqueeze(torch.nn.Module):
    """unsqueeze output of encoder.
    Encoder wants a batch of tiles, but MIL wants a batch of slides.
    """

    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(self.dim)


def process_single_image(
    rgb_img: Image.Image,
    model: torch.nn.Module,
    model_wrapper: type[OnlyTensorModelOutput],
    target_layers: list[torch.nn.Module],
    transform: transforms.Compose,
    target_idx: int,
    device: str = "cpu",
    reshape_transform=None,
) -> Tuple[np.ndarray, str | None]:
    """
    Process a single image through the model and return its GradCAM visualization

    Parameters
    ----------
    rgb_img : Image.Image
        The input RGB image to process
    model : torch.nn.Module
        The model to use for GradCAM
    model_wrapper : type[OnlyTensorModelOutput]
        The wrapper class to use for the model output
    target_layers : list[torch.nn.Module]
        Layers to use for GradCAM
    transform : transforms.Compose
        Image transformations to apply
    target_idx : int
        Index of the target class for GradCAM
    device : str
        Device to run the model on
    reshape_transform : callable, optional
        Function to reshape the activation maps for GradCAM, specific to model architecture

    Returns
    -------
    Tuple[np.ndarray, str | None]
        The processed image and optional title
    """
    input_tensor = transform(rgb_img).unsqueeze(0).to(device)

    with GradCAM(
        model=model_wrapper(model),
        target_layers=target_layers(model),
        reshape_transform=reshape_transform,
    ) as cam:

        grayscale_cam = cam(
            input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_idx)]
        )[0, :]
        visualization = show_cam_on_image(
            np.array(rgb_img).astype(np.float32) / 255.0, grayscale_cam, use_rgb=True
        )

    return visualization, None

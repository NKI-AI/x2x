import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from PIL import Image

from x2x.utils.logger import get_logger
from x2x.viewer.models.mil.kather_crc_prognostics.mil import CRCPrognosticModel
from x2x.viewer.utils.gradcam import (
    AttentionScoreModelOutput,
    AttentionTimesScoreModelOutput,
    ClassScoreModelOutput,
    process_single_image,
)

logger = get_logger(__name__, level=logging.getLogger().getEffectiveLevel())


class PatchProcessor:
    def __init__(self, cfg: Dict[str, Any]):
        """
        Initialize the patch processor with model and configuration.

        Parameters
        ----------
        cfg : Dict[str, Any]
            Configuration dictionary containing model settings
        """

        log_level = cfg.get("LOG_LEVEL", "INFO")
        level = getattr(logging, log_level)
        logging.getLogger().setLevel(level)
        self.logger = get_logger(__name__, level=level)
        self.logger.debug(f"Set log level to: {log_level}")

        # Initialize model
        self.device = cfg["DEVICE"]
        self.model = CRCPrognosticModel(
            mil_weights_path=Path(cfg["REPO_ROOT"]) / cfg["CRC_MIL_MODEL_CKPT_PATH"],
            retccl_weights_path=Path(cfg["REPO_ROOT"]) / cfg["RETCCL_MODEL_CKPT_PATH"],
            macenko_normalization_template_path=Path(cfg["REPO_ROOT"])
            / cfg["MACENKO_NORMALIZATION_TEMPLATE_PATH"],
            device=self.device,
        )
        self.model.to(self.device)
        self.model.eval()

        # This is design choice that may be different for different architectures.
        self.target_layers = lambda model: [model.feature_extractor.net.layer4[-1]]

        self.num_labels = 1
        self.num_classes = 1
        self.reshape_transform = (
            None  # Optional argument for gradcam. May be set for other models
        )

        self.log_dir = Path(cfg["LOG_DIR"]) / cfg["EXPERIMENT_NAME"]
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Log dir set to {self.log_dir}")

        self.cfg = cfg

    def process_patch(
        self,
        patch: Image.Image,
        slide_name: str,
        location: Tuple[int, int],
        mpp: float,
        save_format: str = "jpeg",
    ) -> Dict[str, Any]:
        """
        Process a single patch through the model and generate visualizations

        Parameters
        ----------
        patch : Image.Image
            The patch to process (224x224 RGB image)
        slide_name : str
            Name of the slide the patch is from
        location : Tuple[int, int]
            (x, y) coordinates of the patch in the slide
        mpp : float
            Microns per pixel of the patch
        save_format : str, optional
            Format to save images in ('jpeg' or 'png'), by default 'jpeg'
        """

        # Convert to RGB and get model outputs
        patch = patch.convert("RGB")

        # Update directory path construction to include experiment name
        patch_dir = self.log_dir / slide_name / f"{location[0]}-{location[1]}-{mpp:.1f}"
        patch_dir.mkdir(parents=True, exist_ok=True)

        # Update file extensions based on format
        ext = f".{save_format}"

        # Save all visualizations in the patch directory with proper quality for JPEGs
        patch.save(patch_dir / f"rgb{ext}", quality=95)

        images = {}

        input_tensor = self.model.transform(patch).unsqueeze(0).to(self.device)

        # Save Macenko output if available
        if self.model.last_macenko_output is not None:
            self.logger.info(f"Saving Macenko output to {patch_dir / f'macenko{ext}'}")
            self.model.last_macenko_output.save(patch_dir / f"macenko{ext}", quality=95)
            images["macenko"] = f"macenko{ext}"

        output = self.model(input_tensor)

        scores = output["out"]["logits"].squeeze(0)  # Remove batch dimension
        attention = output["meta"]["attention_logits"].squeeze(
            0
        )  # Remove batch dimension

        positive_idx = 0  # prognostics model gives a single logit output

        images = {}  # Contains filepaths passed to frontend for displaying images

        for model_wrapper, idx, name, filename in (
            (
                AttentionTimesScoreModelOutput,
                positive_idx,
                "attention_times_score",
                "gradcam_attention_times_score",
            ),
            (
                AttentionScoreModelOutput,
                positive_idx,
                "attention_positive",
                "gradcam_attention_positive",
            ),
            (
                lambda model: AttentionScoreModelOutput(model, invert=True),
                positive_idx,
                "attention_negative",
                "gradcam_attention_negative",
            ),
            (
                ClassScoreModelOutput,
                positive_idx,
                "score_positive",
                "gradcam_score_positive",
            ),
            (
                lambda model: ClassScoreModelOutput(model, invert=True),
                positive_idx,
                "score_negative",
                "gradcam_score_negative",
            ),
        ):

            out = process_single_image(
                rgb_img=patch,
                model=self.model,
                model_wrapper=model_wrapper,
                target_layers=self.target_layers,
                transform=self.model.transform,
                target_idx=idx,  # Always only 1 class
                device=self.device,
                reshape_transform=self.reshape_transform,
            )[0]

            self.logger.info(f"Saving {filename}{ext} to {patch_dir}")

            Image.fromarray(out).save(patch_dir / f"{filename}{ext}", quality=95)

            images[name] = f"{filename}{ext}"

        # Save metadata as JSON
        # Compute probabilities once
        probs = (
            torch.softmax(scores, dim=-1)
            if self.num_classes > 1
            else torch.sigmoid(scores)
        )

        metadata = {
            "raw_logits": {
                f"label_{label_idx}": (
                    scores[label_idx].tolist()
                    if self.num_classes > 1
                    else float(scores[label_idx])
                )
                for label_idx in range(self.num_labels)
            },
            "probabilities": {
                f"label_{label_idx}": (
                    probs[label_idx].tolist()
                    if self.num_classes > 1
                    else float(probs[label_idx])
                )
                for label_idx in range(self.num_labels)
            },
            "attention_logits": {
                f"label_{label_idx}": attention[label_idx].tolist()
                for label_idx in range(self.num_labels)
            },
            "notes": "Attention logits are more meaningful when viewing multiple patches on a slide",
        }

        self.logger.info(f"Saving metadata to {patch_dir / 'metadata.json'}")

        with open(patch_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Convert numpy arrays to lists for JSON serialization to pass it back to the frontend
        return {
            "scores": scores.tolist(),  # num_labels x num_classes
            "probabilities": probs.tolist(),  # num_labels x num_classes
            "attention": attention.tolist(),  # num_labels x 1
            "save_path": f"{self.log_dir}/{slide_name}/{location[0]}-{location[1]}-{mpp:.1f}",
            "absolute_save_path": str(patch_dir.resolve()),
            "images": images,
        }

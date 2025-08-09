# Based on model architecture from https://github.com/KatherLab/marugoto/blob/main/marugoto/mil/model.py

from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from x2x.viewer.models.encoders.retccl.retccl import RetCCL
from x2x.viewer.models.mil.kather_crc_prognostics.macenko import Normalizer

__all__ = ["MILModel", "Attention"]


class MILModel(nn.Module):
    def __init__(
        self,
        n_feats: int,
        n_out: int,
        encoder: Optional[nn.Module] = None,
        attention: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
    ) -> None:
        """Create a new attention MIL model.

        Parameters
        ----------
            n_feats: int
                The number of features each bag instance has.
            n_out: int
                The number of output layers of the model.
            encoder: nn.Module
                A network transforming bag instances into feature vectors.
            attention: nn.Module
                A network calculating an embedding's importance weight.
            head: nn.Module
                A network transforming the weighted embedding sum into a score.

        Returns
        -------
        None
        """
        super().__init__()
        self.encoder = encoder or nn.Sequential(nn.Linear(n_feats, 256), nn.ReLU())
        self.attention = attention or Attention(256, n_latent=128)
        self.head = head or nn.Sequential(
            nn.Flatten(), nn.BatchNorm1d(256), nn.Dropout(), nn.Linear(256, n_out)
        )

    def forward(self, bags, lens):
        assert bags.ndim == 3
        assert bags.shape[0] == lens.shape[0]

        embeddings = self.encoder(bags)

        masked_attention_scores, masked_attention_logit = self._masked_attention_scores(
            embeddings, lens
        )
        weighted_embedding_sums = (masked_attention_scores * embeddings).sum(-2)

        scores = self.head(weighted_embedding_sums)

        return scores, masked_attention_logit

    def _masked_attention_scores(
        self, embeddings: torch.Tensor, lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates attention scores for all bags.

        Returns
        -------
        torch.Tensor: A tensor containing the attention score of instance i of bag j if i < len[j] and 0 otherwise
        """
        bs, bag_size = embeddings.shape[0], embeddings.shape[1]
        attention_scores = self.attention(embeddings)

        # a tensor containing a row [0, ..., bag_size-1] for each batch instance
        idx = torch.arange(bag_size).repeat(bs, 1).to(attention_scores.device)

        # False for every instance of bag i with index(instance) >= lens[i]
        attention_mask = (idx < lens.unsqueeze(-1)).unsqueeze(-1)

        masked_attention = torch.where(
            attention_mask, attention_scores, torch.full_like(attention_scores, -1e10)
        )
        return torch.softmax(masked_attention, dim=1), masked_attention


def Attention(n_in: int, n_latent: Optional[int] = None) -> nn.Module:
    """A network calculating an embedding's importance weight.

    Parameters
    ----------
        n_in: int
            The number of input features.
        n_latent: int
            The number of latent features.

    Returns
    -------
    nn.Module: A network calculating an embedding's importance weight.
    """
    n_latent = n_latent or (n_in + 1) // 2

    return nn.Sequential(nn.Linear(n_in, n_latent), nn.Tanh(), nn.Linear(n_latent, 1))


def load_model(model_path: str, device: str = "cpu", eval: bool = True) -> MILModel:
    """Load a model from a path.

    Parameters
    ----------
        model_path: str
            Path to the model weights.
        device: str
            Device to load the model on.
        eval: bool
            Whether to set the model to evaluation mode.

    Returns
    -------
    MILModel: The loaded model.
    """
    model = MILModel(n_feats=2048, n_out=1)
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=torch.device(device),
        ),
    )
    if eval:
        model.eval()
    return model


class CRCPrognosticModel(nn.Module):
    """End-to-end model for CRC prognostic prediction from a single image.

    Parameters
    ----------
        mil_weights_path: str
            Path to the MIL model weights.
        retccl_weights_path: str
            Path to the RetCCL model weights.
        macenko_normalization_template_path: str
            Path to the Macenko normalization template.
        device: str
            Device to load the model on.

    Returns
    -------
    None
    """

    def __init__(
        self,
        mil_weights_path: str,
        retccl_weights_path: str,
        macenko_normalization_template_path: str,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        # Initialize RetCCL
        self.feature_extractor = RetCCL(
            absolute_weights_path=retccl_weights_path, device=device
        )

        # Initialize MIL model
        self.mil = MILModel(n_feats=2048, n_out=1)

        # Load weights
        state_dict = torch.load(mil_weights_path, map_location=torch.device(device))
        self.mil.load_state_dict(state_dict)

        # Initialize Macenko normalizer
        self.macenko_normalizer = Normalizer()
        self.macenko_normalizer.fit(
            np.array(Image.open(macenko_normalization_template_path).convert("RGB"))
        )

        self.last_macenko_output = None

        # Define image transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.Lambda(self._macenko_transform),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.to(device)
        self.eval()

    def _macenko_transform(self, img: Image.Image) -> Image.Image:
        img_array = np.array(img)
        normalized = self.macenko_normalizer.transform(img_array)
        self.last_macenko_output = Image.fromarray(normalized)
        return Image.fromarray(normalized)

    def forward(self, image: torch.Tensor) -> dict:
        """
        Parameters
        ----------
            image: torch.Tensor
                A single RGB image tensor of shape (C, H, W)

        Returns
        -------
        dict: Dictionary containing prediction score and attention weights
        """
        # Add batch and instance dimensions
        if image.ndim == 3:
            image = image.unsqueeze(0)  # Add batch dimension

        features = self.feature_extractor(image)  # Shape: (1, 2048)
        features = features.unsqueeze(1)  # Shape: (1, 1, 2048) for MIL
        lens = torch.ones(1, device=features.device)
        scores, attention_logits = self.mil(features, lens)

        # Add the missing dimension to match grade model structure
        scores = scores.unsqueeze(1)  # Shape: (batch, num_labels=1, num_classes=1)

        return {
            "out": {"logits": scores},
            "meta": {"attention_logits": attention_logits},
        }

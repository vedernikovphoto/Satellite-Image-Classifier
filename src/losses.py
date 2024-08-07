from dataclasses import dataclass
from torch import nn
from typing import List

from src.config import LossConfig
from src.train_utils import load_object


@dataclass
class Loss:
    """
    A dataclass representing a loss function configuration.

    Attributes:
        name (str): The name of the loss function.
        weight (float): The weight of the loss function.
        loss (nn.Module): The loss function module.
    """
    name: str
    weight: float
    loss: nn.Module


def get_losses(losses_cfg: List[LossConfig]) -> List[Loss]:
    """
    Retrieves and initializes loss functions based on the given configuration.

    Args:
        losses_cfg (List[LossConfig]): List of loss configurations.

    Returns:
        List[Loss]: List of initialized loss functions.
    """
    return [
        Loss(
            name=loss_cfg.name,
            weight=loss_cfg.weight,
            loss=load_object(loss_cfg.loss_fn)(**loss_cfg.loss_kwargs),
        )
        for loss_cfg in losses_cfg
    ]

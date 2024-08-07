from typing import List, Dict

from omegaconf import OmegaConf
from pydantic import BaseModel


class LossConfig(BaseModel):
    """
    Configuration for a loss function.

    Attributes:
        name (str): The name of the loss function.
        weight (float): The weight of the loss function in the total loss.
        loss_fn (str): The name or path of the loss function.
        loss_kwargs (dict): Additional keyword arguments for the loss function.
    """
    name: str
    weight: float
    loss_fn: str
    loss_kwargs: dict


class DataConfig(BaseModel):
    """
    Configuration for the dataset.

    Attributes:
        data_path (str): Path to the data directory.
        batch_size (int): Number of samples per batch.
        n_workers (int): Number of worker threads for data loading.
        train_size (float): Proportion of the data to be used for training.
        width (int): Width to resize the images to.
        height (int): Height to resize the images to.
    """
    data_path: str
    batch_size: int
    n_workers: int
    train_size: float
    width: int
    height: int


class AugmentationConfig(BaseModel):
    """
    Configuration for data augmentation.

    Attributes:
        hue_shift_limit (int): Limit for hue shift.
        sat_shift_limit (int): Limit for saturation shift.
        val_shift_limit (int): Limit for value shift.
        brightness_limit (float): Limit for brightness adjustment.
        contrast_limit (float): Limit for contrast adjustment.
    """
    hue_shift_limit: int
    sat_shift_limit: int
    val_shift_limit: int
    brightness_limit: float
    contrast_limit: float


class Config(BaseModel):
    """
    Main configuration class for the project.

    Attributes:
        project_name (str): The name of the project.
        experiment_name (str): The name of the experiment.
        data_config (DataConfig): Configuration for the dataset.
        augmentation_params (AugmentationConfig): Configuration for data augmentation.
        n_epochs (int): Number of training epochs.
        num_classes (int): Number of classes in the dataset.
        accelerator (str): Type of accelerator to use (e.g., 'gpu').
        device (int): Device identifier.
        seed (int): Random seed for reproducibility.
        log_every_n_steps (int): Logging frequency in steps.
        monitor_metric (str): Metric to monitor for early stopping.
        monitor_mode (str): Mode for monitoring ('min' or 'max').
        model_kwargs (dict): Additional keyword arguments for the model.
        optimizer (str): Name of the optimizer.
        optimizer_kwargs (dict): Additional keyword arguments for the optimizer.
        scheduler (str): Name of the learning rate scheduler.
        scheduler_kwargs (dict): Additional keyword arguments for the scheduler.
        losses (List[LossConfig]): List of loss configurations.
        label_encoder (Dict[str, int]): Dictionary for label encoding.

    Methods:
        from_yaml(cls, path: str) -> 'Config': Load configuration from a YAML file.
    """
    project_name: str
    experiment_name: str
    data_config: DataConfig
    augmentation_params: AugmentationConfig
    n_epochs: int
    num_classes: int
    accelerator: str
    device: int
    seed: int
    log_every_n_steps: int
    monitor_metric: str
    monitor_mode: str
    model_kwargs: dict
    optimizer: str
    optimizer_kwargs: dict
    scheduler: str
    scheduler_kwargs: dict
    losses: List[LossConfig]
    label_encoder: Dict[str, int]

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """
        Load configuration from a YAML file.

        Args:
            path (str): Path to the YAML configuration file.

        Returns:
            Config: Loaded configuration object.
        """
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)

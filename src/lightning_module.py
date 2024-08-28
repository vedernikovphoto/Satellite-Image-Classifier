import pytorch_lightning as pl
import torch
import torch.nn as nn
from timm import create_model

from src.config import Config
from src.losses import get_losses
from src.metrics import get_metrics
from src.train_utils import load_object


class PlanetModule(pl.LightningModule):
    """
    PyTorch Lightning module for the Planet dataset.

    Attributes:
        config (Config): Configuration object.
        model (torch.nn.Module): The model to be trained.
        losses (list): List of loss functions.
        valid_metrics (torchmetrics.Metric): Validation metrics.
        test_metrics (torchmetrics.Metric): Test metrics.
    """

    def __init__(self, config: Config):
        """
        Initializes the PlanetModule.

        Args:
            config (Config): Configuration object.
        """
        super().__init__()
        self._config = config

        self._model = create_model(num_classes=self._config.num_classes, **self._config.model_kwargs)
        self._losses = get_losses(self._config.losses)
        metrics = get_metrics(
            num_classes=self._config.num_classes,
            num_labels=self._config.num_classes,
            task='multilabel',
            average='macro',
            threshold=0.5,
        )
        self._valid_metrics = metrics.clone(prefix='val_')
        self._test_metrics = metrics.clone(prefix='test_')

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self._model(x)

    def get_core_model(self) -> nn.Module:
        """
        Returns the core PyTorch model for tracing.
        """
        return self._model
    
    def configure_optimizers(self) -> dict:
        """
        Configures the optimizers and learning rate scheduler.

        Returns:
            dict: Optimizer and scheduler configuration.
        """
        optimizer = load_object(self._config.optimizer)(
            self._model.parameters(),
            **self._config.optimizer_kwargs,
        )
        scheduler = load_object(self._config.scheduler)(optimizer, **self._config.scheduler_kwargs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self._config.monitor_metric,
                'interval': 'epoch',
                'frequency': 1,
            },
        }

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Training step.

        Args:
            batch (tuple): Batch of data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Training loss.
        """
        images, gt_labels = batch
        pr_logits = self(images)
        return self._calculate_loss(pr_logits, gt_labels, 'train_')

    def validation_step(self, batch, batch_idx) -> None:
        """
        Validation step.

        Args:
            batch (tuple): Batch of data.
            batch_idx (int): Batch index.
        """
        images, gt_labels = batch
        pr_logits = self(images)
        pr_labels = torch.sigmoid(pr_logits)
        self._valid_metrics(pr_labels, gt_labels)

    def test_step(self, batch, batch_idx) -> None:
        """
        Test step.

        Args:
            batch (tuple): Batch of data.
            batch_idx (int): Batch index.
        """
        images, gt_labels = batch
        pr_logits = self(images)
        pr_labels = torch.sigmoid(pr_logits)
        self._test_metrics(pr_labels, gt_labels)

    def on_validation_epoch_end(self) -> None:
        """Logs validation metrics at the end of each validation epoch and resets metrics."""
        self.log_dict(self._valid_metrics.compute(), on_epoch=True)
        self._valid_metrics.reset()

    def on_test_epoch_end(self) -> None:
        """Logs test metrics at the end of each test epoch."""
        self.log_dict(self._test_metrics.compute(), on_epoch=True)

    def _calculate_loss(
        self,
        pr_logits: torch.Tensor,
        gt_labels: torch.Tensor,
        prefix: str,
    ) -> torch.Tensor:
        """
        Calculates and logs the loss.

        Args:
            pr_logits (torch.Tensor): Predicted logits.
            gt_labels (torch.Tensor): Ground truth labels.
            prefix (str): Prefix for logging.

        Returns:
            torch.Tensor: Total loss.
        """
        total_loss = 0
        for cur_loss in self._losses:
            loss = cur_loss.loss(pr_logits, gt_labels)
            total_loss += cur_loss.weight * loss
            self.log(f'{prefix}{cur_loss.name}_loss', loss.item())
        self.log(f'{prefix}total_loss', total_loss.item())
        return total_loss

import argparse
import os
import torch
import logging

import pytorch_lightning as pl
from clearml import Task
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from src.config import Config
from src.constants import EXPERIMENTS_PATH
from src.datamodule import PlanetDM
from src.lightning_module import PlanetModule


logging.basicConfig(level=logging.INFO)


def save_torchscript_model(model, save_path, example_input) -> torch.Tensor:
    """
    Converts the PyTorch model to TorchScript and saves it.

    Args:
        model (nn.Module): The model to be saved.
        save_path (str): The path where to save the TorchScript model.
        example_input (torch.Tensor): Example input for tracing the model.
    """
    model.eval()
    traced_model = torch.jit.trace(model, example_input)
    torch.jit.save(traced_model, save_path)


def arg_parse() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    return parser.parse_args()


def prepare_task(config: Config) -> Task:
    """
    Initializes and prepares a ClearML task.

    Args:
        config (Config): Configuration object containing all training parameters.

    Returns:
        Task: The initialized ClearML task.
    """
    task = Task.init(
        project_name=config.project_name,
        task_name=f'{config.experiment_name}',
        auto_connect_frameworks=True,
    )
    task.connect(config.dict())
    return task


def train(config: Config) -> None:
    """
    Trains and tests the Planet model.

    Args:
        config (Config): Configuration object containing all training parameters.
    """
    datamodule = PlanetDM(config)
    model = PlanetModule(config)
    task = prepare_task(config)   # noqa: F841

    experiment_save_path = os.path.join(EXPERIMENTS_PATH, config.experiment_name)
    os.makedirs(experiment_save_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        experiment_save_path,
        monitor=config.monitor_metric,
        mode=config.monitor_mode,
        save_top_k=1,
        filename=f'epoch_{{epoch:02d}}-{{{config.monitor_metric}:.3f}}',
    )
    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        accelerator=config.accelerator,
        devices=[config.device],
        log_every_n_steps=config.log_every_n_steps,
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor=config.monitor_metric, patience=4, mode=config.monitor_mode),
            LearningRateMonitor(logging_interval='epoch'),
        ],
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(ckpt_path=checkpoint_callback.best_model_path, datamodule=datamodule)

    # Starting testing with the best checkpoint
    trainer.test(ckpt_path=checkpoint_callback.best_model_path, datamodule=datamodule)

    # Loading the best model from checkpoint
    best_model = PlanetModule.load_from_checkpoint(checkpoint_callback.best_model_path, config=config)

    # Extracting the core model from the loaded checkpoint
    core_model = best_model.get_core_model()

    # Moving core model to device: {config.device}
    core_model.to(config.device)

    # Setting core model to evaluation mode
    core_model.eval()

    # Generating example input for model tracing
    example_input = torch.randn(1, *config.data_config.input_size).to(config.device)

    # Saving the TorchScript model
    torchscript_model_path = os.path.join(experiment_save_path, 'model.pt')
    save_torchscript_model(core_model, torchscript_model_path, example_input)

    logging.info(f'TorchScript model saved successfully at {torchscript_model_path}')


if __name__ == '__main__':
    args = arg_parse()
    config = Config.from_yaml(args.config_file)
    pl.seed_everything(config.seed, workers=True)
    train(config)

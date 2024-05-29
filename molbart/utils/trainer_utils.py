from typing import List, Optional

import hydra
import math
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import Plugin

from molbart.utils.callbacks import CallbackCollection
from molbart.utils.scores import ScoreCollection


def instantiate_callbacks(callbacks_config: Optional[DictConfig]) -> CallbackCollection:
    """Instantiates callbacks from config."""
    callbacks = CallbackCollection()

    if not callbacks_config:
        print("No callbacks configs found! Skipping...")
        return callbacks

    callbacks.load_from_config(callbacks_config)
    return callbacks


def instantiate_scorers(scorer_config: Optional[DictConfig]) -> CallbackCollection:
    """Instantiates scorer from config."""

    scorer = ScoreCollection()
    if not scorer_config:
        print("No scorer configs found! Skipping...")
        return scorer

    scorer.load_from_config(scorer_config)
    return scorer


def instantiate_logger(logger_config: Optional[DictConfig]) -> TensorBoardLogger:
    """Instantiates logger from config."""
    logger: TensorBoardLogger = []

    if not logger_config:
        print("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_config, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    if isinstance(logger_config, DictConfig) and "_target_" in logger_config:
        print(f"Instantiating logger <{logger_config._target_}>")
        logger = hydra.utils.instantiate(logger_config)

    return logger


def instantiate_plugins(plugin_cfg: Optional[DictConfig]) -> List[Plugin]:
    """Instantiates plugins from config."""
    plugin: list[Plugin] = []

    if not plugin_cfg:
        print("No plugin configs found! Skipping...")
        return plugin

    if not isinstance(plugin_cfg, DictConfig):
        raise TypeError("Plugin config must be a DictConfig!")

    for _, plugin_conf in plugin_cfg.items():
        if isinstance(plugin_conf, DictConfig) and "_target_" in plugin_conf:
            print(f"Instantiating logger <{plugin_conf._target_}>")
            plugin.append(hydra.utils.instantiate(plugin_conf))

    return plugin


def calc_train_steps(args, dm, n_gpus=None):
    n_gpus = getattr(args, "n_gpus", n_gpus)
    dm.setup()
    if n_gpus > 0:
        batches_per_gpu = math.ceil(len(dm.train_dataloader()) / float(n_gpus))
    else:
        raise ValueError("Number of GPUs should be > 0 in training.")
    train_steps = math.ceil(batches_per_gpu / args.acc_batches) * args.n_epochs
    return train_steps


def build_trainer(config, n_gpus=None):

    print("Instantiating loggers...")
    logger = instantiate_logger(config.get("logger"))

    print("Instantiating callbacks...")
    callbacks: CallbackCollection = instantiate_callbacks(config.get("callbacks"))

    print("Instantiating plugins...")
    plugins: list[Plugin] = instantiate_plugins(config.get("plugin"))

    if n_gpus > 1:
        config.trainer.accelerator = "ddp"
    else:
        plugins = None

    print("Building trainer...")
    trainer: pl.Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks.objects(), logger=logger, plugins=plugins
    )
    print("Finished trainer.")

    print(f"Default logging and checkpointing directory: {trainer.default_root_dir} or {trainer.weights_save_path}")
    return trainer

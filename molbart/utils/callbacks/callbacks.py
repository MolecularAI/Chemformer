from __future__ import annotations
import os
from typing import Any, Dict, Optional
from typing import TYPE_CHECKING

import pandas as pd
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import torch
from pytorch_lightning.callbacks import Callback

if TYPE_CHECKING:
    from molbart.models import _AbsTransformerModel


class LearningRateMonitor(plc.LearningRateMonitor):
    callback_name = "LearningRateMonitor"

    def __init__(self, logging_interval: str = "step", log_momentum: bool = False, **kwargs: Any) -> None:
        super().__init__(logging_interval=logging_interval, log_momentum=log_momentum, **kwargs)

    def __repr__(self):
        return self.callback_name


class ModelCheckpoint(plc.ModelCheckpoint):
    callback_name = "ModelCheckpoint"

    def __init__(
        self,
        dirpath: Optional[str] = None,
        filename: Optional[str] = None,
        monitor: str = "validation_loss",
        verbose: bool = False,
        save_last: bool = True,
        save_top_k: int = 3,
        save_weights_only: bool = False,
        mode: str = "auto",
        period: int = 1,
        prefix: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            period=period,
            prefix=prefix,
            **kwargs,
        )

    def __repr__(self):
        return self.callback_name


class StepCheckpoint(Callback):
    callback_name = "StepCheckpoint"

    def __init__(self, step_interval: int = 50000) -> None:
        super().__init__()

        if not isinstance(step_interval, int):
            raise TypeError(f"step_interval must be of type int, got type {type(step_interval)}")

        self.step_interval = step_interval

    def __repr__(self):
        return self.callback_name

    # def on_batch_end(self, trainer, model):
    # Ideally this should on_after_optimizer_step, but that isn't available in pytorch lightning (yet?)
    def on_after_backward(self, trainer: pl.Trainer, model: _AbsTransformerModel) -> None:
        step = trainer.global_step
        if (step != 0) and (step % self.step_interval == 0):
            # if (step % self.step_interval == 0):
            self._save_model(trainer, model, step)

    def _save_model(self, trainer: pl.Trainer, model: _AbsTransformerModel, step: int) -> None:
        if trainer.logger is not None:
            if trainer.weights_save_path != trainer.default_root_dir:
                save_dir = trainer.weights_save_path
            else:
                save_dir = trainer.logger.save_dir or trainer.default_root_dir

            version = (
                trainer.logger.version
                if isinstance(trainer.logger.version, str)
                else f"version_{trainer.logger.version}"
            )
            version, name = trainer.training_type_plugin.broadcast((version, trainer.logger.name))
            ckpt_path = os.path.join(save_dir, str(name), version, "checkpoints")

        else:
            ckpt_path = os.path.join(trainer.weights_save_path, "checkpoints")

        save_path = f"{ckpt_path}/step={str(step)}.ckpt"
        print(f"Saving step checkpoint in {save_path}")
        trainer.save_checkpoint(save_path)


class OptLRMonitor(Callback):
    callback_name = "OptLRMonitor"

    def __init__(self) -> None:
        super().__init__()

    def __repr__(self):
        return self.callback_name

    def on_train_batch_start(self, trainer: pl.Trainer, *args: Any, **kwargs: Any) -> None:
        # Only support one optimizer
        opt = trainer.optimizers[0]

        # Only support one param group
        stats = {"lr-Adam": opt.param_groups[0]["lr"]}
        trainer.logger.log_metrics(stats, step=trainer.global_step)


class ValidationScoreCallback(Callback):
    """
    Retrieving scores from the validation epochs and write to file continuously.
    """

    callback_name = "ValidationScoreCallback"

    def __init__(self) -> None:
        super().__init__()
        self._metrics = pd.DataFrame()
        self._skip_logging = True

    def __repr__(self):
        return self.callback_name

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._skip_logging:
            self._skip_logging = False
            return

        logged_metrics = {
            key: [val.to(torch.device("cpu")).numpy()]
            for key, val in trainer.callback_metrics.items()
            if key != "mol_acc"
        }

        metrics = {"epoch": pl_module.current_epoch}
        metrics.update(logged_metrics)
        metrics_df = pd.DataFrame(metrics)

        self._metrics = pd.concat([self._metrics, metrics_df], axis=0, ignore_index=True)

        self.out_directory = self._get_out_directory(trainer)
        self._save_logged_data()
        return

    def _get_out_directory(self, trainer: pl.Trainer) -> str:
        if trainer.logger is not None:
            if trainer.weights_save_path != trainer.default_root_dir:
                save_dir = trainer.weights_save_path
            else:
                save_dir = trainer.logger.save_dir or trainer.default_root_dir

            version = (
                trainer.logger.version
                if isinstance(trainer.logger.version, str)
                else f"version_{trainer.logger.version}"
            )
            version, name = trainer.training_type_plugin.broadcast((version, trainer.logger.name))
            data_path = os.path.join(save_dir, str(name), version)
        else:
            data_path = trainer.weights_save_path
        return data_path

    def _save_logged_data(self) -> None:
        """
        Retrieve and write data (model validation) logged during training.
        """
        outfile = self.out_directory + "/logged_train_metrics.csv"
        self._metrics.to_csv(outfile, sep="\t", index=False)
        print("Logged training/validation set loss written to: " + outfile)
        return


class ScoreCallback(Callback):
    """
    Retrieving scores from test step and write to file continuously.
    """

    callback_name = "ScoreCallback"

    def __init__(
        self,
        output_scores: str = "metrics_scores.csv",
        output_sampled_smiles: str = "sampled_smiles.json",
    ) -> None:
        super().__init__()
        self._metrics = pd.DataFrame()
        self._sampled_smiles = pd.DataFrame()

        self._metrics_output = output_scores
        self._smiles_output = output_sampled_smiles

    def __repr__(self):
        return self.callback_name

    def set_output_files(self, output_score_data: str, output_sampled_smiles: str) -> None:
        self._metrics_output = output_score_data
        self._smiles_output = output_sampled_smiles

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        model: _AbsTransformerModel,
        test_output: Dict[str, Any],
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        smiles_keys = [
            "sampled_molecules",
            "sampled_molecules(unique)",
            "target_smiles",
        ]

        logged_metrics = {key: [val] for key, val in test_output.items() if key not in smiles_keys}

        for key, val in logged_metrics.items():
            if isinstance(val[0], torch.Tensor):
                logged_metrics[key] = [val[0].to(torch.device("cpu")).numpy()]

        sampled_smiles = {key: [val] for key, val in test_output.items() if key in smiles_keys}

        metrics_df = pd.DataFrame(logged_metrics)
        sampled_smiles_df = pd.DataFrame(sampled_smiles)

        self._metrics = pd.concat([self._metrics, metrics_df], axis=0, ignore_index=True)
        self._sampled_smiles = pd.concat([self._sampled_smiles, sampled_smiles_df], axis=0, ignore_index=True)

        self._save_logged_data()

    def _save_logged_data(self) -> None:
        """
        Retrieve and write data (model validation) logged during training.
        """
        self._metrics.to_csv(self._metrics_output, sep="\t", index=False)
        print("Test set metrics written to file: " + self._metrics_output)

        self._sampled_smiles.to_json(self._smiles_output, orient="table")
        print("Test set sampled smiles written to file: " + self._smiles_output)
        return

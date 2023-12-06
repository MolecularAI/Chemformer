import copy
import math
import os
from argparse import Namespace

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DeepSpeedPlugin

from molbart.models.transformer_models import BARTModel, UnifiedModel
from molbart.modules.data.mol_data import ChemblDataModule, ZincDataModule
from molbart.modules.data.seq2seq_data import (
    MolOptDataModule,
    SynthesisDataModule,
    Uspto50DataModule,
    UsptoMixedDataModule,
    UsptoSepDataModule,
)

# Default model hyperparams
DEFAULT_D_MODEL = 512
DEFAULT_NUM_LAYERS = 6
DEFAULT_NUM_HEADS = 8
DEFAULT_D_FEEDFORWARD = 2048
DEFAULT_ACTIVATION = "gelu"
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_DROPOUT = 0.1

DEFAULT_MODEL = "bart"
DEFAULT_DATASET_TYPE = "synthesis"
DEFAULT_DEEPSPEED_CONFIG_PATH = "ds_config.json"
DEFAULT_LOG_DIR = "tb_logs"
DEFAULT_VOCAB_PATH = "bart_vocab.json"
DEFAULT_CHEM_TOKEN_START = 272
REGEX = r"\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]"

DEFAULT_GPUS = 1
DEFAULT_NUM_NODES = 1

USE_GPU = True
use_gpu = USE_GPU and torch.cuda.is_available()


class StepCheckpoint(Callback):
    def __init__(self, step_interval):
        super().__init__()

        if type(step_interval) != int:
            raise TypeError(
                f"step_interval must be of type int, got type {type(step_interval)}"
            )

        self.step_interval = step_interval

    # def on_batch_end(self, trainer, model):
    # Ideally this should on_after_optimizer_step, but that isn't available in pytorch lightning (yet?)
    def on_after_backward(self, trainer, model):
        step = trainer.global_step
        if (step != 0) and (step % self.step_interval == 0):
            # if (step % self.step_interval == 0):
            self._save_model(trainer, model, step)

    def _save_model(self, trainer, model, step):
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
            version, name = trainer.training_type_plugin.broadcast(
                (version, trainer.logger.name)
            )
            ckpt_path = os.path.join(save_dir, str(name), version, "checkpoints")

        else:
            ckpt_path = os.path.join(trainer.weights_save_path, "checkpoints")

        save_path = f"{ckpt_path}/step={str(step)}.ckpt"
        print(f"Saving step checkpoint in {save_path}")
        trainer.save_checkpoint(save_path)


class OptLRMonitor(Callback):
    def __init__(self):
        super().__init__()

    def on_train_batch_start(self, trainer, *args, **kwargs):
        # Only support one optimizer
        opt = trainer.optimizers[0]

        # Only support one param group
        stats = {"lr-Adam": opt.param_groups[0]["lr"]}
        trainer.logger.log_metrics(stats, step=trainer.global_step)


class MetricsCallback(Callback):
    """
    Storing metrics in lists during training for later retrieval and processing.
    """

    def __init__(self):
        super().__init__()
        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.val_token_acc = []
        self.perplexity = []
        self.mol_acc = []
        self.skip_logging = True

    def _get_out_directory(self, trainer):
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
            version, name = trainer.training_type_plugin.broadcast(
                (version, trainer.logger.name)
            )
            data_path = os.path.join(save_dir, str(name), version)
        else:
            data_path = trainer.weights_save_path
        return data_path

    def _get_dataframe(self):
        """
        Build dataframe from logged metrics, which are stored in self.trainer.callbacks[2]
        """
        epochs = self.epochs
        train_loss = self.train_loss
        val_loss = self.val_loss
        val_token_acc = self.val_token_acc
        perplexity = self.perplexity
        mol_acc = self.mol_acc

        epochs = np.array(epochs).ravel()
        train_loss = np.array(
            [sample.to(torch.device("cpu")).numpy() for sample in train_loss]
        ).ravel()
        val_loss = np.array(
            [sample.to(torch.device("cpu")).numpy() for sample in val_loss]
        ).ravel()
        val_token_acc = np.array(
            [sample.to(torch.device("cpu")).numpy() for sample in val_token_acc]
        ).ravel()
        perplexity = np.array(
            [sample.to(torch.device("cpu")).numpy() for sample in perplexity]
        ).ravel()
        mol_acc = np.array(
            [sample.to(torch.device("cpu")).numpy() for sample in mol_acc]
        ).ravel()

        df = pd.DataFrame(epochs, columns=["epoch"])
        df["training_loss"] = train_loss
        df["validation_loss"] = val_loss
        df["validation_token_accuracy"] = val_token_acc
        df["perplexity"] = perplexity
        df["molecular_accuracy"] = mol_acc
        return df

    def _save_logged_data(self):
        """
        Retrieve and write data (model validation) logged during training.
        """
        metrics_df = self._get_dataframe()
        outfile = self.out_directory + "/logged_train_metrics.csv"
        metrics_df.to_csv(outfile, sep="\t", index=False)
        print("Logged training/validation set loss written to: " + outfile)
        return

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.skip_logging:
            self.skip_logging = False
            return
        current_metrics = copy.deepcopy(trainer.callback_metrics)

        self.epochs.append(pl_module.current_epoch)
        self.train_loss.append(current_metrics["train_loss"])
        self.val_loss.append(current_metrics["val_loss"])
        self.val_token_acc.append(current_metrics["val_token_accuracy"])
        self.perplexity.append(current_metrics["perplexity"])
        self.mol_acc.append(current_metrics["val_molecular_accuracy"])

        self.out_directory = self._get_out_directory(trainer)
        self._save_logged_data()
        return


def build_molecule_datamodule(args, tokenizer, augment=None, masker=None):
    dm_cls = {
        "chembl": ChemblDataModule,
        "zinc": ZincDataModule,
    }
    dm = dm_cls[args.dataset_type](
        task=args.task,
        augment=args.augmentation_strategy if augment is None else augment,
        masker=masker,
        dataset_path=args.data_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        train_token_batch_size=args.train_tokens,
        num_buckets=args.n_buckets,
        unified_model=args.model_type == "unified",
    )
    return dm


def build_seq2seq_datamodule(args, tokenizer, forward=True):
    dm_cls = {
        "uspto_50": Uspto50DataModule,
        "uspto_50_with_type": Uspto50DataModule,
        "uspto_mixed": UsptoMixedDataModule,
        "uspto_sep": UsptoSepDataModule,
        "mol_opt": MolOptDataModule,
        "synthesis": SynthesisDataModule,
    }
    kwargs = {
        "uspto_50_with_type": {
            "include_type_token": True,
        }
    }
    dm = dm_cls[args.dataset_type](
        augment_prob=getattr(args, "augmentation_probability", 0.0),
        reverse=not forward,
        dataset_path=args.data_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_len=getattr(args, "max_seq_len", DEFAULT_MAX_SEQ_LEN),
        train_token_batch_size=args.train_tokens,
        num_buckets=args.n_buckets,
        unified_model=args.model_type == "unified",
        **kwargs.get(args.dataset_type, {}),
    )
    return dm


def build_trainer(args, n_gpus=None, dataset=None):
    n_gpus = getattr(args, "n_gpus", n_gpus)
    dataset = getattr(args, "dataset_type", dataset)

    logger = TensorBoardLogger(args.output_directory, name=args.task)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    plugins = None
    accelerator = None
    if n_gpus > 1:
        print(f"Injecting DeepSpeed plugin for multi-gpu ({n_gpus}) training")
        accelerator = "ddp"
        lr_monitor = OptLRMonitor()
        plugins = [DeepSpeedPlugin(config=args.deepspeed_config_path)]

    callbacks = []
    check_val = args.check_val_every_n_epoch

    # Zinc is so big we need to checkpoint more frequently than every epoch
    if dataset in ["zinc", "synthesis"]:
        checkpoint_freq = args.checkpoint_every_n_step
        intra_epoch_checkpoint = StepCheckpoint(checkpoint_freq)
        callbacks.append(intra_epoch_checkpoint)
        print(
            f"Big dataset requested ({dataset}), enabling intra-epoch checkpointing every {checkpoint_freq} optimizer steps)"
        )
        # check_val = 1

    # Checkpoint should ideally depend on the validation metric mol_acc
    # However, this metric is only available if validation has been run
    # This currently lead to a lot of errors, due to unalignment of intra-epoch checkpointing and validation run
    checkpoint_cb = ModelCheckpoint(
        monitor="train_loss", period=check_val, save_last=True, save_top_k=3
    )
    metrics_cb = MetricsCallback()

    print(f"Enabling model checkpointing every {check_val} epochs")
    print(f"Num gpus: {n_gpus}")
    print(f"Accelerator: {accelerator}")

    callbacks.extend([lr_monitor, checkpoint_cb, metrics_cb])
    print(f"Callbacks: {callbacks}")
    trainer = Trainer(
        accelerator=accelerator,
        logger=logger,
        gpus=n_gpus,
        num_nodes=args.n_nodes,
        min_epochs=args.n_epochs,
        max_epochs=args.n_epochs,
        accumulate_grad_batches=args.acc_batches,
        gradient_clip_val=args.clip_grad,
        limit_val_batches=args.limit_val_batches,
        callbacks=callbacks,
        plugins=plugins,
        check_val_every_n_epoch=check_val,
        precision=16,
    )
    print(
        f"Default logging and checkpointing directory: {trainer.default_root_dir} or {trainer.weights_save_path}"
    )
    return trainer


def seed_everything(seed):
    pl.utilities.seed.seed_everything(seed)


def load_bart(args, sampler):
    model = BARTModel.load_from_checkpoint(args.model_path, decode_sampler=sampler)
    model.eval()
    return model


def load_unified(args, sampler):
    model = UnifiedModel.load_from_checkpoint(args.model_path, decode_sampler=sampler)
    model.eval()
    return model


def calc_train_steps(args, dm, n_gpus=None):
    n_gpus = getattr(args, "n_gpus", n_gpus)
    dm.setup()
    if n_gpus > 0:
        batches_per_gpu = math.ceil(len(dm.train_dataloader()) / float(n_gpus))
    else:
        raise ValueError("Number of GPUs should be > 0 in training.")
    train_steps = math.ceil(batches_per_gpu / args.acc_batches) * args.n_epochs
    return train_steps


def get_chemformer_args(args):
    if args.task in ["forward_prediction", "mol_opt"]:
        forward_prediction = True
    elif args.task == "backward_prediction":
        forward_prediction = False
    else:
        raise ValueError(f"Unknown task {args.task}")

    data_args = {
        "data_path": getattr(args, "data_path", ""),
        "reactants_path": getattr(args, "reactants_path", ""),
        "dataset_type": getattr(args, "dataset_type", ""),
        "max_seq_len": getattr(args, "max_seq_len", DEFAULT_MAX_SEQ_LEN),
        "augmentation_probability": getattr(args, "augmentation_probability", 0.0),
        "batch_size": args.batch_size,
        "train_tokens": getattr(args, "train_tokens", None),
        "n_buckets": getattr(args, "n_buckets", None),
        "model_type": getattr(args, "model_type", "bart"),
        "forward_prediction": forward_prediction,
    }

    model_args = {
        "output_directory": getattr(args, "output_directory", ""),
        "task": args.task,
        "model_type": getattr(args, "model_type", "bart"),
        "acc_batches": getattr(args, "acc_batches", None),
        "d_model": getattr(args, "d_model", None),
        "n_layers": getattr(args, "n_layers", None),
        "n_heads": getattr(args, "n_heads", None),
        "d_feedforward": getattr(args, "d_feedforward", None),
        "n_epochs": getattr(args, "n_epochs", None),
        "augmentation_strategy": getattr(args, "augmentation_strategy", None),
        "augmentation_probability": getattr(args, "augmentation_probability", 0.0),
        "warm_up_steps": getattr(args, "warm_up_steps", None),
        "deepspeed_config_path": getattr(args, "deepspeed_config_path", None),
        "learning_rate": getattr(args, "learning_rate", None),
        "weight_decay": getattr(args, "weight_decay", None),
        "clip_grad": getattr(args, "clip_grad", None),
        "schedule": getattr(args, "schedule", None),
        "limit_val_batches": getattr(args, "limit_val_batches", None),
        "check_val_every_n_epoch": getattr(args, "check_val_every_n_epoch", None),
        "checkpoint_every_n_step": getattr(args, "checkpoint_every_n_step", None),
        "n_nodes": getattr(args, "n_nodes", DEFAULT_NUM_NODES),
    }

    return Namespace(**model_args), Namespace(**data_args)


def _clean_string(x, expr_list):
    y = copy.copy(x)
    y = y.replace("''", "&")  # Mark empty SMILES string with dummy character
    for expr in expr_list:
        y = y.replace(expr, "")
    return y


def _convert_to_array(data_list):
    data_new = np.zeros(len(data_list), dtype="object")
    for ix, x in enumerate(data_list):
        data_new[ix] = x
    return data_new


def read_score_tsv(
    filename,
    str_to_list_columns,
    is_numeric,
    expr_list1=["'", "[array([", "array([", "[array(", "array(", " ", "\n"],
):
    """
    Read TSV-file generated by the Chemformer.score_model() function.
    Args:
    - filename: str (path to .csv file)
    - str_to_list_columns: list(str) (list of columns to convert from string to nested list)
    - is_numeric: list(bool) (list denoting which columns contain strings that should be converted to lists of floats)
    """

    sep = ","
    numeric_expr_list = ["(", ")", "[", "]", "\n"]
    data = pd.read_csv(filename, sep="\t")

    for col, to_float in zip(str_to_list_columns, is_numeric):
        print("Converting string to data of column: " + col)
        data_str = data[col].values

        data_list = []
        for X in data_str:
            X = [x for x in X.split(sep) if "dtype=" not in x]
            inner_list = []
            X_new = []
            is_last_molecule = False
            for x in X:
                x = _clean_string(x, expr_list1)

                if x == "":
                    continue

                if x[-1] == ")" and sum([token == "(" for token in x]) < sum(
                    [token == ")" for token in x]
                ):
                    x = x[:-1]
                    is_last_molecule = True

                if x[-1] == "]" and sum([token == "[" for token in x]) < sum(
                    [token == "]" for token in x]
                ):
                    x = x[:-1]
                    is_last_molecule = True

                inner_list.append(x)

                if is_last_molecule:
                    if to_float:
                        inner_list = [
                            _clean_string(element, numeric_expr_list)
                            for element in inner_list
                        ]
                        inner_list = [
                            float(element) if element != "" else np.nan
                            for element in inner_list
                        ]
                    X_new.append(inner_list)
                    inner_list = []
                    is_last_molecule = False

            print(
                "Batch size after cleaning (for validating cleaning): "
                + str(len(X_new))
            )
            data_list.append(X_new)
        data.drop(columns=[col], inplace=True)
        data_list = _convert_to_array(data_list)
        data[col] = data_list
    return data

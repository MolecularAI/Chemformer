import os
import math
import json
import torch
import pickle
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DeepSpeedPlugin
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback

from molbart.tokeniser import MolEncTokeniser
from molbart.models.pre_train import BARTModel, UnifiedModel
from molbart.data.datasets import Chembl, Uspto50, UsptoMixed, UsptoSep, MolOpt, Zinc, ZincSlice
from molbart.data.datamodules import MoleculeDataModule, FineTuneReactionDataModule


# Default model hyperparams
DEFAULT_D_MODEL = 512
DEFAULT_NUM_LAYERS = 6
DEFAULT_NUM_HEADS = 8
DEFAULT_D_FEEDFORWARD = 2048
DEFAULT_ACTIVATION = "gelu"
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_DROPOUT = 0.1

DEFAULT_DEEPSPEED_CONFIG_PATH = "ds_config.json"
DEFAULT_LOG_DIR = "tb_logs"
DEFAULT_VOCAB_PATH = "bart_vocab.txt"
DEFAULT_CHEM_TOKEN_START = 272
REGEX = "\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]"

DEFAULT_GPUS = 1
DEFAULT_NUM_NODES = 1

USE_GPU = True
use_gpu = USE_GPU and torch.cuda.is_available()


class StepCheckpoint(Callback):
    def __init__(self, step_interval):
        super().__init__()

        if type(step_interval) != int:
            raise TypeError(f"step_interval must be of type int, got type {type(step_interval)}")

        self.step_interval = step_interval

    def on_batch_end(self, trainer, model):
        step = trainer.global_step
        if step % self.step_interval == 0:
            self._save_model(trainer, model, step)

    def _save_model(self, trainer, model, step):
        if trainer.logger is not None:
            if trainer.weights_save_path != trainer.default_root_dir:
                save_dir = trainer.weights_save_path
            else:
                save_dir = trainer.logger.save_dir or trainer.default_root_dir

            version = (
                trainer.logger.version
                if isinstance(trainer.logger.version, str) else f"version_{trainer.logger.version}"
            )
            version, name = trainer.training_type_plugin.broadcast((version, trainer.logger.name))
            ckpt_path = os.path.join(save_dir, str(name), version, "checkpoints")

        else:
            ckpt_path = os.path.join(trainer.weights_save_path, "checkpoints")

        save_path = f"{ckpt_path}/step={str(step)}.ckpt"
        trainer.save_checkpoint(save_path)


class OptLRMonitor(Callback):
    def __init__(self):
        super().__init__()

    def on_train_batch_start(self, trainer, *args, **kwargs):
        # Only support one optimizer
        opt = trainer.optimizers[0]

        # Only support one param group
        stats = {
            "lr-Adam":  opt.param_groups[0]["lr"]
        }
        trainer.logger.log_metrics(stats, step=trainer.global_step)


def number_of_mols(data_path):
    path = Path(data_path)

    idx_file_mapping = []
    if path.is_dir():
        num_lines = 0
        for f in path.iterdir():
            text = f.read_text()
            num_mols = len(text.split("\n")) - 1
            idx_file_mapping.append((num_lines, num_lines + num_mols, f))
            num_lines += num_mols

    else:
        text = path.read_text()
        num_lines = len(text.split("\n"))
        idx_file_mapping.append((0, num_lines, path))

    return num_lines, idx_file_mapping


def read_df_slice(idxs, idx_file_mapping):
    """ Read a slice of the dataset from disk by looking up the required files in the mapping

    Args:
        idxs (List[int]): Contiguous list of indices into the full dataset of molecules to read 
        idx_file_mapping (dict): Mapping returned by number_of_mols function

    Returns:
        (pd.DataFrame): DataFrame of lines from dataset 
    """

    file_idx_map = {}

    curr_idx = 0
    for start, end, file_path in idx_file_mapping:
        while curr_idx < len(idxs) and start <= idxs[curr_idx] < end:
            file_idx_map.setdefault(str(file_path), [])
            file_idx_map[str(file_path)].append(idxs[curr_idx] - start)
            curr_idx += 1

    dfs = []
    for file_path, file_idxs in file_idx_map.items():
        file_df = pd.read_csv(Path(file_path))
        df = file_df.iloc[file_idxs]
        dfs.append(df)

    df_slice = pd.concat(dfs, ignore_index=True, copy=False)
    return df_slice


def read_zinc_slice(data_path, rank, num_gpus, batch_size):
    num_mols, idx_file_mapping = number_of_mols(data_path)
    rank_idxs = [idxs.tolist() for idxs in np.array_split(list(range(num_mols)), num_gpus)]

    # Drop last mols to ensure all processes have the same number of batches
    num_mols = min([len(idxs) for idxs in rank_idxs])
    num_mols = batch_size * (num_mols // batch_size)
    idxs = rank_idxs[rank][:num_mols]

    df_slice = read_df_slice(idxs, idx_file_mapping)
    print(f"Read {str(len(df_slice.index))} molecules for gpu {str(rank)}")
    dataset = ZincSlice(df_slice)
    return dataset


def build_dataset(args, forward=True):
    aug_prob = getattr(args, "aug_prob", None)
    if args.dataset == "uspto_50":
        dataset = Uspto50(args.data_path, aug_prob, forward=forward)
        print("Using USPTO 50K dataset without type tokens.")
    elif args.dataset == "uspto_50_with_type":
        dataset = Uspto50(args.data_path, aug_prob, type_token=True, forward=forward)
        print("Using USPTO 50K dataset with type tokens.")
    elif args.dataset == "uspto_mixed":
        dataset = UsptoMixed(args.data_path, aug_prob)
        print("Using USPTO MIT Mixed dataset.")
    elif args.dataset == "uspto_sep":
        dataset = UsptoSep(args.data_path, aug_prob)
        print("Using USPTO MIT Separated dataset.")
    elif args.dataset == "mol_opt":
        dataset = MolOpt(args.data_path, aug_prob)
        print("Using Molecular Optimisation dataset.")
    elif args.dataset == "chembl":
        dataset = Chembl(args.data_path)
        print("Using Chembl dataset.")
    elif args.dataset == "zinc":
        dataset = Zinc(args.data_path)
        print("Using ZINC dataset.")
    else:
        raise ValueError(f"Unknown dataset {args.dataset}.")

    return dataset


def build_molecule_datamodule(args, dataset, tokeniser, augment=None):
    augment = args.augment if augment is None else augment
    uni_model = args.model_type == "unified"
    dm = MoleculeDataModule(
        dataset, 
        tokeniser,
        args.batch_size,  
        args.max_seq_len,
        args.task,
        train_token_batch_size=args.train_tokens,
        num_buckets=args.num_buckets,
        val_idxs=dataset.val_idxs,
        test_idxs=dataset.test_idxs,
        augment=augment,
        unified_model=uni_model
    )
    return dm


def build_reaction_datamodule(args, dataset, tokeniser, forward=True):
    uni_model = args.model_type == "unified"
    dm = FineTuneReactionDataModule(
        dataset,
        tokeniser,
        args.batch_size,
        DEFAULT_MAX_SEQ_LEN,
        forward_pred=forward,
        val_idxs=dataset.val_idxs,
        test_idxs=dataset.test_idxs,
        train_token_batch_size=args.train_tokens,
        num_buckets=args.num_buckets,
        unified_model=uni_model
    )
    return dm


def load_tokeniser(vocab_path, chem_token_start):
    tokeniser = MolEncTokeniser.from_vocab_file(vocab_path, REGEX, chem_token_start)
    return tokeniser


def build_trainer(args):
    logger = TensorBoardLogger(args.log_dir, name=args.task)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_cb = ModelCheckpoint(monitor="val_molecular_accuracy", save_last=True)

    plugins = None
    accelerator = None
    if args.gpus > 1:
        accelerator = "ddp"
        lr_monitor = OptLRMonitor()
        plugins = [DeepSpeedPlugin(config=args.deepspeed_config_path)]

    callbacks = [lr_monitor, checkpoint_cb]

    # Zinc is so big we need to checkpoint more frequently than every epoch
    check_val = 10
    if args.dataset == "zinc":
        checkpoint_freq = 50000
        intra_epoch_checkpoint = StepCheckpoint(checkpoint_freq)
        callbacks.append(intra_epoch_checkpoint)
        check_val = 1

    print(f"Num gpus: {args.gpus}")
    print(f"Accelerator: {accelerator}")

    trainer = Trainer(
        accelerator=accelerator,
        logger=logger,
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        min_epochs=args.epochs,
        max_epochs=args.epochs,
        accumulate_grad_batches=args.acc_batches,
        gradient_clip_val=args.clip_grad,
        limit_val_batches=args.limit_val_batches,
        callbacks=callbacks,
        plugins=plugins,
        check_val_every_n_epoch=check_val,
        precision=16
    )
    return trainer


def seed_everything(seed):
    pl.utilities.seed.seed_everything(seed)


def load_bart(args, sampler):
    model = BARTModel.load_from_checkpoint(
        args.model_path,
        decode_sampler=sampler
    )
    model.eval()
    return model


def load_unified(args, sampler):
    model = UnifiedModel.load_from_checkpoint(
        args.model_path,
        decode_sampler=sampler
    )
    model.eval()
    return model


def calc_train_steps(args, dm):
    dm.setup()
    batches_per_gpu = math.ceil(len(dm.train_dataloader()) / float(args.gpus))
    train_steps = math.ceil(batches_per_gpu / args.acc_batches) * args.epochs
    return train_steps


def print_results(args, results):
    print(f"Results for model: {args.model_path}")
    print(f"{'Item':<25}Result")
    for key, val in results.items():
        print(f"{key:<25} {val:.4f}")

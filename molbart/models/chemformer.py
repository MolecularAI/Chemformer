import os
from argparse import Namespace
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import DataLoader

from molbart.data import DataCollection
import molbart.utils.data_utils as util
from molbart.models import BARTModel, UnifiedModel
from molbart.utils.samplers import BeamSearchSampler
from molbart.utils.tokenizers import ChemformerTokenizer
from molbart.utils import trainer_utils

DEFAULT_WEIGHT_DECAY = 0


class Chemformer:
    """
    Class for building (synthesis) Chemformer model, fine-tuning seq-seq model,
    and predicting/scoring model.
    """

    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        """
        Args:
            config: OmegaConf config loaded by hydra. Contains the input args of the model,
                including vocabulary, model checkpoint, beam size, etc.

            The config includes the following arguments:
                # Trainer args
                seed: 1
                batch_size: 128
                n_gpus (int): Number of GPUs to use.
                i_chunk: 0              # For inference
                n_chunks: 1             # For inference
                limit_val_batches: 1.0  # For training
                n_buckets: 12           # For training
                n_nodes: 1              # For training
                acc_batches: 1          # For training
                accelerator: null       # For training

                # Data args
                data_path (str): path to data used for training or inference
                backward_predictions (str): path to sampled smiles (for round-trip inference)
                dataset_part (str): Which dataset split to run inference on. ["full", "train", "val", "test"]
                dataset_type (str): The specific dataset type used as input.
                datamodule_type (Optinal[str]): The type of datamodule to build (seq2seq).
                vocabulary_path (str): path to bart_vocabulary.
                task (str): the model task ["forward_prediction", "backward_prediction"]
                data_device (str): device used for handling the data in optimized beam search (use cpu if memor issues).

                # Model args
                model_path (Optional[str]): Path to model weights.
                model_type (str): the model type ["bart", "unified"]
                n_beams (int): Number of beams / predictions from the sampler.
                n_unique_beams (Optional[int]): Restrict number of unique predictions.
                    If None => return all unique solutions.
                train_mode(str): Whether to train the model ("training") or use
                    model for evaluations ("eval").

                train_mode (str): Whether to train the model ("training") or use
                    model for evaluations ("eval").
                device (str): Which device to run model and beam search on ("cuda" / "cpu").
                resume_training (bool): Whether to continue training from the supplied
                    .ckpt file.

                learning_rate (float): the learning rate (for training/fine-tuning)
                weight_decay (float): the weight decay (for training/fine-tuning)

                # Molbart model parameters
                d_model (int): 512
                n_layers (int): 6
                n_heads (int): 8
                d_feedforward (int): 2048

                callbacks: list of Callbacks
                datamodule: the DataModule to use

                # Inference args
                scorers: list of Scores to evaluate sampled smiles against target smiles
                output_score_data: null
                output_sampled_smiles: null
        """

        self.config = config

        self.train_mode = config.train_mode
        print(f"train mode: {self.train_mode}")
        self.train_tokens = config.get("train_tokens")
        self.n_buckets = config.get("n_buckets")
        self.resume_training = False
        if self.train_mode.startswith("train"):
            self.resume_training = config.resume

            if self.resume_training:
                print("Resuming training.")

        device = config.get("device", "cuda")
        data_device = config.get("data_device", "cuda")
        if config.n_gpus < 1:
            device = "cpu"
            data_device = "cpu"

        self.device = device

        self.tokenizer = ChemformerTokenizer(filename=config.vocabulary_path)

        self.model_type = config.model_type
        self.model_path = config.model_path

        self.n_gpus = config.n_gpus
        self.is_data_setup = False
        self.set_datamodule(datamodule_type=config.get("datamodule"))

        print("Vocabulary_size: " + str(len(self.tokenizer)))
        self.vocabulary_size = len(self.tokenizer)

        if self.train_mode.startswith("train"):
            self.train_steps = trainer_utils.calc_train_steps(config, self.datamodule, self.n_gpus)
            print(f"Train steps: {self.train_steps}")

        sample_unique = config.get("n_unique_beams") is not None

        self.sampler = BeamSearchSampler(
            self.tokenizer,
            trainer_utils.instantiate_scorers(self.config.get("scorers")),
            util.DEFAULT_MAX_SEQ_LEN,
            device=device,
            data_device=data_device,
            sample_unique=sample_unique,
        )

        self.build_model(config)
        self.model.num_beams = config.n_beams
        if sample_unique:
            self.model.n_unique_beams = np.min(np.array([self.model.num_beams, config.n_unique_beams]))

        self.trainer = None
        if "trainer" in self.config:
            self.trainer = trainer_utils.build_trainer(config, self.n_gpus)

        self.model = self.model.to(device)
        return

    def encode(
        self,
        dataset: str = "full",
        dataloader: Optional[DataLoader] = None,
    ) -> List[torch.Tensor]:
        """
        Compute memory from transformer inputs.

        Args:
            dataset (str): (Which part of the dataset to use (["train", "val", "test",
                "full"]).)
            dataloader (DataLoader): (If None -> dataloader
                will be retrieved from self.datamodule)
        Returns:
            List[torch.Tensor]: Tranformer memory
        """

        self.model.to(self.device)
        self.model.eval()

        if dataloader is None:
            dataloader = self.get_dataloader(dataset)

        X_encoded = []
        for b_idx, batch in enumerate(dataloader):
            batch = self.on_device(batch)
            with torch.no_grad():
                batch_encoded = self.model.encode(batch).permute(
                    1, 0, 2
                )  # Return on shape [n_samples, n_tokens, max_seq_length]

            X_encoded.extend(batch_encoded)
        return X_encoded

    def decode(
        self,
        memory: torch.Tensor,
        memory_pad_mask: torch.Tensor,
        decoder_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Output token probabilities from a given decoder input

        Args:
            memory_input (torch.Tensor): tensor from encoded input of shape (src_len,
                batch_size, d_model)
            memory_pad_mask (torch.Tensor): bool tensor of memory padding mask of shape
                (src_len, batch_size)
            decoder_input (torch.Tensor): tensor of decoder token_ids of shape (tgt_len,
                batch_size)
        """
        self.model.to(self.device)
        self.model.eval()

        batch_input = {
            "memory_input": memory,
            "memory_pad_mask": memory_pad_mask.permute(1, 0),
            "decoder_input": decoder_input.permute(1, 0),
            "decoder_pad_mask": torch.zeros_like(decoder_input, dtype=bool).permute(1, 0),
        }
        with torch.no_grad():
            return self.model.decode(batch_input)

    def set_datamodule(
        self,
        datamodule: Optional[pl.LightningDataModule] = None,
        datamodule_type: Optional[ListConfig] = None,
    ) -> None:
        """
        Create a new datamodule by either supplying a datamodule (created elsewhere) or
        a pre-defined datamodule type as input.

        Args:
            datamodule (Optional[pl.LightningDataModule]): pytorchlightning datamodule
            datamodule_type (Optional[str]): The type of datamodule to build if no
                datamodule is given as input.
        """
        if datamodule is None and datamodule_type is not None:
            data_collection = DataCollection(self.config, self.tokenizer)
            self.datamodule = data_collection.get_datamodule(datamodule_type)
        elif datamodule is None:
            print("Did not initialize datamodule.")
            return
        else:
            self.datamodule = datamodule

        self.datamodule.setup()
        n_cpus = len(os.sched_getaffinity(0))
        if self.n_gpus > 0:
            n_workers = n_cpus // self.n_gpus
        else:
            n_workers = n_cpus
        self.datamodule._num_workers = n_workers
        print(f"Using {str(n_workers)} workers for data module.")
        return

    def fit(self) -> None:
        """
        Fit model to training data in self.datamodule and using parameters specified in
        the trainer object.
        """
        self.trainer.fit(self.model, datamodule=self.datamodule)
        return

    def parameters(self) -> Iterator:
        return self.model.parameters()

    def _random_initialization(
        self, args: Namespace, extra_args: Dict[str, Any], pad_token_idx: int
    ) -> Union[BARTModel, UnifiedModel]:
        """
        Constructing a model with randomly initialized weights.

        Args:
            args (Namespace): Grouped model arguments.
            extra_args (Dict[str, Any]): Extra arguments passed to the BARTModel.
            Will be saved as hparams by pytorchlightning.
            pad_token_idx: The index denoting padding in the vocabulary.
        """

        if self.train_mode.startswith("train"):
            total_steps = self.train_steps + 1
        else:
            total_steps = 0

        if self.model_type == "bart":
            model = BARTModel(
                self.sampler,
                pad_token_idx,
                self.vocabulary_size,
                args.d_model,
                args.n_layers,
                args.n_heads,
                args.d_feedforward,
                args.get("learning_rate"),
                DEFAULT_WEIGHT_DECAY,
                util.DEFAULT_ACTIVATION,
                total_steps,
                util.DEFAULT_MAX_SEQ_LEN,
                schedule=args.get("schedule"),
                dropout=util.DEFAULT_DROPOUT,
                warm_up_steps=args.get("warm_up_steps"),
                **extra_args,
            )
        elif self.model_type == "unified":
            model = UnifiedModel(
                self.sampler,
                pad_token_idx,
                self.vocabulary_size,
                args.d_model,
                args.n_layers,
                args.n_heads,
                args.d_feedforward,
                args.get("learning_rate"),
                DEFAULT_WEIGHT_DECAY,
                util.DEFAULT_ACTIVATION,
                total_steps,
                util.DEFAULT_MAX_SEQ_LEN,
                schedule=args.get("schedule"),
                dropout=util.DEFAULT_DROPOUT,
                warm_up_steps=args.get("warm_up_steps"),
                **extra_args,
            )
        else:
            raise ValueError(f"Unknown model type [bart, unified]: {self.model_type}")

        return model

    def _initialize_from_ckpt(
        self, args: Namespace, extra_args: Dict[str, Any], pad_token_idx: int
    ) -> Union[BARTModel, UnifiedModel]:
        """
        Constructing a model with weights from a ckpt-file.

        Args:
            args (Namespace): Grouped model arguments.
            extra_args (Dict[str, Any]): Extra arguments passed to the BARTModel.
            Will be saved as hparams by pytorchlightning.
            pad_token_idx: The index denoting padding in the vocabulary.
        """
        if self.train_mode == "training" or self.train_mode == "train":
            total_steps = self.train_steps + 1

        if self.model_type == "bart":
            if self.train_mode == "training" or self.train_mode == "train":
                if self.resume_training:
                    model = BARTModel.load_from_checkpoint(
                        self.model_path,
                        decode_sampler=self.sampler,
                        num_steps=total_steps,
                        pad_token_idx=pad_token_idx,
                        vocabulary_size=self.vocabulary_size,
                    )
                else:
                    model = BARTModel.load_from_checkpoint(
                        self.model_path,
                        decode_sampler=self.sampler,
                        pad_token_idx=pad_token_idx,
                        vocabulary_size=self.vocabulary_size,
                        num_steps=total_steps,
                        lr=args.learning_rate,
                        weight_decay=args.weight_decay,
                        schedule=args.schedule,
                        warm_up_steps=args.warm_up_steps,
                        **extra_args,
                    )
            elif (
                self.train_mode == "validation"
                or self.train_mode == "val"
                or self.train_mode == "test"
                or self.train_mode == "testing"
                or self.train_mode == "eval"
            ):
                model = BARTModel.load_from_checkpoint(self.model_path, decode_sampler=self.sampler)
                model.eval()
            else:
                raise ValueError(f"Unknown training mode: {self.train_mode}")
        elif self.model_type == "unified":
            if self.train_mode == "training" or self.train_mode == "train":
                if self.resume_training:
                    model = UnifiedModel.load_from_checkpoint(self.model_path, decode_sampler=self.sampler)
                    model.train()
                else:
                    model = UnifiedModel.load_from_checkpoint(
                        self.model_path,
                        decode_sampler=self.sampler,
                        pad_token_idx=pad_token_idx,
                        vocabulary_size=self.vocabulary_size,
                        num_steps=total_steps,
                        lr=args.learning_rate,
                        weight_decay=args.weight_decay,
                        schedule=args.schedule,
                        warm_up_steps=args.warm_up_steps,
                        **extra_args,
                    )
            elif (
                self.train_mode == "validation"
                or self.train_mode == "val"
                or self.train_mode == "test"
                or self.train_mode == "testing"
                or self.train_mode == "eval"
            ):
                model = UnifiedModel.load_from_checkpoint(self.model_path, decode_sampler=self.sampler)
                model.eval()
            else:
                raise ValueError(f"Unknown training mode: {self.train_mode}")
        else:
            raise ValueError(f"Unknown model type [bart, unified]: {self.model_type}")
        return model

    def build_model(self, args: Namespace) -> None:
        """
        Build transformer model, either
        1. By loading pre-trained model from checkpoint file, or
        2. Initializing new model with random weight initialization

        Args:
            args (Namespace): Grouped model arguments.
        """

        pad_token_idx = self.tokenizer["pad"]

        # These args don't affect the model directly but will be saved by lightning as hparams
        # Tensorboard doesn't like None so we need to convert to string
        train_tokens = "None" if self.train_tokens is None else self.train_tokens
        n_buckets = "None" if self.n_buckets is None else self.n_buckets

        if self.train_mode == "training" or self.train_mode == "train":
            extra_args = {
                "batch_size": self.datamodule.batch_size,
                "acc_batches": args.acc_batches,
                "epochs": args.n_epochs,
                "clip_grad": args.clip_grad,
                "augment": args.augmentation_strategy,
                "aug_prob": args.augmentation_probability,
                "train_tokens": train_tokens,
                "n_buckets": n_buckets,
                "limit_val_batches": args.limit_val_batches,
            }
        else:
            extra_args = {}

        # If no model is given, use random init
        if not self.model_path:
            self.model = self._random_initialization(args, extra_args, pad_token_idx)
        else:
            self.model = self._initialize_from_ckpt(args, extra_args, pad_token_idx)
        return

    def get_dataloader(self, dataset: str, datamodule: Optional[pl.LightningDataModule] = None) -> DataLoader:
        """
        Get the dataloader for a subset of the data from a specific datamodule.

        Args:
            dataset (str): One in ["full", "train", "val", "test"].
                Specifies which part of the data to return.
            datamodule (Optional[pl.LightningDataModule]): pytorchlightning datamodule.
                If None -> Will use self.datamodule.
        """
        if dataset not in ["full", "train", "val", "test"]:
            raise ValueError(f"Unknown dataset : {dataset}. Should be either 'full', 'train', 'val' or 'test'.")

        if datamodule is None:
            datamodule = self.datamodule

        dataloader = None
        if dataset == "full":
            dataloader = datamodule.full_dataloader()
        elif dataset == "train":
            dataloader = datamodule.train_dataloader()
        elif dataset == "val":
            dataloader = datamodule.val_dataloader()
        elif dataset == "test":
            dataloader = datamodule.test_dataloader()

        return dataloader

    @torch.no_grad()
    def log_likelihood(
        self,
        dataset: str = "full",
        dataloader: Optional[DataLoader] = None,
    ) -> List[float]:
        """
        Computing the likelihood of the encoder_input SMILES and decoder_input SMILES
        pairs.

        Args:
            dataset (str): Which part of the dataset to use (["train", "val", "test",
                "full"]).
            dataloader (Optional[DataLoader]): If None -> dataloader
                will be retrieved from self.datamodule.
        Returns:
            List[float]: List with log-likelihoods of each reactant/product pairs.
        """

        if dataloader is None:
            dataloader = self.get_dataloader(dataset)

        self.model.to(self.device)
        self.model.eval()

        log_likelihoods = []
        for batch in dataloader:
            batch = self.on_device(batch)
            output = self.model.forward(batch)
            log_probabilities = self.model.generator(output["model_output"])

            target_ids_lst = batch["decoder_input"].permute(1, 0)

            for target_ids, log_prob in zip(target_ids_lst[:, 1::], log_probabilities.permute(1, 0, 2)):
                llhs = 0.0
                for i_token, token in enumerate(target_ids):
                    llhs += log_prob[i_token, token].item()
                    break_condition = token == self.tokenizer["end"] or token == self.tokenizer["pad"]
                    if break_condition:
                        break

                log_likelihoods.append(llhs)
        return log_likelihoods

    def on_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move data in "batch" to the current model device.

        Args:
            batch (Dict[str, Any]): batch input data to model.
        Returns:
            Dict[str, Any]: batch data on current device.
        """
        device_batch = {
            key: val.to(self.device) if isinstance(val, torch.Tensor) else val for key, val in batch.items()
        }
        return device_batch

    def predict(
        self,
        dataset: str = "full",
        dataloader: Optional[DataLoader] = None,
        return_tokenized: bool = False,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Predict SMILES output given dataloader, specified by 'dataset'.
        Args:
            dataset: Which part of the dataset to use (["train", "val", "test",
                "full"]).
            dataloader: If None -> dataloader
                will be retrieved from self.datamodule.
            return_tokenized: Whether to return the tokenized beam search
                solutions instead of strings.
        Returns:
            (sampled_smiles List[np.ndarray], log_lhs List[np.ndarray], target_smiles List[np.ndarray])
        """

        if dataloader is None:
            dataloader = self.get_dataloader(dataset)

        self.model.to(self.device)
        self.model.eval()

        sampled_smiles = []
        log_lhs = []
        target_smiles = []
        for batch in dataloader:
            batch = self.on_device(batch)
            with torch.no_grad():
                smiles_batch, log_lhs_batch = self.model.sample_molecules(
                    batch, sampling_alg="beam", return_tokenized=return_tokenized
                )
                if self.model.sampler.sample_unique:
                    smiles_batch = self.sampler.smiles_unique
                    log_lhs_batch = self.sampler.log_lhs_unique

            sampled_smiles.extend(smiles_batch)
            log_lhs.extend(log_lhs_batch)
            target_smiles.extend(batch["target_smiles"])

        return sampled_smiles, log_lhs, target_smiles

    def score_model(
        self,
        n_unique_beams: Optional[int] = None,
        dataset: str = "full",
        dataloader: Optional[DataLoader] = None,
        output_scores: Optional[str] = None,
        output_sampled_smiles: Optional[str] = None,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Score model performance on dataset in terms of accuracy (top-1 and top-K) and
        similarity of top-1 molecules. Also collects basic logging scores (loss, etc.).

        Args:
            n_unique_beams: Number of unique beams after canonicalizing sampled
                SMILES strings.
            dataset: Which part of the dataset to use (["train", "val", "test",
                "full"]).
            dataloader (DataLoader): If None -> dataloader will be
                retrieved from self.datamodule.
            output_scores: Path to output .csv file with model performance. If None ->
                Will not write DataFrame to file.
            output_sampled_smiles: Path to output .json file with sampled smiles.
                If None -> Will not write DataFrame to file.
        Returns:
            [pandas.DataFrame with calculated scores/metrics, pandas.DataFrame with
                sampled SMILES]
            or
            pandas.DataFrame with calculated scores/metrics
        """

        if output_scores and output_sampled_smiles:
            for callback in self.trainer.callbacks:
                if hasattr(callback, "set_output_files"):
                    callback.set_output_files(output_scores, output_sampled_smiles)

        if n_unique_beams is None and self.sampler.smiles_unique:
            n_unique_beams = self.model.num_beams
        self.model.n_unique_beams = n_unique_beams

        if dataloader is None:
            dataloader = self.get_dataloader(dataset)

        self.model.eval()
        self.model.to(self.device)

        for b_idx, batch in enumerate(dataloader):
            batch = self.on_device(batch)
            metrics = self.model.test_step(batch, b_idx)

            if self.model.sampler.sample_unique:
                sampled_smiles_unique = self.model.sampler.smiles_unique
                log_lhs_unique = self.model.sampler.log_lhs_unique

                # Get data of unique SMILES/solutions (keeping both non-unique
                # and unique metrics)
                metrics_unique = self.model.sampler.compute_sampling_metrics(
                    sampled_smiles_unique, metrics["target_smiles"], is_canonical=False
                )

                metrics_unique.update(
                    {
                        "sampled_molecules": sampled_smiles_unique,
                        "log_lhs": log_lhs_unique,
                    }
                )

                drop_cols = [
                    "fraction_invalid",
                    "fraction_unique",
                    "top1_tanimoto_similarity",
                ]
                metrics_unique = {f"{key}(unique)": val for key, val in metrics_unique.items() if key not in drop_cols}
                metrics.update(metrics_unique)

            for callback in self.trainer.callbacks:
                if not isinstance(callback, pl.callbacks.progress.ProgressBar):
                    callback.on_test_batch_end(self.trainer, self.model, metrics, batch, b_idx, 0)

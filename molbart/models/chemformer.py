import os
from argparse import Namespace
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

import molbart.modules.util as util
from molbart.models import BARTModel, UnifiedModel
from molbart.modules.data.base import SimpleReactionListDataModule
from molbart.modules.decoder import BeamSearchSampler
from molbart.modules.tokenizer import ChemformerTokenizer

DEFAULT_WEIGHT_DECAY = 0


class Chemformer:
    """
    Class for building (synthesis) Chemformer model, fine-tuning seq-seq model,
    and predicting/scoring model.
    """

    def __init__(
        self,
        vocabulary_path: str,
        model_args: Namespace,
        data_args: Namespace,
        model_path: Optional[str] = None,
        n_gpus: int = 1,
        n_beams: int = 1,
        n_unique_beams: Optional[int] = None,
        datamodule_type: str = "seq2seq",
        train_mode: str = "training",
        device: str = "cuda",
        data_device: str = "cuda",
        build_trainer: bool = False,
        sample_unique: bool = True,
        resume_training: bool = False,
    ) -> None:
        """
        Args:
            vocabulary_path (str): path to bart_vocabulary.
            model_args (Namespace): Arguments for building the chemformer model.
            data_args (Namespace): Arguments for building torch datamodule.
            model_path (Optional[str]): Path to model weights.
            n_gpus (int): Number of GPUs to use.
            n_beams (int): Number of beams in beam search.
            n_unique_beams (Optional[int]): Restrict number of unique beam search solutions.
                If None => return all unique solutions.
            datamodule_type (str): The type of datamodule to build.
            train_model (str): Whether to train the model ("training") or use
                model for evaluations ("eval").
            sampler (str): Which beam search sampler to use ("optimized" => GPU
                optimized beam search).
            device (str): Which device to run model and beam search on ("cuda" / "cpu").
            data_device (str): device used for handling the data in optimized beam search.
                If memory issues, could help to set data_device="cpu"
            build_trainer (bool): If True, build a trainer which can be used for
                fine-tuning the model.
            sample_unique (bool): Whether to return unique beam search solutions from the
                optimized beam search.
            resume_training (bool): Whether to continue training from the supplied
                .ckpt file.
        """

        self.train_mode = train_mode
        self.resume_training = resume_training
        if resume_training:
            print("Resuming training.")

        if n_gpus < 1:
            device = "cpu"
            data_device = "cpu"

        self.device = device

        self.tokenizer = ChemformerTokenizer(filename=vocabulary_path)
        self.train_tokens = data_args.train_tokens
        self.n_buckets = data_args.n_buckets

        self.model_type = model_args.model_type
        self.model_path = model_path
        if self.model_path is None:
            self.model_path = "None"

        self.data_args = data_args
        self.n_gpus = n_gpus
        self.is_data_setup = False
        self.set_datamodule(datamodule_type=datamodule_type)

        print("Vocabulary_size: " + str(len(self.tokenizer)))
        self.vocabulary_size = len(self.tokenizer)

        if self.train_mode == "training" or self.train_mode == "train":
            self.train_steps = util.calc_train_steps(
                model_args, self.datamodule, n_gpus
            )
            print(f"Train steps: {self.train_steps}")

        self.sampler = BeamSearchSampler(
            self.tokenizer,
            util.DEFAULT_MAX_SEQ_LEN,
            device=device,
            data_device=data_device,
            sample_unique=sample_unique,
        )

        print("Building model.")
        self.build_model(model_args)
        self.model.num_beams = n_beams
        if n_unique_beams is None:
            n_unique_beams = n_beams
        self.model.n_unique_beams = np.min(
            np.array([self.model.num_beams, n_unique_beams])
        )

        if (
            self.train_mode == "training" or self.train_mode == "train"
        ) or build_trainer:
            print("Building trainer.")
            self.trainer = util.build_trainer(
                model_args, n_gpus, data_args.dataset_type
            )
            if not self.resume_training:
                self.out_directory = self._set_out_directory()
            print("Model initialization done.")

        self.model.to(device)
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
            "decoder_pad_mask": torch.zeros_like(decoder_input, dtype=bool).permute(
                1, 0
            ),
        }
        with torch.no_grad():
            return self.model.decode(batch_input)

    def _set_out_directory(self) -> str:
        """
        Defining the output directory for fine-tuned model weights. Will create a new
        version_[X] directory to not overwrite previous training sessions.

        Returns:
            str: Output directory
        """
        if self.trainer.logger is not None:
            if self.trainer.weights_save_path != self.trainer.default_root_dir:
                save_dir = self.trainer.weights_save_path
            else:
                save_dir = self.trainer.logger.save_dir or self.trainer.default_root_dir

            version = (
                self.trainer.logger.version
                if isinstance(self.trainer.logger.version, str)
                else f"version_{self.trainer.logger.version}"
            )
            version, name = self.trainer.training_type_plugin.broadcast(
                (version, self.trainer.logger.name)
            )
            out_dir = os.path.join(save_dir, str(name), version)
        else:
            out_dir = self.trainer.weights_save_path
        return out_dir

    def set_datamodule(
        self,
        datamodule: Optional[pl.LightningDataModule] = None,
        datamodule_type: Optional[str] = None,
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
            print("Datamodule type: " + str(datamodule_type))
            if datamodule_type == "seq2seq":
                self.datamodule = util.build_seq2seq_datamodule(
                    self.data_args, self.tokenizer, self.data_args.forward_prediction
                )
            elif datamodule_type == "simple_reaction_list":
                self.datamodule = SimpleReactionListDataModule(
                    dataset_path=self.data_args.reactants_path,
                    tokenizer=self.tokenizer,
                    batch_size=self.data_args.batch_size,
                    max_seq_len=util.DEFAULT_MAX_SEQ_LEN,
                    reverse=not self.data_args.forward_prediction,
                )
            else:
                raise ValueError(f"Unknown datamodule type: {datamodule_type}")
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

        total_steps = self.train_steps + 1

        if self.model_type == "bart":
            model = BARTModel(
                self.sampler,
                pad_token_idx,
                self.vocabulary_size,
                args.d_model,
                args.n_layers,
                args.n_heads,
                args.d_feedforward,
                args.learning_rate,
                DEFAULT_WEIGHT_DECAY,
                util.DEFAULT_ACTIVATION,
                total_steps,
                util.DEFAULT_MAX_SEQ_LEN,
                schedule=args.schedule,
                dropout=util.DEFAULT_DROPOUT,
                warm_up_steps=args.warm_up_steps,
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
                args.learning_rate,
                DEFAULT_WEIGHT_DECAY,
                util.DEFAULT_ACTIVATION,
                total_steps,
                util.DEFAULT_MAX_SEQ_LEN,
                schedule=args.schedule,
                dropout=util.DEFAULT_DROPOUT,
                warm_up_steps=args.warm_up_steps,
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
                model = BARTModel.load_from_checkpoint(
                    self.model_path, decode_sampler=self.sampler
                )
                model.eval()
            else:
                raise ValueError(f"Unknown training mode: {self.train_mode}")
        elif self.model_type == "unified":
            if self.train_mode == "training" or self.train_mode == "train":
                if self.resume_training:
                    model = UnifiedModel.load_from_checkpoint(
                        self.model_path, decode_sampler=self.sampler
                    )
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
                model = UnifiedModel.load_from_checkpoint(
                    self.model_path, decode_sampler=self.sampler
                )
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
        if self.model_path in ["none", "None"]:
            self.model = self._random_initialization(args, extra_args, pad_token_idx)
        else:
            self.model = self._initialize_from_ckpt(args, extra_args, pad_token_idx)
        return

    def get_logged_data(self) -> pd.DataFrame:
        """
        Build dataframe from logged metrics, which are stored in self.trainer.callbacks[2]

        Returns:
            pd.DataFrame: DataFrame with training-loss, validation-loss and molecular
            accuracy of the validation set.
        """
        epochs = self.trainer.callbacks[2].epochs
        train_loss = self.trainer.callbacks[2].train_loss
        val_loss = self.trainer.callbacks[2].val_loss
        val_token_acc = self.trainer.callbacks[2].val_token_acc
        perplexity = self.trainer.callbacks[2].perplexity
        mol_acc = self.trainer.callbacks[2].mol_acc

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

    def save_logged_data(self) -> None:
        """
        Retrieve and write data (model validation) logged during training.
        """
        metrics_df = self.get_logged_data()
        if self.resume_training:
            self._set_out_directory()
        outfile = self.out_directory + "/logged_train_metrics.csv"
        metrics_df.to_csv(outfile, sep="\t", index=False)
        print("Logged training/validation set loss written to: " + outfile)
        return

    def get_dataloader(
        self, dataset: str, datamodule: Optional[pl.LightningDataModule] = None
    ) -> DataLoader:
        """
        Get the dataloader for a subset of the data from a specific datamodule.

        Args:
            dataset (str): One in ["full", "train", "val", "test"].
                Specifies which part of the data to return.
            datamodule (Optional[pl.LightningDataModule]): pytorchlightning datamodule.
                If None -> Will use self.datamodule.
        """
        if dataset not in ["full", "train", "val", "test"]:
            raise ValueError(
                f"Unknown dataset : {dataset}. Should be either 'full', 'train', 'val' or 'test'."
            )

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

    def on_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move data in "batch" to the current model device.

        Args:
            batch (Dict[str, Any]): batch input data to model.
        Returns:
            Dict[str, Any]: batch data on current device.
        """
        device_batch = {
            key: val.to(self.device) if isinstance(val, torch.Tensor) else val
            for key, val in batch.items()
        }
        return device_batch

    def predict(
        self,
        dataset: str = "full",
        dataloader: Optional[DataLoader] = None,
        return_tokenized: bool = False,
        i_chunk: int = 0,
        n_chunks: int = 1,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Predict SMILES output given dataloader, specified by 'dataset'.
        Args:
            dataset (str): Which part of the dataset to use (["train", "val", "test",
                "full"]).
            dataloader (Optional[DataLoader]): If None -> dataloader
                will be retrieved from self.datamodule.
            return_tokenized (bool): Whether to return the tokenized beam search
                solutions instead of strings.
        Returns:
            (sampled_smiles List[np.ndarray], log_lhs List[np.ndarray], target_smiles List[np.ndarray])
        """
        if dataloader is None:
            dataloader = self.get_dataloader(dataset)

        # Divide batches into chunks of batches
        n_batches_in_chunk = int(len(dataloader) / float(n_chunks))
        start_batch_idx = i_chunk * n_batches_in_chunk

        self.model.to(self.device)
        self.model.eval()

        sampled_smiles = []
        log_lhs = []
        target_smiles = []
        for b_idx, batch in enumerate(dataloader):
            if n_chunks > 1:
                if b_idx < start_batch_idx:
                    continue

                if i_chunk != n_chunks - 1:
                    if (
                        i_chunk != n_chunks - 1
                        and b_idx == start_batch_idx + n_batches_in_chunk
                    ):
                        break

            batch = self.on_device(batch)
            with torch.no_grad():
                smiles_batch, log_lhs_batch = self.model.sample_molecules(
                    batch, sampling_alg="beam", return_tokenized=return_tokenized
                )

            sampled_smiles.extend(smiles_batch)
            log_lhs.extend(log_lhs_batch)
            target_smiles.extend(batch["target_smiles"])

        return sampled_smiles, log_lhs, target_smiles

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

            for target_ids, log_prob in zip(
                target_ids_lst[:, 1::], log_probabilities.permute(1, 0, 2)
            ):
                llhs = 0.0
                for i_token, token in enumerate(target_ids):
                    llhs += log_prob[i_token, token].item()
                    break_condition = (
                        token == self.tokenizer["end"] or token == self.tokenizer["pad"]
                    )
                    if break_condition:
                        break

                log_likelihoods.append(llhs)
        return log_likelihoods

    def score_model(
        self,
        n_unique_beams: Optional[int] = None,
        dataset: str = "full",
        dataloader: Optional[DataLoader] = None,
        outfile: Optional[str] = None,
        save_sampled_smiles: bool = True,
        outfile_sampled_smiles: Optional[str] = None,
        compute_similarity: bool = True,
        sample_molecules: bool = True,
        i_chunk: int = 0,
        n_chunks: int = 1,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Score model performance on dataset in terms of accuracy (top-1 and top-K) and
        similarity of top-1 molecules. Also collects basic logging scores (loss, etc.).

        Args:
            n_unique_beams (int): Number of unique beams after canonicalizing sampled
                SMILES strings.
            dataset (str): Which part of the dataset to use (["train", "val", "test",
                "full"]).
            dataloader (DataLoader): If None -> dataloader will be
                retrieved from self.datamodule.
            outfile (str): Path to output .csv file with model performance. If None ->
                Will not write DataFrame to file.
            save_sampled_smiles (bool): Indicating whether to write the sampled
                smiles to file.
            outfile_sampled_smiles (str): Path to output .json file with sampled smiles.
                If None -> Will not write DataFrame to file.
            compute_similarity (bool): Whether to compute molecular similarity between
                top-1 prediction and target smiles.
            sample_molecules (bool): Whether to sample molecules. If False, will not
                run beam search and will only compute basic logging metrics.
        Returns:
            [pandas.DataFrame with calculated scores/metrics, pandas.DataFrame with
                sampled SMILES]
            or
            pandas.DataFrame with calculated scores/metrics
        """
        if n_unique_beams is None:
            n_unique_beams = self.model.num_beams
        self.model.n_unique_beams = n_unique_beams

        if dataloader is None:
            dataloader = self.get_dataloader(dataset)

        # Divide batches into chunks of batches
        n_batches_in_chunk = int(len(dataloader) / float(n_chunks))
        start_batch_idx = i_chunk * n_batches_in_chunk

        metrics_df_out = None
        mol_df_out = None
        self.model.eval()
        self.model.to(self.device)

        for b_idx, batch in enumerate(dataloader):
            if n_chunks > 1:
                if b_idx < start_batch_idx:
                    continue

                if i_chunk != n_chunks - 1:
                    if (
                        i_chunk != n_chunks - 1
                        and b_idx == start_batch_idx + n_batches_in_chunk
                    ):
                        break

            with torch.no_grad():
                print("Scoring batch: " + str(b_idx + 1) + "/" + str(len(dataloader)))
                batch = self.on_device(batch)
                model_output = self.model.forward(batch)
                target_smiles = batch["target_smiles"]

                loss = self.model._calc_loss(batch, model_output)
                token_accuracy = self.model._calc_token_acc(batch, model_output)
                perplexity = self.model._calc_perplexity(batch, model_output)
                if sample_molecules:
                    sampled_smiles, log_lhs = self.model.sample_molecules(
                        batch, sampling_alg="beam"
                    )
                    sampled_smiles_unique = self.model.sampler.smiles_unique
                    log_lhs_unique = self.model.sampler.log_lhs_unique

                    metrics = pd.DataFrame.from_dict(
                        self.model.sampler.compute_sampling_metrics(
                            sampled_smiles,
                            target_smiles,
                            is_canonical=False,
                            compute_similarity=compute_similarity,
                        )
                    )

                    if self.model.num_beams > 1:
                        # Get data of unique SMILES/solutions (keeping both non-unique
                        # and unique metrics)
                        metrics_unique = pd.DataFrame.from_dict(
                            self.model.sampler.compute_sampling_metrics(
                                sampled_smiles_unique, target_smiles, is_canonical=True
                            )
                        )
                        drop_cols = ["accuracy", "fraction_invalid", "fraction_unique"]
                        if compute_similarity:
                            drop_cols.append("similarity")
                        metrics_unique.drop(columns=drop_cols, inplace=True)
                        for col in metrics_unique.columns.values:
                            metrics_unique.rename(
                                columns={col: col + "(unique)"}, inplace=True
                            )

                    # Putting data on the same format as sampled_smiles_unique
                    sampled_smiles = [np.array(smi) for smi in sampled_smiles]
                    target_smiles = [np.array([smi]) for smi in target_smiles]
                    log_lhs = [np.array(ll) for ll in log_lhs]
                    if compute_similarity:
                        similarity = [
                            np.array(sim) for sim in metrics["similarity"].values[0]
                        ]
                else:
                    metrics = pd.DataFrame.from_dict({})

                # Adding metrics to dataframe
                metrics["batch_idx"] = b_idx
                metrics["loss"] = loss.item()
                metrics["token_accuracy"] = np.array([token_accuracy.cpu()])
                metrics["perplexity"] = np.array([perplexity.cpu()])

                if sample_molecules:
                    tmp_x = np.zeros((1, 1), dtype="object")
                    tmp_x[0] = [log_lhs]
                    metrics["log_lhs"] = np.copy(tmp_x)

                    if self.model.num_beams > 1:
                        # Putting log_lhs_unique on the same format as
                        # sampled_smiles_unique
                        log_lhs_unique = [np.array(ll) for ll in log_lhs_unique]
                        tmp_x[0] = [log_lhs_unique]
                        metrics_unique["log_lhs(unique)"] = np.copy(tmp_x)

                    if compute_similarity:
                        tmp_x[0] = [similarity]
                        metrics["similarity"] = np.copy(tmp_x)

                    tmp_x[0] = [target_smiles]
                    mol_df = pd.DataFrame(np.copy(tmp_x), columns=["target_smiles"])
                    tmp_x[0] = [sampled_smiles]
                    mol_df["sampled_molecules"] = np.copy(tmp_x)
                    tmp_x[0] = [sampled_smiles_unique]
                    mol_df["sampled_molecules(unique)"] = np.copy(tmp_x)

                    if self.model.num_beams > 1:
                        metrics = pd.concat([metrics, metrics_unique], axis=1)

            if metrics_df_out is None:
                metrics_df_out = metrics
                if sample_molecules and save_sampled_smiles:
                    mol_df_out = mol_df
            else:
                metrics_df_out = pd.concat([metrics_df_out, metrics], axis=0)
                if sample_molecules and save_sampled_smiles:
                    mol_df_out = pd.concat([mol_df_out, mol_df], axis=0)

            # Save results for each batch
            if outfile is not None:
                if outfile.endswith(".json"):
                    metrics_df_out.to_json(outfile, orient="table")
                else:
                    metrics_df_out.to_csv(outfile, sep="\t", index=False)
                print("Model scores written to: " + outfile)

                if save_sampled_smiles:
                    if outfile_sampled_smiles is None:
                        outfile_sampled_smiles = (
                            outfile[0:-4] + "_SMILES" + outfile[-4::]
                        )

                    mol_df_out.to_json(outfile_sampled_smiles, orient="table")

                    print("Sampled SMILES written to: " + outfile_sampled_smiles)

        if save_sampled_smiles:
            return metrics_df_out, mol_df_out
        else:
            return metrics_df_out

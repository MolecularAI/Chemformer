""" Module containing a class for the DataSet used as well as base-classes for DataModules"""
import functools
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from pysmilesutils.augment import SMILESAugmenter
from pysmilesutils.datautils import TokenSampler
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from pysmilesutils.datautils import ChunkBatchSampler

from molbart.data.util import BatchEncoder, build_attention_mask, build_target_mask
from molbart.utils.tokenizers import ChemformerTokenizer, TokensMasker


class ChemistryDataset(Dataset):
    """
    Generic dataset that consists of a dictionary, where each
    value is an equal-sized sequence of data.

    Such a dictionary can be seen as a dictionary representation
    of a pandas DataFrame, but indexing a DataFrame object is slow
    and therefore the data is stored as a dictionary.

    One can obtain the length of the dataset using the `len` operator
    and individual "rows" of the data can be accessed by indexing

    .. code-block::

        row = dataset[10]

    the `row` variable returned is also a dictionary, but each value
    is a single value. The keys are the same as in the original dictionary.

    As such a batch of such rows sampled by a DataLoader is a list of dictionaries.
    And one can obtain invidiual lists with e.g.

    .. code-block::

        molecules = [item["molecules"] for item in batch]

    :param data: the dataset
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data
        if self._data:
            key_zero = list(self._data.keys())[0]
            self._len = len(self._data[key_zero])
        else:
            self._len = 0

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, item: int) -> Dict[str, Any]:
        return {key: values[item] for key, values in self._data.items()}

    @property
    def seq_lengths(self) -> List[int]:
        """Return the sequence lengths data if such data exists"""
        if "seq_lengths" in self._data:
            return self._data["seq_lengths"]
        if "lengths" in self._data:
            return self._data["lengths"]
        raise KeyError("This dataset does not store any sequence lengths")


class _AbsDataModule(pl.LightningDataModule):
    """Base class for all DataModules"""

    def __init__(
        self,
        dataset_path: str,
        tokenizer: ChemformerTokenizer,
        batch_size: int,
        max_seq_len: int,
        train_token_batch_size: int = None,
        num_buckets: int = None,
        val_idxs: Sequence[int] = None,
        test_idxs: Sequence[int] = None,
        train_idxs: Sequence[int] = None,
        train_set_rest: bool = True,
        split_perc: float = 0.2,
        pin_memory: bool = True,
        unified_model: bool = False,
        i_chunk: int = 0,
        n_chunks: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()

        if val_idxs is not None and test_idxs is not None:
            idxs_intersect = set(val_idxs).intersection(set(test_idxs))
            if len(idxs_intersect) > 0:
                raise ValueError("Val idxs and test idxs overlap")

        if train_token_batch_size is not None and num_buckets is not None:
            print(
                f"""Training with approx. {train_token_batch_size} tokens per batch"""
                f""" and {num_buckets} buckets in the sampler."""
            )
        else:
            print(f"Using a batch size of {str(batch_size)}.")

        self.dataset_path = dataset_path
        self.tokenizer = tokenizer

        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.train_token_batch_size = train_token_batch_size
        self.num_buckets = num_buckets
        self.val_idxs = val_idxs
        self.test_idxs = test_idxs
        self.train_idxs = train_idxs
        self.train_set_rest = train_set_rest
        self.split_perc = split_perc
        self.pin_memory = pin_memory
        self.unified_model = unified_model

        self._num_workers = len(os.sched_getaffinity(0))

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.i_chunk = i_chunk
        self.n_chunks = n_chunks

        if self.n_chunks > 1:
            print("Using chunk of data:")
            print(f"- i_chunk: {i_chunk}, n_chunks: {n_chunks}")

        self._all_data: Dict[str, Any] = {}

    def train_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the training set"""
        if self.train_token_batch_size is None:
            if self.n_chunks > 1:
                # Should only be used for inference / postprocessing
                dataloader = self._create_chunk_dataloader(self.train_dataset, self._collate)
                return dataloader

            dataloader = self._create_basic_dataloader(self.train_dataset, self._collate, shuffle=True)
            return dataloader

        sampler = TokenSampler(
            self.num_buckets,
            self.train_dataset.seq_lengths,
            self.train_token_batch_size,
            shuffle=True,
        )
        dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=self._num_workers,
            collate_fn=self._collate,
            pin_memory=self.pin_memory,
        )
        return dataloader

    def val_dataloader(self):
        """Returns the DataLoader for the validation set"""
        if self.n_chunks > 1:
            dataloader = self._create_chunk_dataloader(self.val_dataset, functools.partial(self._collate, train=False))
            return dataloader

        dataloader = self._create_basic_dataloader(self.val_dataset, functools.partial(self._collate, train=False))
        return dataloader

    def test_dataloader(self):
        """Returns the DataLoader for the test set"""
        if self.n_chunks > 1:
            dataloader = self._create_chunk_dataloader(self.test_dataset, functools.partial(self._collate, train=False))
            return dataloader

        dataloader = self._create_basic_dataloader(self.test_dataset, functools.partial(self._collate, train=False))
        return dataloader

    def full_dataloader(self, train=False):
        """Returns the DataLoader for the test set"""
        if self.n_chunks > 1:
            dataloader = self._create_chunk_dataloader(
                ChemistryDataset(self._all_data), functools.partial(self._collate, train=train)
            )
            return dataloader

        dataloader = self._create_basic_dataloader(
            ChemistryDataset(self._all_data), functools.partial(self._collate, train=train)
        )
        return dataloader

    def setup(self, stage=None):
        """Load and split the dataset"""
        self._load_all_data()
        self._split_dataset()

    def _all_data_len(self) -> int:
        return len(ChemistryDataset(self._all_data))

    def _build_attention_mask(self, enc_length: int, dec_length: int) -> torch.Tensor:
        return build_attention_mask(enc_length, dec_length)

    def _collate(self, batch: List[Dict[str, Any]], train: bool = True) -> Dict[str, Any]:
        (
            encoder_ids,
            encoder_mask,
            decoder_ids,
            decoder_mask,
            smiles,
        ) = self._transform_batch(batch, train)
        if self.unified_model:
            return self._make_unified_model_batch(encoder_ids, encoder_mask, decoder_ids, decoder_mask, smiles)
        return {
            "encoder_input": encoder_ids,
            "encoder_pad_mask": encoder_mask,
            "decoder_input": decoder_ids[:-1, :],
            "decoder_pad_mask": decoder_mask[:-1, :],
            "target": decoder_ids.clone()[1:, :],
            "target_mask": decoder_mask.clone()[1:, :],
            "target_smiles": smiles,
        }

    def _create_chunk_dataloader(self, dataset, collate_fn):
        sampler = SequentialSampler(dataset)
        batch_sampler = ChunkBatchSampler(
            sampler=sampler, batch_size=self.batch_size, drop_last=False, i_chunk=self.i_chunk, n_chunks=self.n_chunks
        )

        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self._num_workers,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
        )
        return dataloader

    def _create_basic_dataloader(self, dataset, collate_fn, shuffle=False) -> DataLoader:
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self._num_workers,
            collate_fn=collate_fn,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
        )
        return dataloader

    def _load_all_data(self) -> None:
        raise NotImplementedError("Data loading is not implemented in base class")

    def _make_random_split_indices(self) -> None:
        dataset_len = self._all_data_len()
        val_len = round(dataset_len * self.split_perc)
        test_len = round(dataset_len * self.split_perc)
        all_idxs = range(dataset_len)
        idxs = random.sample(all_idxs, val_len + test_len)
        self.val_idxs = idxs[:val_len]
        self.test_idxs = idxs[val_len:]
        self.train_idxs = [idx for idx in all_idxs if idx not in idxs]

    def _make_unified_model_batch(
        self,
        encoder_ids: torch.Tensor,
        encoder_mask: torch.Tensor,
        decoder_ids: torch.Tensor,
        decoder_mask: torch.Tensor,
        smiles: List[str],
    ) -> Dict[str, Any]:
        decoder_ids = decoder_ids[1:]
        decoder_mask = decoder_mask[1:]

        enc_length, batch_size = tuple(encoder_ids.shape)
        dec_length, _ = tuple(decoder_ids[:-1, :].shape)
        att_mask = self._build_attention_mask(enc_length, dec_length)

        target = torch.cat((encoder_ids.clone()[:-1, :], decoder_ids.clone()), dim=0)
        target_mask = build_target_mask(enc_length, dec_length, batch_size)
        target_mask = target_mask + (torch.cat((encoder_mask[:-1, :], decoder_mask), dim=0))

        return {
            "encoder_input": encoder_ids,
            "encoder_pad_mask": encoder_mask,
            "decoder_input": decoder_ids[:-1, :],
            "decoder_pad_mask": decoder_mask[:-1, :],
            "attention_mask": att_mask,
            "target": target,
            "target_mask": target_mask,
            "target_smiles": smiles,
        }

    def _set_split_indices_from_dataframe(self, df: pd.DataFrame) -> None:
        # Don't set idx if they were provided as input to the class
        if self.val_idxs is not None or self.test_idxs is not None or self.train_idxs is not None:
            return

        val_idxs = df.query("set in ['val','valid','validation']").index.tolist()
        train_idxs = df.query("set in ['train','Train']").index.tolist()
        test_idxs = df.index[df["set"] == "test"].tolist()
        idxs_intersect = set(val_idxs).intersection(set(test_idxs))
        if len(idxs_intersect) > 0:
            raise ValueError("Val idxs and test idxs overlap")

        self.val_idxs = val_idxs
        self.test_idxs = test_idxs
        self.train_idxs = train_idxs

    def _split_dataset(self) -> None:
        def _subsample_data(indices):
            data = defaultdict(list)
            for idx in indices:
                for key in self._all_data.keys():
                    data[key].append(self._all_data[key][idx])
            return dict(data)

        if self.val_idxs is None and self.test_idxs is None:
            self._make_random_split_indices()
        elif self.val_idxs is None:
            self.val_idxs = []
        elif self.test_idxs is None:
            self.test_idxs = []

        self.val_dataset = ChemistryDataset(_subsample_data(self.val_idxs))
        self.test_dataset = ChemistryDataset(_subsample_data(self.test_idxs))

        if self.train_set_rest:
            # Below assumes all that is not test and val is train if not specified.
            all_idxs = set(range(self._all_data_len()))
            self.train_idxs = all_idxs - set(self.val_idxs).union(set(self.test_idxs))

        if self.train_idxs is None:
            self.train_idxs = []
        self.train_dataset = ChemistryDataset(_subsample_data(self.train_idxs))

    def _transform_batch(
        self, batch: List[Dict[str, Any]], train: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        raise NotImplementedError("Batch transformation is not implemented in base class")


class MoleculeListDataModule(_AbsDataModule):
    """
    DataModule that is used for sampling molecules. Can be
    used as base class for other DataModules that samples molecules.

    The molecules are read from a text-file containing SMILES strings,
    one on each row

    The `task` argument can be:
        * mask - the molecule tokens of the encoder are masked
        * aug - the molecules of the decoder are augmented
        * aug_mask - a combination of the above

    :param task: the model task, can be "mask", "aug" or "aug_mask"
    :param augment: if True, will augment the SMILES
    :param masker: the masker to use when `task` is "mask" or "aug_mask"
    :param dataset_path: the path to the dataset on disc
    :param tokenizer: the tokenizer to use
    :param batch_size: the batch size to use
    :param max_seq_len: the maximum allowed sequence length
    :param train_token_batch_size: if given, a `TokenSampler` is used
    :param num_buckets: the number of buckets for the `TokenSampler`
    :param val_idxs: if given, selects the validation set
    :param test_idxs: if given, selects the test set
    :param split_perc: determines the percentage of data that goes into validation and test sets
    :param pin_memory: if True, pins the memory of the DataLoader
    :param unified_model: if True, collate batches for unified model, not BART
    """

    def __init__(
        self,
        task: str = "mask",
        augment_prob: float = 0.0,
        masker: TokensMasker = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if "mask" in task and TokensMasker is None:
            raise ValueError(f"Need to provide a masker with task = {task}")

        self._augmenter = SMILESAugmenter(augment_prob=augment_prob)
        self._encoder = BatchEncoder(tokenizer=self.tokenizer, masker=masker, max_seq_len=self.max_seq_len)
        self.task = task
        self.augment = augment_prob > 0.0

    def _augment_batch(self, batch: List[str]) -> Tuple[List[str], List[str]]:
        if self.augment:
            encoder_smiles = self._augmenter(batch)
        else:
            encoder_smiles = batch[:]

        if "aug" in self.task:
            decoder_smiles = self._augmenter(encoder_smiles)
        else:
            decoder_smiles = encoder_smiles[:]
        return encoder_smiles, decoder_smiles

    def _load_all_data(self):
        with open(self.dataset_path, "r") as fileobj:
            self._all_data = {"smiles": fileobj.read().splitlines()}

    def _transform_batch(
        self, batch: List[Dict[str, Any]], train: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        smiles = [item["smiles"] for item in batch]
        encoder_smiles, decoder_smiles = self._augment_batch(smiles)
        encoder_ids, encoder_mask = self._encoder(
            encoder_smiles, mask="mask" in self.task, add_sep_token=self.unified_model
        )
        decoder_ids, decoder_mask = self._encoder(decoder_smiles, mask=False)

        # Ensure that the canonical form is used for evaluation
        dec_mols = [Chem.MolFromSmiles(smi) for smi in decoder_smiles]
        canon_targets = [Chem.MolToSmiles(mol) for mol in dec_mols]

        return encoder_ids, encoder_mask, decoder_ids, decoder_mask, canon_targets


class ReactionListDataModule(_AbsDataModule):
    """
    DataModule that is used for sampling reactions. It also serves
    as the base class for other DataModules that samples sequences
    to sequence data.

    The reactions are read from a text-file containing reactions
    SMILES strings, one on each row.
    If only sinlge molecules are provided in the text-file, the
    product and reactants are intepreted to be equal.

    :param augment_prob: the probability of augmenting the sequences in training
    :param reverse: if True, will return the encoder data as the decoder data and vice versa
    :param dataset_path: the path to the dataset on disc
    :param tokenizer: the tokenizer to use
    :param batch_size: the batch size to use
    :param max_seq_len: the maximum allowed sequence length
    :param train_token_batch_size: if given, a `TokenSampler` is used
    :param num_buckets: the number of buckets for the `TokenSampler`
    :param val_idxs: if given, selects the validation set
    :param test_idxs: if given, selects the test set
    :param split_perc: determines the percentage of data that goes into validation and test sets
    :param pin_memory: if True, pins the memory of the DataLoader
    :param unified_model: if True, collate batches for unified model, not BART
    """

    def __init__(self, augment_prob: float = 0.0, reverse: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._batch_augmenter = SMILESAugmenter(augment_prob=augment_prob)
        self._encoder = BatchEncoder(tokenizer=self.tokenizer, masker=None, max_seq_len=self.max_seq_len)
        self.reverse = reverse

    def _build_attention_mask(self, enc_length: int, dec_length: int) -> torch.Tensor:
        return build_attention_mask(enc_length - 1, dec_length + 1)

    def _get_sequences(self, batch: List[Dict[str, Any]], train: bool) -> Tuple[List[str], List[str]]:
        reactants = [item["reactants"] for item in batch]
        products = [item["products"] for item in batch]

        if train and self._batch_augmenter.augment_prob > 0.0:
            reactants = self._batch_augmenter(reactants)
            products = self._batch_augmenter(products)

        return reactants, products

    def _load_all_data(self) -> None:
        with open(self.dataset_path, "r") as fileobj:
            lines = fileobj.read().splitlines()
        if ">>" in lines[0]:
            reactants, products = zip(*[line.split(">>") for line in lines])
        else:
            reactants = lines
            products = lines.copy()
        self._all_data = {"reactants": reactants, "products": products}

    def _transform_batch(
        self, batch: List[Dict[str, Any]], train: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        encoder_smiles, decoder_smiles = self._get_sequences(batch, train)
        encoder_ids, encoder_mask = self._encoder(encoder_smiles, add_sep_token=self.unified_model and not self.reverse)
        decoder_ids, decoder_mask = self._encoder(decoder_smiles, add_sep_token=self.unified_model and self.reverse)
        if not self.reverse:
            return encoder_ids, encoder_mask, decoder_ids, decoder_mask, decoder_smiles
        return decoder_ids, decoder_mask, encoder_ids, encoder_mask, encoder_smiles

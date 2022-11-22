import os
import torch
import random
import pytorch_lightning as pl
from rdkit import Chem
from functools import partial
from typing import List, Optional
from torch.utils.data import DataLoader
from pysmilesutils.augment import MolAugmenter

from molbart.tokeniser import MolEncTokeniser
from molbart.data.util import TokenSampler
from molbart.data.datasets import MoleculeDataset, ReactionDataset


class _AbsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        tokeniser,
        batch_size,
        max_seq_len,
        train_token_batch_size=None,
        num_buckets=None,
        val_idxs=None, 
        test_idxs=None,
        split_perc=0.2,
        pin_memory=True
    ):
        super().__init__()

        if val_idxs is not None and test_idxs is not None:
            idxs_intersect = set(val_idxs).intersection(set(test_idxs))
            if len(idxs_intersect) > 0:
                raise ValueError(f"Val idxs and test idxs overlap")

        if train_token_batch_size is not None and num_buckets is not None:
            print(f"""Training with approx. {train_token_batch_size} tokens per batch"""
                f""" and {num_buckets} buckets in the sampler.""")
        else:
            print(f"Using a batch size of {str(batch_size)}.")

        self.dataset = dataset
        self.tokeniser = tokeniser

        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.train_token_batch_size = train_token_batch_size
        self.num_buckets = num_buckets
        self.val_idxs = val_idxs
        self.test_idxs = test_idxs
        self.split_perc = split_perc
        self.pin_memory = pin_memory

        self._num_workers = len(os.sched_getaffinity(0))

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    # Use train_token_batch_size with TokenSampler for training and batch_size for validation and testing
    def train_dataloader(self):
        if self.train_token_batch_size is None:
            loader = DataLoader(
                self.train_dataset, 
                batch_size=self.batch_size,
                num_workers=self._num_workers, 
                collate_fn=self._collate,
                shuffle=True,
                pin_memory=self.pin_memory
            )
            return loader

        sampler = TokenSampler(
            self.num_buckets,
            self.train_dataset.seq_lengths,
            self.train_token_batch_size,
            shuffle=True
        )
        loader = DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=self._num_workers,
            collate_fn=self._collate,
            pin_memory=self.pin_memory
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            num_workers=self._num_workers, 
            collate_fn=partial(self._collate, train=False),
            pin_memory=self.pin_memory
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            num_workers=self._num_workers, 
            collate_fn=partial(self._collate, train=False),
            pin_memory=self.pin_memory
        )
        return loader

    def setup(self, stage=None):
        train_dataset = None
        val_dataset = None
        test_dataset = None

        # Split datasets by idxs passed in...
        if self.val_idxs is not None and self.test_idxs is not None:
            train_dataset, val_dataset, test_dataset = self.dataset.split_idxs(self.val_idxs, self.test_idxs)

        # Or randomly
        else:
            train_dataset, val_dataset, test_dataset = self.dataset.split(self.split_perc, self.split_perc)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def _collate(self, batch, train=True):
        raise NotImplementedError()

    def _check_seq_len(self, tokens, mask):
        """ Warn user and shorten sequence if the tokens are too long, otherwise return original

        Args:
            tokens (List[List[str]]): List of token sequences
            mask (List[List[int]]): List of mask sequences

        Returns:
            tokens (List[List[str]]): List of token sequences (shortened, if necessary)
            mask (List[List[int]]): List of mask sequences (shortened, if necessary)
        """

        seq_len = max([len(ts) for ts in tokens])
        if seq_len > self.max_seq_len:
            print(f"WARNING -- Sequence length {seq_len} is larger than maximum sequence size")

            tokens_short = [ts[:self.max_seq_len] for ts in tokens]
            mask_short = [ms[:self.max_seq_len] for ms in mask]

            return tokens_short, mask_short

        return tokens, mask

    def _build_att_mask(self, enc_length, dec_length):
        seq_len = enc_length + dec_length
        enc_mask = torch.zeros((seq_len, enc_length))
        upper_dec_mask = torch.ones((enc_length, dec_length))
        lower_dec_mask = torch.ones((dec_length, dec_length)).triu_(1)
        dec_mask = torch.cat((upper_dec_mask, lower_dec_mask), dim=0)
        mask = torch.cat((enc_mask, dec_mask), dim=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def _build_target_mask(self, enc_length, dec_length, batch_size):
        # Take one and add one because we shift the target left one token
        # So the first token of the target output will be at the same position as the separator token of the input,
        # And the separator token is not present in the output
        enc_mask = [1] * (enc_length - 1)
        dec_mask = [0] * (dec_length + 1)
        mask = [enc_mask + dec_mask] * batch_size
        mask = torch.tensor(mask, dtype=torch.bool).T
        return mask


class MoleculeDataModule(_AbsDataModule):
    def __init__(
        self,
        dataset: MoleculeDataset,
        tokeniser: MolEncTokeniser,
        batch_size: int,
        max_seq_len: int,
        task: str,
        train_token_batch_size: Optional[int] = None,
        num_buckets: Optional[int] = None,
        val_idxs: Optional[List[int]] = None,
        test_idxs: Optional[List[int]] = None,
        split_perc: Optional[float] = 0.2,
        augment: Optional[bool] = True,
        pin_memory: Optional[bool] = True,
        unified_model: Optional[bool] = False
    ):
        super().__init__(
            dataset,
            tokeniser,
            batch_size,
            max_seq_len,
            train_token_batch_size=train_token_batch_size,
            num_buckets=num_buckets,
            val_idxs=val_idxs, 
            test_idxs=test_idxs,
            split_perc=split_perc,
            pin_memory=pin_memory
        )

        if augment:
            print("Using molecule data module with augmentations.")
            self.aug = MolAugmenter() 
        else:
            print("No molecular augmentation.")
            self.aug = None

        self.task = None if task == "None" else task
        self.unified_model = unified_model

    def _collate(self, batch, train=True):
        if self.unified_model:
            collate_output = self._collate_unified(batch, train)
            return collate_output

        token_output = self._prepare_tokens(batch, train)
        enc_tokens = token_output["encoder_tokens"]
        enc_pad_mask = token_output["encoder_pad_mask"]
        dec_tokens = token_output["decoder_tokens"]
        dec_pad_mask = token_output["decoder_pad_mask"]
        target_smiles = token_output["target_smiles"]

        enc_token_ids = self.tokeniser.convert_tokens_to_ids(enc_tokens)
        dec_token_ids = self.tokeniser.convert_tokens_to_ids(dec_tokens)

        enc_token_ids = torch.tensor(enc_token_ids).transpose(0, 1)
        enc_pad_mask = torch.tensor(enc_pad_mask, dtype=torch.bool).transpose(0, 1)
        dec_token_ids = torch.tensor(dec_token_ids).transpose(0, 1)
        dec_pad_mask = torch.tensor(dec_pad_mask, dtype=torch.bool).transpose(0, 1)

        collate_output = {
            "encoder_input": enc_token_ids,
            "encoder_pad_mask": enc_pad_mask,
            "decoder_input": dec_token_ids[:-1, :],
            "decoder_pad_mask": dec_pad_mask[:-1, :],
            "target": dec_token_ids.clone()[1:, :],
            "target_mask": dec_pad_mask.clone()[1:, :],
            "target_smiles": target_smiles
        }
        return collate_output

    def _collate_unified(self, batch, train):
        token_output = self._prepare_tokens(batch, train)
        enc_tokens = token_output["encoder_tokens"]
        enc_pad_mask = token_output["encoder_pad_mask"]
        dec_tokens = token_output["decoder_tokens"]
        dec_pad_mask = token_output["decoder_pad_mask"]
        target_smiles = token_output["target_smiles"]

        sep_token = self.tokeniser.sep_token
        enc_tokens = [tokens + [sep_token] for tokens in enc_tokens]
        enc_pad_mask = [mask + [0] for mask in enc_pad_mask]

        enc_token_ids = self.tokeniser.convert_tokens_to_ids(enc_tokens)
        dec_token_ids = self.tokeniser.convert_tokens_to_ids(dec_tokens)

        enc_token_ids = torch.tensor(enc_token_ids).transpose(0, 1)
        enc_pad_mask = torch.tensor(enc_pad_mask, dtype=torch.bool).transpose(0, 1)
        dec_token_ids = torch.tensor(dec_token_ids).transpose(0, 1)[1:, :]
        dec_pad_mask = torch.tensor(dec_pad_mask, dtype=torch.bool).transpose(0, 1)[1:, :]

        enc_length, batch_size = tuple(enc_token_ids.shape)
        dec_length, _ = tuple(dec_token_ids[:-1, :].shape)
        att_mask = self._build_att_mask(enc_length, dec_length)

        target = torch.cat((enc_token_ids.clone()[:-1, :], dec_token_ids.clone()), dim=0)
        target_mask = self._build_target_mask(enc_length, dec_length, batch_size)
        target_mask = target_mask + (torch.cat((enc_pad_mask[:-1, :], dec_pad_mask), dim=0))

        collate_output = {
            "encoder_input": enc_token_ids,
            "encoder_pad_mask": enc_pad_mask,
            "decoder_input": dec_token_ids[:-1, :],
            "decoder_pad_mask": dec_pad_mask[:-1, :],
            "attention_mask": att_mask,
            "target": target,
            "target_mask": target_mask,
            "target_smiles": target_smiles
        }
        return collate_output

    def _prepare_tokens(self, batch, train):
        aug = self.aug is not None
        if aug:
            encoder_mols = self.aug(batch)
        else:
            encoder_mols = batch[:]

        if self.task == "mask" or self.task is None:
            decoder_mols = encoder_mols[:]
        elif self.task == "mask_aug" or self.task == "aug":
            decoder_mols = self.aug(encoder_mols)
        else:
            raise ValueError(f"Unknown task: {self.task}")

        canonical = self.aug is None
        enc_smiles = []
        dec_smiles = []

        # There is a very rare possibility that RDKit will not be able to generate the SMILES for the augmented mol
        # In this case we just use the canonical mol to generate the SMILES
        for idx, (enc_mol, dec_mol) in enumerate(zip(encoder_mols, decoder_mols)):
            try:
                enc_smi = Chem.MolToSmiles(enc_mol, canonical=canonical)
            except RuntimeError:
                enc_smi = Chem.MolToSmiles(batch[idx], canonical=True)
                print(f"Could not generate smiles after augmenting: {enc_smi}")

            try:
                dec_smi = Chem.MolToSmiles(dec_mol, canonical=canonical)
            except RuntimeError:
                dec_smi = Chem.MolToSmiles(batch[idx], canonical=True)
                print(f"Could not generate smiles after augmenting: {dec_smi}")

            enc_smiles.append(enc_smi)
            dec_smiles.append(dec_smi)

        if self.task == "aug" or self.task is None:
            enc_token_output = self.tokeniser.tokenise(enc_smiles, pad=True)
            enc_tokens = enc_token_output["original_tokens"]
            enc_mask = enc_token_output["original_pad_masks"]
        elif self.task == "mask" or self.task == "mask_aug":
            enc_token_output = self.tokeniser.tokenise(enc_smiles, mask=True, pad=True)
            enc_tokens = enc_token_output["masked_tokens"]
            enc_mask = enc_token_output["masked_pad_masks"]
        else:
            raise ValueError(f"Unknown task: {self.task}")

        dec_token_output = self.tokeniser.tokenise(dec_smiles, pad=True)
        dec_tokens = dec_token_output["original_tokens"]
        dec_mask = dec_token_output["original_pad_masks"]

        enc_tokens, enc_mask = self._check_seq_len(enc_tokens, enc_mask)
        dec_tokens, dec_mask = self._check_seq_len(dec_tokens, dec_mask)

        # Ensure that the canonical form is used for evaluation
        dec_mols = [Chem.MolFromSmiles(smi) for smi in dec_smiles]
        canon_targets = [Chem.MolToSmiles(mol) for mol in dec_mols]

        token_output = {
            "encoder_tokens": enc_tokens,
            "encoder_pad_mask": enc_mask,
            "decoder_tokens": dec_tokens,
            "decoder_pad_mask": dec_mask,
            "target_smiles": canon_targets
        }
        return token_output


class FineTuneReactionDataModule(_AbsDataModule):
    def __init__(
        self,
        dataset: ReactionDataset,
        tokeniser: MolEncTokeniser,
        batch_size: int,
        max_seq_len: int,
        train_token_batch_size: Optional[int] = None,
        num_buckets: Optional[int] = None,
        forward_pred: Optional[bool] = True,
        val_idxs: Optional[List[int]] = None, 
        test_idxs: Optional[List[int]] = None,
        split_perc: Optional[float] = 0.2,
        pin_memory: Optional[bool] = True,
        unified_model: Optional[bool] = False
    ):
        super().__init__(
            dataset,
            tokeniser,
            batch_size,
            max_seq_len,
            train_token_batch_size=train_token_batch_size,
            num_buckets=num_buckets,
            val_idxs=val_idxs, 
            test_idxs=test_idxs,
            split_perc=split_perc,
            pin_memory=pin_memory
        )

        if forward_pred:
            print("Building data module for forward prediction task...")
        else:
            print("Building data module for backward prediction task...")

        self.forward_pred = forward_pred
        self.unified_model = unified_model

    def _collate(self, batch, train=True):
        if self.unified_model:
            collate_output = self._collate_unified(batch)
            return collate_output

        reacts_smiles, prods_smiles = tuple(zip(*batch))
        reacts_output = self.tokeniser.tokenise(reacts_smiles, pad=True)
        prods_output = self.tokeniser.tokenise(prods_smiles, pad=True)

        reacts_tokens = reacts_output["original_tokens"]
        reacts_mask = reacts_output["original_pad_masks"]
        reacts_tokens, reacts_mask = self._check_seq_len(reacts_tokens, reacts_mask)

        prods_tokens = prods_output["original_tokens"]
        prods_mask = prods_output["original_pad_masks"]
        prods_tokens, prods_mask = self._check_seq_len(prods_tokens, prods_mask)

        reacts_token_ids = self.tokeniser.convert_tokens_to_ids(reacts_tokens)
        prods_token_ids = self.tokeniser.convert_tokens_to_ids(prods_tokens)

        reacts_token_ids = torch.tensor(reacts_token_ids).transpose(0, 1)
        reacts_pad_mask = torch.tensor(reacts_mask, dtype=torch.bool).transpose(0, 1)
        prods_token_ids = torch.tensor(prods_token_ids).transpose(0, 1)
        prods_pad_mask = torch.tensor(prods_mask, dtype=torch.bool).transpose(0, 1)

        if self.forward_pred:
            collate_output = {
                "encoder_input": reacts_token_ids,
                "encoder_pad_mask": reacts_pad_mask,
                "decoder_input": prods_token_ids[:-1, :],
                "decoder_pad_mask": prods_pad_mask[:-1, :],
                "target": prods_token_ids.clone()[1:, :],
                "target_mask": prods_pad_mask.clone()[1:, :],
                "target_smiles": prods_smiles
            }
        else:
            collate_output = {
                "encoder_input": prods_token_ids,
                "encoder_pad_mask": prods_pad_mask,
                "decoder_input": reacts_token_ids[:-1, :],
                "decoder_pad_mask": reacts_pad_mask[:-1, :],
                "target": reacts_token_ids.clone()[1:, :],
                "target_mask": reacts_pad_mask.clone()[1:, :],
                "target_smiles": reacts_smiles
            }

        return collate_output

    def _collate_unified(self, batch):
        reacts_smiles, prods_smiles = tuple(zip(*batch))
        reacts_output = self.tokeniser.tokenise(reacts_smiles, pad=True)
        prods_output = self.tokeniser.tokenise(prods_smiles, pad=True)

        if self.forward_pred:
            enc_tokens = reacts_output["original_tokens"]
            enc_mask = reacts_output["original_pad_masks"]
            dec_tokens = prods_output["original_tokens"]
            dec_mask = prods_output["original_pad_masks"]
            target_smiles = prods_smiles
        else:
            dec_tokens = reacts_output["original_tokens"]
            dec_mask = reacts_output["original_pad_masks"]
            enc_tokens = prods_output["original_tokens"]
            enc_mask = prods_output["original_pad_masks"]
            target_smiles = reacts_smiles

        sep_token = self.tokeniser.sep_token
        enc_tokens = [tokens + [sep_token] for tokens in enc_tokens]
        enc_mask = [mask + [0] for mask in enc_mask]

        # TODO Check length of combined sequence

        enc_token_ids = self.tokeniser.convert_tokens_to_ids(enc_tokens)
        dec_token_ids = self.tokeniser.convert_tokens_to_ids(dec_tokens)

        enc_token_ids = torch.tensor(enc_token_ids).transpose(0, 1)
        enc_mask = torch.tensor(enc_mask, dtype=torch.bool).transpose(0, 1)
        dec_token_ids = torch.tensor(dec_token_ids).transpose(0, 1)[1:, :]
        dec_mask = torch.tensor(dec_mask, dtype=torch.bool).transpose(0, 1)[1:, :]

        enc_length, batch_size = tuple(enc_token_ids.shape)
        dec_length, _ = tuple(dec_token_ids[:-1, :].shape)
        att_mask = self._build_att_mask(enc_length - 1, dec_length + 1)

        target = torch.cat((enc_token_ids.clone()[:-1, :], dec_token_ids.clone()), dim=0)
        target_mask = self._build_target_mask(enc_length, dec_length, batch_size)
        target_mask = target_mask + (torch.cat((enc_mask[:-1, :], dec_mask), dim=0))

        collate_output = {
            "encoder_input": enc_token_ids,
            "encoder_pad_mask": enc_mask,
            "decoder_input": dec_token_ids[:-1, :],
            "decoder_pad_mask": dec_mask[:-1, :],
            "attention_mask": att_mask,
            "target": target,
            "target_mask": target_mask,
            "target_smiles": target_smiles
        }
        return collate_output

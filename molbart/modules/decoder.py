from __future__ import annotations

from operator import mod
from os import truncate
from tabnanny import verbose
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from pysmilesutils.tokenize import SMILESTokenizer
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from molbart.modules.search import EOS, LogicalOr, MaxLength, Node, beamsearch

if TYPE_CHECKING:
    from typing import Any, Dict, List, Tuple

    from molbart.models.base_transformer import _AbsTransformerModel

RDLogger.DisableLog("rdApp.*")


class BeamSearchSampler:
    """
    GPU-optimized beam search sampler/decoder. Generates predictions and
    calculates performance metrics.
    """

    def __init__(
        self,
        tokenizer: SMILESTokenizer,
        max_sequence_length: int,
        device: str = "cuda",
        data_device: str = "cuda",
        sample_unique: bool = True,
    ) -> None:
        """
        Args:
            tokenizer (SMILESTokenizer): Tokenizer with vocabulary.
            max_sequence_length (int): Maximum generated sequence length.
            device (str): "cuda" or "cpu".
            data_device (str): device used for handling the data. If memory issues,
                could help to set data_device="cpu"
            sampled_unique (bool):  Whether to return unique beam search solutions.
        """
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.device = device
        self.fraction_invalid = None
        self.fraction_unique = None
        self.molecules_unique = None
        self.smiles_unique = None
        self.log_lhs_unique = None
        self.sampling_alg = None
        self.beam_size = None
        self.n_unique_beams = 1
        self.data_device = data_device
        self.sample_unique = sample_unique
        return

    @staticmethod
    def _canonicalize_smiles(input_smiles: str) -> str:
        """
        Canonicalize smiles and sort the (possible) multiple molcules.

        Args:
            input_smiles (str): SMILES string.
        Returns:
            str: Canonicalized SMILES string.
        """
        mol = Chem.MolFromSmiles(input_smiles)
        if mol is None:
            return input_smiles
        smiles_canonical = Chem.MolToSmiles(mol)

        smiles_sep = np.array(smiles_canonical.split("."))
        smiles_canonical = ".".join(np.sort(smiles_sep))
        return smiles_canonical

    def _postprocess_sampling(
        self, batch_smiles: np.ndarray, batch_log_lhs: np.ndarray, n_unique_beams: int
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], float]:
        """
        For beam_size > 1: Uniqueifying sampled molecules and select
        'n_unique_beams'-top molecules.

        Args:
            batch_smiles (np.ndarray): sampled top-N SMILES from the input batch smiles.
            batch_log_lhs (np.ndarray): log-likelihoods of sampled SMILES.
            n_unique_beams (int): upper limit on number of unique SMILES to return.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, float]: Uniqueified SMILES,
                corresponding rdkit molecules, log-likelihoods and fraction of
                unique SMILES.
        """

        n_samples = len(batch_smiles)
        n_beams = len(batch_smiles[0])

        self.n_unique_beams = n_unique_beams

        batch_mols_unique = [""] * n_samples
        batch_smiles_unique = [""] * n_samples
        batch_log_lhs_unique = [0] * n_samples
        inds = np.arange(n_samples)

        n_unique_total = 0
        for i, sampled_smiles, log_lhs in zip(inds, batch_smiles, batch_log_lhs):
            # Canonicalize sampled SMILES
            smiles_canonical = [
                self._canonicalize_smiles(smi) for smi in sampled_smiles
            ]
            sampled_mols = [Chem.MolFromSmiles(smi) for smi in smiles_canonical]

            smiles_canonical = np.array(
                [
                    smi
                    for smi, mol in zip(smiles_canonical, sampled_mols)
                    if mol is not None
                ]
            )

            log_lhs_valid = np.array(
                [llh for llh, mol in zip(log_lhs, sampled_mols) if mol is not None]
            )
            sampled_mols = np.array([mol for mol in sampled_mols if mol is not None])

            # Uniquely sampled SMILES
            smiles_unique = np.unique(smiles_canonical)
            n_unique = len(smiles_unique)
            n_unique_total += n_unique
            log_lhs_unique = np.zeros(n_unique)
            mols_unique = np.zeros(n_unique, dtype=Chem.rdchem.Mol)

            for i_mol, smi in enumerate(smiles_unique):
                log_lhs_unique[i_mol] = np.max(log_lhs_valid[smi == smiles_canonical])
                mols_unique[i_mol] = sampled_mols[smi == smiles_canonical][0]

            # Get top-K unique beams
            sort_inds = np.argsort(log_lhs_unique)[::-1]
            smiles_unique = smiles_unique[sort_inds[0:n_unique_beams]]
            mols_unique = mols_unique[sort_inds[0:n_unique_beams]]
            log_lhs_unique = log_lhs_unique[sort_inds[0:n_unique_beams]]

            batch_smiles_unique[i] = smiles_unique
            batch_mols_unique[i] = mols_unique
            batch_log_lhs_unique[i] = log_lhs_unique

        frac_unique = n_unique_total / (n_beams * n_samples)
        return batch_smiles_unique, batch_mols_unique, batch_log_lhs_unique, frac_unique

    @staticmethod
    def _compute_accuracy(
        sampled_smiles: List[List[str]], target_smiles: List[str], top_Ks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computing top-K accuracy for each K in 'top_Ks'.
        """

        n_beams = np.max(
            np.array(
                [1, np.max(np.asarray([len(smiles) for smiles in sampled_smiles]))]
            )
        )
        top_Ks = top_Ks[top_Ks <= n_beams]
        n_Ks = len(top_Ks)

        accuracy = np.zeros(n_Ks)

        is_in_set = np.zeros((len(sampled_smiles), n_Ks), dtype=bool)
        for i_k, K in enumerate(top_Ks):
            for i_sample, mols in enumerate(sampled_smiles):
                top_K_mols = mols[0:K]

                if len(top_K_mols) == 0:
                    continue
                is_in_set[i_sample, i_k] = target_smiles[i_sample] in top_K_mols

        is_in_set = np.cumsum(is_in_set, axis=1)
        accuracy = np.mean(is_in_set > 0, axis=0)
        return accuracy, top_Ks

    @staticmethod
    def _compute_similarity(
        sampled_mols: List[List[Chem.rdchem.Mol]], target_mols: List[Chem.rdchem.Mol]
    ) -> np.ndarray:
        """
        Compute similarities of ECPF4 fingerprints of target and top-1 sampled molecules.
        """
        n_samples = len(target_mols)

        similarity = np.nan * np.ones(n_samples)
        for idx, sampled, target in zip(range(n_samples), sampled_mols, target_mols):
            if len(sampled) > 0:
                if sampled[0] is None or target is None:
                    continue
                fp1 = AllChem.GetMorganFingerprint(sampled[0], 2)
                fp2 = AllChem.GetMorganFingerprint(target, 2)
                similarity[idx] = DataStructs.TanimotoSimilarity(
                    fp1, fp2
                )  # Tanimoto similarity = Jaccard similarity
        return similarity

    def _evaluate_sampled_molecules(
        self,
        sampled_smiles: List[List[str]],
        target_smiles: List[str],
        top_Ks: np.ndarray = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50]),
        is_canonical: bool = False,
        compute_similarity: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute top-1 to top-50 accuracy of (unique set of) sampled smiles.
        """
        # Canonicalizing target SMILES
        target_smiles_canonical = [
            self._canonicalize_smiles(smi) for smi in target_smiles
        ]
        target_mols = [Chem.MolFromSmiles(smi) for smi in target_smiles_canonical]

        # Canonicalizing sampled SMILES
        if not is_canonical:
            sampled_smiles = [
                [self._canonicalize_smiles(smi) for smi in smiles_list]
                for smiles_list in sampled_smiles
            ]
        sampled_molecules = [
            [Chem.MolFromSmiles(smi) for smi in smiles_list]
            for smiles_list in sampled_smiles
        ]

        accuracy, top_Ks = self._compute_accuracy(
            sampled_smiles, target_smiles_canonical, top_Ks
        )

        similarity = None
        if compute_similarity:
            similarity = self._compute_similarity(sampled_molecules, target_mols)
        return accuracy, top_Ks, similarity

    @torch.no_grad()
    def sample_molecules(
        self,
        model: _AbsTransformerModel,
        batch_input: Dict[str, Any],
        beam_size: int,
        sampling_alg: str = "beam",
        return_tokenized: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample molecules from the model using the search.py implementation of beam
        search.

        Args:
            model (_AbsTransformerModel): The transformer base model (e.g. BARTModel or
                UnifiedModel)
            batch_input (Dict): The input, X, to the network
            beam_size (int): Number of beams in beam search
            sampling_alg (str): Algorithm to use for sampling from the model ("greedy"
                or "beam")

        Returns:
            (np.ndarray, np.ndarray): (SMILES of sampled molecules, log-likelihoods) or
            (token indices of sampled molecules, log-likelihoods)
        """
        self.sampling_alg = sampling_alg
        self.molecules_unique = None
        self.beam_size = beam_size

        if self.device is None:
            self.device = next(model.parameters()).device

        _, batch_size = tuple(batch_input["encoder_input"].shape)

        if sampling_alg == "greedy":
            beam_size = 1
        elif sampling_alg != "beam":
            raise ValueError(f"Unknown sampling algorithm {sampling_alg}")

        stop_criterion = LogicalOr((MaxLength(self.max_sequence_length - 1), EOS()))

        node = Node(
            model,
            batch_input,
            self.tokenizer,
            self.device,
            batch_size=batch_size,
            data_device=self.data_device,
        )

        beamsearch(node, beam_size, stop_criterion)

        Y = node.y.detach().cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(Y)

        sampled_smiles = np.asarray(
            self.tokenizer.detokenize(tokens, truncate_at_end_token=True)
        ).reshape((-1, beam_size))

        log_lhs = (node.loglikelihood.detach().cpu().numpy()).reshape(-1, beam_size)

        if beam_size > 1 and self.sample_unique:
            (
                self.smiles_unique,
                self.molecules_unique,
                self.log_lhs_unique,
                self.fraction_unique,
            ) = self._postprocess_sampling(
                sampled_smiles, log_lhs, model.n_unique_beams
            )

        # Computing fraction of invalid top-1 SMILES
        sampled_mols = [Chem.MolFromSmiles(smi[0]) for smi in sampled_smiles]
        is_invalid = [1 for mol in sampled_mols if mol is None]
        self.fraction_invalid = len(is_invalid) / len(sampled_smiles)

        if return_tokenized:
            return Y, log_lhs
        else:
            return sampled_smiles, log_lhs

    def compute_sampling_metrics(
        self,
        sampled_smiles,
        target_smiles,
        is_canonical=False,
        compute_similarity=False,
    ):
        """
        Compute sampling metrics:
        1. Molecular accuracy (fraction of sampled set of molecules that include
        the target molecule (top-K)).
        2. Similarity of each sampled top-1 molecule and target molecule.
        """
        n_samples = len(sampled_smiles)
        n_targets = len(target_smiles)
        err_msg = f"The number of sampled and target molecules must be the same, got {n_samples} and {n_targets}"
        assert n_samples == n_targets, err_msg

        if self.sampling_alg == "greedy":
            top_Ks = np.array([1])
        else:
            top_Ks = np.array([1, 3, 5, 10, 20, 50])

        accuracy, top_Ks, similarity = self._evaluate_sampled_molecules(
            sampled_smiles, target_smiles, top_Ks, is_canonical, compute_similarity
        )

        metrics = {
            "accuracy": accuracy[0],  # top-1 accuracy
            "fraction_invalid": self.fraction_invalid,
            "similarity": [similarity],
            "fraction_unique": self.fraction_unique,
        }

        for i, k in enumerate(top_Ks):
            metrics.update({"accuracy_top_" + str(k): accuracy[i]})
        return metrics


class DecodeSampler:
    def __init__(self, tokenizer, max_seq_len, length_norm=None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.length_norm = length_norm

        assert max_seq_len > 1, f"Max sequence must be at least 2, got {max_seq_len}"

        self.begin_token_id = self.tokenizer["start"]
        self.pad_token_id = self.tokenizer["pad"]
        self.end_token_id = self.tokenizer["end"]

        self.bad_token_ll = -1e5

    def decode(
        self, decode_fn, batch_size, sampling_alg="greedy", device="cpu", **kwargs
    ):
        """Sample a molecule from a model by calling the decode function argument

        Args:
            decode_fn: A function mapping a batched sequence of token identifiers and
                    their associated pad masks to a log probability distribution over
                    possible next tokens
            batch_size: The number of elements to pass into the decode function in one batch
            sampling_alg: Algorithm to use for sampling from the model

        Returns:
            (SMILES of sampled molecules (List[str]), log likelihoods (List[float]))
        """

        if sampling_alg == "greedy":
            output = self.greedy_decode(decode_fn, batch_size, device)

        elif sampling_alg == "beam":
            output = self.beam_decode(decode_fn, batch_size, device, kwargs)

        else:
            raise ValueError(f"Unknown sampling algorithm {sampling_alg}")

        return output

    def greedy_decode(self, decode_fn, batch_size, device="cpu"):
        """Sample molecules from the model using greedy search

        Args:
            decode_fn (fn): Function used to apply tokens to model and produce log probability distribution
            batch_size (int): Number of molecules to sample
            device: Torch device to create tensors on

        Returns:
            (List[str], List[float]): Tuple of (molecules, their log likelihoods)
        """

        # Create tensors which will be reused
        token_ids = [self.begin_token_id] + (
            [self.pad_token_id] * (self.max_seq_len - 1)
        )
        token_ids = [token_ids] * batch_size
        token_ids = torch.tensor(token_ids, device=device).transpose(0, 1)
        pad_mask = torch.zeros(
            (self.max_seq_len, batch_size), device=device, dtype=torch.bool
        )
        log_lhs = torch.zeros((batch_size))

        # Iteratively apply the tokens to the model and build up the sequence
        for i in range(1, self.max_seq_len):
            token_ids_seq = token_ids[:i, :]
            pad_mask_seq = pad_mask[:i, :]

            # Sample next id for each element in the batch
            output_dist = decode_fn(token_ids_seq, pad_mask_seq)
            probs, output_ids = output_dist.max(dim=2)
            new_ids = output_ids[-1, :]
            new_probs = probs[-1, :]

            # Generate next elements in the pad mask. An element is padded if:
            # 1. The previous token is an end token
            # 2. The previous token is a pad token
            is_end_token = token_ids[i - 1, :] == self.end_token_id
            is_pad_token = token_ids[i - 1, :] == self.pad_token_id
            new_pad_mask = torch.logical_or(is_end_token, is_pad_token)

            # Break if sampling is complete
            if new_pad_mask.sum().item() == new_pad_mask.numel():
                break

            # Ensure all sequences contain an end token
            if i == self.max_seq_len - 1:
                new_ids[~new_pad_mask] = self.end_token_id

            # Set the token to pad where required, update the token ids and update lls
            new_ids[new_pad_mask] = self.pad_token_id
            token_ids[i, :] = new_ids
            pad_mask[i, :] = new_pad_mask
            log_lhs += new_probs.cpu()

        tokens = token_ids.transpose(0, 1)
        tokens = self.tokenizer.convert_ids_to_tokens(tokens)
        mol_strs = self.tokenizer.detokenize(tokens, truncate_at_end_token=True)
        log_lhs = log_lhs.tolist()

        return mol_strs, log_lhs

    def beam_decode(self, decode_fn, batch_size, device="cpu", k=5):
        """Sample molecules from the model using beam search

        Samples molecules by iteratively building up the sequence of SMILES characters using beam search.
        Molecules are returned in a 2D list where batch_size is the outer dimension and k is the inner dimension.

        Args:
            decode_fn (fn): Function used to apply tokens to model and produce log probability distribution
            batch_size (int): Number of molecules to sample
            device: Torch device to create tensors on
            k (int): Number of beams

        Returns:
            (List[List[str]], List[List[float]]): Tuple of (molecules, their log likelihoods)
        """

        # Create tensors which will be reused
        token_ids = [self.begin_token_id] + (
            [self.pad_token_id] * (self.max_seq_len - 1)
        )
        token_ids = [token_ids] * batch_size
        token_ids = torch.tensor(token_ids, device=device).transpose(0, 1)
        pad_mask = torch.zeros(
            (self.max_seq_len, batch_size), device=device, dtype=torch.bool
        )

        ts = token_ids[:1, :]
        ms = pad_mask[:1, :]
        ll = torch.zeros((batch_size))

        # Apply starting token to model to get a distribution over next tokens
        first_lls = self._beam_step(decode_fn, ts, ms, ll)
        top_lls, top_idxs = torch.topk(first_lls, k, dim=1)
        top_ids = list(top_idxs.T)

        # Setup tensors for each beam which will be reused
        token_ids_list = [token_ids.clone() for _ in range(k)]
        pad_mask_list = [pad_mask.clone() for _ in range(k)]
        lls_list = list(top_lls.cpu().T)

        for beam_idx, ids in enumerate(top_ids):
            token_ids_list[beam_idx][1, :] = ids
            pad_mask_list[beam_idx][1, :] = 0

        for i in range(2, self.max_seq_len):
            complete = self._update_beams_(
                i, decode_fn, token_ids_list, pad_mask_list, lls_list
            )
            if complete:
                break

        tokens_list = [token_ids.transpose(0, 1) for token_ids in token_ids_list]
        tokens_list = [
            self.tokenizer.convert_ids_to_tokens(tokens) for tokens in tokens_list
        ]

        mol_strs_list = [
            self.tokenizer.detokenize(tokens, truncate_at_end_token=True)
            for tokens in tokens_list
        ]
        log_lhs_list = [log_lhs.tolist() for log_lhs in lls_list]

        # Transpose and sort list of molecules based on ll
        new_mol_strs = self._transpose_list(mol_strs_list)
        new_log_lhs = self._transpose_list(log_lhs_list)
        sorted_mols, sorted_lls = self._sort_beams(new_mol_strs, new_log_lhs)

        return sorted_mols, sorted_lls

    def _update_beams_(self, i, decode_fn, token_ids_list, pad_mask_list, lls_list):
        """Update beam tokens and pad mask in-place using a single decode step

        Updates token ids and pad mask in-place by producing the probability distribution over next tokens
        and choosing the top k (number of beams) log likelihoods to choose the next tokens.
        Sampling is complete if every batch element in every beam has produced an end token.

        Args:
            i (int): The current iteration counter
            decode_fn (fn): Function used to apply tokens to model and produce log probability distribution
            token_ids_list (List[torch.Tensor]): List of token_ids, each of shape [seq_len, batch_size]
            pad_mask_list (List[torch.Tensor]): List of pad_masks, each of shape [seq_len, batch_size]
            lls_list (List[torch.Tensor]): List of log likelihoods, each of shape [batch_size]

        Returns:
            (bool): Specifies whether all of the beams are complete
        """

        assert len(token_ids_list) == len(pad_mask_list) == len(lls_list)

        num_beams = len(token_ids_list)

        ts = [token_ids[:i, :] for token_ids in token_ids_list]
        ms = [pad_mask[:i, :] for pad_mask in pad_mask_list]

        # Apply current seqs to model to get a distribution over next tokens
        # new_lls is a tensor of shape [batch_size, vocab_size * num_beams]
        new_lls = [
            self._beam_step(decode_fn, t, m, lls) for t, m, lls in zip(ts, ms, lls_list)
        ]
        norm_lls = [self._norm_length(lls, mask) for lls, mask in zip(new_lls, ms)]

        _, vocab_size = tuple(norm_lls[0].shape)
        new_lls = torch.cat(new_lls, dim=1)
        norm_lls = torch.cat(norm_lls, dim=1)

        # Keep lists (of length num_beams) of tensors of shape [batch_size]
        top_lls, top_idxs = torch.topk(norm_lls, num_beams, dim=1)
        new_ids_list = list((top_idxs % vocab_size).T)
        beam_idxs_list = list((top_idxs // vocab_size).T)
        top_lls = [new_lls[b_idx, idx] for b_idx, idx in enumerate(list(top_idxs))]
        top_lls = torch.stack(top_lls).T

        beam_complete = []
        new_ts_list = []
        new_pm_list = []
        new_lls_list = []

        # Set the sampled tokens, pad masks and log likelihoods for each of the new beams
        for new_beam_idx, (new_ids, beam_idxs, lls) in enumerate(
            zip(new_ids_list, beam_idxs_list, top_lls)
        ):
            # Get the previous sequences corresponding to the new beams
            token_ids = [
                token_ids_list[beam_idx][:, b_idx]
                for b_idx, beam_idx in enumerate(beam_idxs)
            ]
            token_ids = torch.stack(token_ids).transpose(0, 1)

            # Generate next elements in the pad mask. An element is padded if:
            # 1. The previous token is an end token
            # 2. The previous token is a pad token
            is_end_token = token_ids[i - 1, :] == self.end_token_id
            is_pad_token = token_ids[i - 1, :] == self.pad_token_id
            new_pad_mask = torch.logical_or(is_end_token, is_pad_token)
            beam_complete.append(new_pad_mask.sum().item() == new_pad_mask.numel())

            # Ensure all sequences contain an end token
            if i == self.max_seq_len - 1:
                new_ids[~new_pad_mask] = self.end_token_id

            # Set the tokens to pad if an end token as already been produced
            new_ids[new_pad_mask] = self.pad_token_id
            token_ids[i, :] = new_ids

            # Generate full pad mask sequence for new token sequence
            pad_mask = [
                pad_mask_list[beam_idx][:, b_idx]
                for b_idx, beam_idx in enumerate(beam_idxs)
            ]
            pad_mask = torch.stack(pad_mask).transpose(0, 1)
            pad_mask[i, :] = new_pad_mask

            # Add tokens, pad mask and lls to list to be updated after all beams have been processed
            new_ts_list.append(token_ids)
            new_pm_list.append(pad_mask)
            new_lls_list.append(lls)

        complete = sum(beam_complete) == len(beam_complete)

        # Update all tokens, pad masks and lls
        if not complete:
            for beam_idx, (ts, pm, lls) in enumerate(
                zip(new_ts_list, new_pm_list, new_lls_list)
            ):
                token_ids_list[beam_idx] = ts
                pad_mask_list[beam_idx] = pm
                lls_list[beam_idx] = lls

        return complete

    def _beam_step(self, decode_fn, tokens, mask, lls):
        """Apply tokens to model to produce the log likelihoods for the full sequence

        A single iteration of decode is applied to the model to produce the next tokens in the sequences
        and the log likelihoods for the entire sequences (including the next token)
        The lls are returned as a distribution over all possible next tokens

        Args:
            decode_fn (fn): Function used to apply tokens to model and produce log probability distribution
            tokens (torch.Tensor): Tensor of shape [seq_len, batch_size] containing the current token ids
            mask (torch.Tensor): BoolTensor of shape [seq_len, batch_size] containing the padding mask
            lls (torch.Tensor): Tensor of shape [batch_size] containing log likelihoods for seqs so far

        Returns:
            seq_lls (torch.Tensor): Tensor of shape [batch_size, vocab_size]
        """
        output_dist = decode_fn(tokens, mask)
        next_token_lls = output_dist[-1, :, :].cpu()

        # Create a vector from which only a pad token can be sampled
        _, vocab_size = tuple(next_token_lls.shape)
        complete_seq_ll = torch.ones((1, vocab_size)) * self.bad_token_ll
        complete_seq_ll[:, self.pad_token_id] = 0.0

        # Use this vector in the output for sequences which are complete
        is_end_token = tokens[-1, :] == self.end_token_id
        is_pad_token = tokens[-1, :] == self.pad_token_id
        ll_mask = torch.logical_or(is_end_token, is_pad_token).cpu().unsqueeze(1)
        masked_lls = (ll_mask * complete_seq_ll) + (~ll_mask * next_token_lls)

        seq_lls = (lls + masked_lls.T).T
        return seq_lls

    def _norm_length(self, seq_lls, mask):
        """Normalise log-likelihoods using the length of the constructed sequence
        Equation from:
        Wu, Yonghui, et al.
        "Google's neural machine translation system: Bridging the gap between human and machine translation."
        arXiv preprint arXiv:1609.08144 (2016).

        Args:
            seq_lls (torch.Tensor): Tensor of shape [batch_size, vocab_size] containing log likelihoods for seqs so far
            mask (torch.Tensor): BoolTensor of shape [seq_len, batch_size] containing the padding mask

        Returns:
            norm_lls (torch.Tensor): Tensor of shape [batch_size, vocab_size]
        """

        if self.length_norm is not None:
            seq_lengths = (~mask).sum(dim=0)
            norm = torch.pow(5 + seq_lengths, self.length_norm) / pow(
                6, self.length_norm
            )
            norm_lls = (seq_lls.T / norm.cpu()).T
            return norm_lls

        return seq_lls

    @staticmethod
    def _transpose_list(list_):
        """Transpose 2D list so that inner dimension is first

        Args:
            l (List[Any]): List to be transposed

        Returns:
            (List[Any]): Transposed list
        """

        outer_dim = len(list_)
        inner_dim = len(list_[0])

        transposed = [[[]] * outer_dim for _ in range(inner_dim)]
        for outer_idx, inner in enumerate(list_):
            for inner_idx, item in enumerate(inner):
                transposed[inner_idx][outer_idx] = item

        return transposed

    @staticmethod
    def _sort_beams(mol_strs, log_lhs):
        """Return mols sorted by their log likelihood

        Args:
            mol_strs (List[List[str]]): SMILES encoding of molecules
            log_lhs (List[List[float]]): Log likelihood for each molecule

        Returns:
            (List[str], List[float]): Tuple of sorted molecules and sorted log lhs
        """

        assert len(mol_strs) == len(log_lhs)

        sorted_mols = []
        sorted_lls = []

        for mols, lls in zip(mol_strs, log_lhs):
            mol_lls = sorted(zip(mols, lls), reverse=True, key=lambda mol_ll: mol_ll[1])
            mols, lls = tuple(zip(*mol_lls))
            sorted_mols.append(list(mols))
            sorted_lls.append(list(lls))

        return sorted_mols, sorted_lls

    @staticmethod
    def compute_sampling_metrics(sampled_smiles, target_smiles):
        """Calculate sampling metrics for the model

        If sampled_smiles is a List[List[str]] then the following metrics for beam search are calculated (up to the
        maximum given by the number of elements in the inner lists):
            - "top_1_accuracy"
            - "top_5_accuracy"
            - "top_10_accuracy"
            - "top_20_accuracy"
            - "top_50_accuracy"
        The SMILES strings must be sorted in decreasing order of their predicted likelihood

        If the sampled_smiles is a List[str] then "accuracy" is calculated

        The the number of invalid SMILES "fraction_invalid" is also returned (for beam search this is just from the top_1)

        Args:
            sampled_smiles: SMILES strings produced by decode function,
            target_smiles: target molecules as canonicalised SMILES strings

        Returns:
            dict containing results
        """

        num_sampled = len(sampled_smiles)
        num_target = len(target_smiles)
        err_msg = f"The number of sampled and target molecules must be the same, got {num_sampled} and {num_target}"
        assert num_sampled == num_target, err_msg

        mol_targets = [Chem.MolFromSmiles(smi) for smi in target_smiles]
        canon_targets = [Chem.MolToSmiles(mol) for mol in mol_targets]

        data_type = type(sampled_smiles[0])
        if data_type == str:
            results = DecodeSampler._calc_greedy_metrics(sampled_smiles, canon_targets)
        elif data_type == list:
            results = DecodeSampler._calc_beam_metrics(sampled_smiles, canon_targets)
        else:
            raise TypeError(
                f"Elements of sampled_smiles must be either a str or a list, got {data_type}"
            )

        return results

    @staticmethod
    def _calc_greedy_metrics(sampled_smiles, target_smiles):
        sampled_mols = [Chem.MolFromSmiles(smi) for smi in sampled_smiles]
        invalid = [mol is None for mol in sampled_mols]

        canon_smiles = [
            "Unknown" if mol is None else Chem.MolToSmiles(mol) for mol in sampled_mols
        ]
        correct_smiles = [
            target_smiles[idx] == smi for idx, smi in enumerate(canon_smiles)
        ]

        num_correct = sum(correct_smiles)
        total = len(correct_smiles)
        num_invalid = sum(invalid)
        perc_invalid = num_invalid / total
        accuracy = num_correct / total

        metrics = {"accuracy": accuracy, "fraction_invalid": perc_invalid}

        return metrics

    @staticmethod
    def _calc_beam_metrics(sampled_smiles, target_smiles):
        top_1_samples = [mols[0] for mols in sampled_smiles]
        top_1_results = DecodeSampler._calc_greedy_metrics(top_1_samples, target_smiles)

        metrics = {
            "accuracy": top_1_results["accuracy"],
            "fraction_invalid": top_1_results["fraction_invalid"],
        }

        ks = [2, 3, 5, 10, 20, 50]
        num_samples_list = [k for k in ks if k <= len(sampled_smiles[0])]

        for num_samples in num_samples_list:
            top_k_correct = []
            num_mols = len(sampled_smiles)

            for batch_idx, mols in enumerate(sampled_smiles):
                samples = mols[:num_samples]
                samples_mols = [Chem.MolFromSmiles(smi) for smi in samples]
                samples_smiles = [
                    "Unknown" if mol is None else Chem.MolToSmiles(mol)
                    for mol in samples_mols
                ]
                correct_smiles = [
                    smi == target_smiles[batch_idx] for smi in samples_smiles
                ]
                is_correct = sum(correct_smiles) >= 1
                top_k_correct.append(is_correct)

            accuracy = sum(top_k_correct) / num_mols
            metrics[f"top_{str(num_samples)}_accuracy"] = accuracy

        return metrics

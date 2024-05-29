from typing import Any, Dict, List, Tuple, Union

import numpy as np

from molbart.models import Chemformer


def compute_round_trip_accuracy(
    chemformer: Chemformer,
    sampled_smiles: List[np.ndarray],
    target_smiles: List[List[str]],
) -> List[Dict[str, Any]]:
    """
    Calculating (round-trip) accuracy given sampled and target SMILES (products).

    Args:
        chemformer: a Chemformer model with a decoder sampler
        sampled_smiles: product SMILES sampled by forward Chemformer
        target_smiles: ground truth product SMILES
    """
    print("Evaluating predictions.")

    metrics_out = []
    batch_idx = 0
    for sampled_batch, target_batch in zip(sampled_smiles, target_smiles):
        metrics = chemformer.model.sampler.compute_sampling_metrics(
            sampled_batch,
            target_batch,
            is_canonical=False,
        )

        metrics.update({"sampled_molecules": sampled_batch, "target_smiles": target_batch})

        metrics_out.append(metrics)
        batch_idx += 1
    return metrics_out


def batchify(smiles_lst: Union[List[str], np.ndarray], batch_size: int) -> Union[List[List[str]], List[np.ndarray]]:
    """
    Create batches given an input list of SMILES or list of list of SMILES.

    Args:
        smiles_list: list of SMILES
        batch_size: number of samples in batch
    Returns:
        batched SMILES in a list
    """
    n_samples = len(smiles_lst)
    n_batches = int(np.ceil(n_samples / batch_size))

    batched_smiles = []
    for i_batch in range(n_batches):
        if i_batch != n_batches - 1:
            batched_smiles.append(smiles_lst[i_batch * batch_size : (i_batch + 1) * batch_size])
        else:
            batched_smiles.append(smiles_lst[i_batch * batch_size : :])
    return batched_smiles


def convert_to_input_format(
    sampled_smiles: List[List[str]],
    target_smiles: List[List[str]],
    sampled_data_params: Dict[str, Any],
    n_chunks: int = 1,
) -> Tuple[List[np.ndarray], List[List[str]]]:
    """
    Converting sampled data to original input format such that,
    sampled_smiles: [n_batches, batch_size, n_beams],
    target_smiles: [n_batches, batch_size, 1].

    Args:
        sampled_smiles: SMILES sampled in round-trip inference
        target_smiles: target SMILES (ground truth product)
        sampled_data_params: parameters of the input data from backward predictions
            (batch_size, beam_size, n_samples)
    Returns:
        Reshaped round-trip predictions.
    """
    batch_size = sampled_data_params["batch_size"]
    n_beams = sampled_data_params["beam_size"]
    n_samples = sampled_data_params["n_samples"]

    sampled_smiles = np.array(sampled_smiles)
    target_smiles = np.array(target_smiles)

    sampled_smiles = np.reshape(sampled_smiles, (-1, n_beams))
    target_smiles = np.reshape(target_smiles, (-1, n_beams))

    if n_chunks == 1:
        assert target_smiles.shape[0] == n_samples

    # Sanity-check that target smiles are the same within beams
    for tgt_beams in target_smiles:
        assert np.all(tgt_beams == tgt_beams[0])

    # Extract the target smiles for each original sample
    target_smiles = [tgt_smi[0] for tgt_smi in target_smiles]

    smpl_smiles_reform = batchify(sampled_smiles, batch_size)
    tgt_smiles_reform = batchify(target_smiles, batch_size)

    return smpl_smiles_reform, tgt_smiles_reform


def set_output_files(args, chemformer):
    if args.output_score_data and args.output_sampled_smiles:
        for callback in chemformer.trainer.callbacks:
            if hasattr(callback, "set_output_files"):
                callback.set_output_files(args.output_score_data, args.output_sampled_smiles)

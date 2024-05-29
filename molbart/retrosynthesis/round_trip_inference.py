"""Module for running round-trip inference and accuracy scoring of backward predictions
using a forward Chemformer model"""
import subprocess
import tempfile
from argparse import Namespace
from typing import Any, Dict, List, Tuple

import hydra
import pandas as pd
import pytorch_lightning as pl

import molbart.utils.data_utils as util
from molbart.models import Chemformer
from molbart.retrosynthesis import round_trip_utils as rt_utils


def create_round_trip_dataset(args: Namespace) -> Tuple[Namespace, Dict[str, Any]]:
    """
    Reading sampled smiles and creating dataframe on synthesis-datamodule format.

    Args:
        args: Input arguments with parameters for Chemformer, data paths etc.
    Returns:
        updated arguments and input-data metadata dictionary
    """
    print("Creating input data from sampled predictions.")

    _, round_trip_input_data = tempfile.mkstemp(suffix=".csv")

    input_data = pd.read_csv(args.input_data, sep="\t")
    input_data = input_data.iloc[input_data["set"].values == args.dataset_part]

    target_column = args.target_column

    input_targets = input_data[target_column].values

    predicted_data = pd.read_json(args.backward_predictions, orient="table")

    batch_size = len(predicted_data["sampled_molecules"].values[0])
    n_samples = sum([len(batch_smiles) for batch_smiles in predicted_data["sampled_molecules"].values])
    n_beams = len(predicted_data["sampled_molecules"].values[0][0])

    sampled_data_params = {
        "n_samples": n_samples,
        "beam_size": n_beams,
        "batch_size": batch_size,
        "round_trip_input_data": round_trip_input_data,
    }

    counter = 0
    sampled_smiles = []
    target_smiles = []
    # Unravel predictions
    for batch_smiles in predicted_data["sampled_molecules"].values:
        for top_n_smiles in batch_smiles:
            sampled_smiles.extend(top_n_smiles)
            target_smiles.extend([input_targets[counter] for _ in range(n_beams)])
            counter += 1

    input_data = pd.DataFrame(
        {
            "reactants": sampled_smiles,
            "products": target_smiles,
            "set": len(target_smiles) * ["test"],
        }
    )

    print(f"Writing data to temporary file: {round_trip_input_data}")
    input_data.to_csv(round_trip_input_data, sep="\t", index=False)

    args.data_path = round_trip_input_data
    return args, sampled_data_params


def _run_test_callbacks(chemformer: Chemformer, metrics_scores: List[Dict[str, Any]]) -> None:
    """Run callback.on_test_batch_end on all (scoring) callbacks."""
    for batch_idx, scores in enumerate(metrics_scores):
        for callback in chemformer.trainer.callbacks:
            if not isinstance(callback, pl.callbacks.progress.ProgressBar):
                callback.on_test_batch_end(chemformer.trainer, chemformer.model, scores, {}, batch_idx, 0)


@hydra.main(version_base=None, config_path="../config", config_name="round_trip_inference")
def main(args) -> None:
    util.seed_everything(args.seed)

    args, sampled_data_params = create_round_trip_dataset(args)
    chemformer = Chemformer(args)
    rt_utils.set_output_files(args, chemformer)

    print("Running round-trip inference.")
    sampled_smiles, log_lhs, target_smiles = chemformer.predict()

    # Reformat on original shape [n_batches, batch_size, n_beams]
    sampled_smiles, target_smiles = rt_utils.convert_to_input_format(
        sampled_smiles, target_smiles, sampled_data_params, args.n_chunks
    )

    metrics = rt_utils.compute_round_trip_accuracy(chemformer, sampled_smiles, target_smiles)
    _run_test_callbacks(chemformer, metrics)

    print(f"Removing temporary file: {sampled_data_params['round_trip_input_data']}")
    subprocess.check_output(["rm", sampled_data_params["round_trip_input_data"]])
    print("Round-trip inference done!")
    return


if __name__ == "__main__":
    main()

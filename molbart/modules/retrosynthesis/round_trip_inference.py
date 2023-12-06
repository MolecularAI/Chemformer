"""Module for running round-trip inference and accuracy scoring of backward predictions 
using a forward Chemformer model"""
import argparse
import subprocess
from argparse import Namespace
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

import molbart.modules.util as util
from molbart.models import Chemformer
from molbart.modules.retrosynthesis import round_trip_utils as rt_utils

DEFAULT_BATCH_SIZE = 120
DEFAULT_VOCAB_PATH = "bart_vocab_downstream.json"
NUM_BEAMS = 1
DATASET_TYPE = "synthesis"
TASK = "forward_prediction"


def create_round_trip_dataset(args: Namespace) -> Tuple[Namespace, Dict[str, Any]]:
    """
    Reading sampled smiles and creating dataframe on synthesis-datamodule format.

    Args:
        args: Input arguments with parameters for Chemformer, data paths etc.
    Returns:
        updated arguments and input-data metadata dictionary
    """
    print("Creating input data from sampled predictions.")
    out_directory = "/".join(args.output_score_data.split("/")[:-1])
    if out_directory:
        out_directory += +"/"
    round_trip_input_data = out_directory + "tmp_round_trip_input.csv"

    input_data = pd.read_csv(args.input_data, sep="\t")
    input_data = input_data.iloc[input_data["set"].values == args.dataset_part]

    sampled_column = "reactants"
    target_column = "products"

    input_targets = input_data[target_column].values

    predicted_data = pd.read_json(args.backward_predictions, orient="table")

    batch_size = len(predicted_data["sampled_molecules"].values[0])
    n_samples = sum(
        [
            len(batch_smiles)
            for batch_smiles in predicted_data["sampled_molecules"].values
        ]
    )
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
            sampled_column: sampled_smiles,
            target_column: target_smiles,
            "set": len(target_smiles) * ["test"],
        }
    )

    print(f"Writing data to temporary file: {round_trip_input_data}")
    input_data.to_csv(round_trip_input_data, sep="\t", index=False)

    args.data_path = round_trip_input_data
    args.dataset_type = DATASET_TYPE
    args.task = TASK
    return args, sampled_data_params


def main(args: Namespace) -> None:
    util.seed_everything(1)

    args, sampled_data_params = create_round_trip_dataset(args)
    model_args, data_args = util.get_chemformer_args(args)

    kwargs = {
        "vocabulary_path": args.vocabulary_path,
        "n_gpus": args.n_gpus,
        "model_path": args.model_path,
        "model_args": model_args,
        "data_args": data_args,
        "n_beams": NUM_BEAMS,
        "train_mode": "eval",
    }

    chemformer = Chemformer(**kwargs)

    print("Running round-trip inference.")
    sampled_smiles, log_lhs, target_smiles = chemformer.predict(
        dataset=args.dataset_part, i_chunk=args.i_chunk, n_chunks=args.n_chunks
    )

    # Reformat on original shape [n_batches, batch_size, n_beams]
    sampled_smiles, target_smiles = rt_utils.convert_to_input_format(
        sampled_smiles, target_smiles, sampled_data_params
    )

    metrics_df = rt_utils.compute_round_trip_accuracy(
        chemformer, sampled_smiles, target_smiles
    )

    print(f"Removing temporary file: {sampled_data_params['round_trip_input_data']}")
    subprocess.check_output(["rm", sampled_data_params["round_trip_input_data"]])

    metrics_df.to_csv(args.output_score_data, index=False, sep="\t")
    print(f"Metrics written to file: {args.output_score_data}")

    if args.output_sampled_smiles is not None:
        # Create dataframe on the same format
        sampled_df = pd.DataFrame(
            np.array(sampled_smiles, dtype="object"), columns=["sampled_molecules"]
        )
        sampled_df["target_smiles"] = target_smiles

        sampled_df.to_json(args.output_sampled_smiles, orient="table")
        print(f"Sampled smiles written to file: {args.output_sampled_smiles}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Program level args
    parser.add_argument(
        "-o",
        "--output_score_data",
        help="Path to .csv file to which model score results should be written.",
    )
    parser.add_argument(
        "-os",
        "--output_sampled_smiles",
        help="Path to .json file to which sampled smiles should be written.",
    )
    parser.add_argument("-f", "--input_data")
    parser.add_argument("-p", "--backward_predictions")
    parser.add_argument("--model_path")
    parser.add_argument("--dataset_part", default="test")
    parser.add_argument("--vocabulary_path", default=DEFAULT_VOCAB_PATH)

    # Model args
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_seq_len", type=int, default=util.DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--n_chunks", type=int, default=1)
    parser.add_argument("-i", "--i_chunk", type=int, default=0)

    args = parser.parse_args()
    main(args)

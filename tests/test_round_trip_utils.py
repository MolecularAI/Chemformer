import os

import pandas as pd
import pytest

from molbart.retrosynthesis.round_trip_inference import (
    create_round_trip_dataset,
)
from molbart.retrosynthesis.round_trip_utils import (
    compute_round_trip_accuracy,
    convert_to_input_format,
)


def test_create_round_trip_dataset(round_trip_namespace_args, round_trip_params):
    _, sampled_data_params_test = create_round_trip_dataset(round_trip_namespace_args)

    assert sampled_data_params_test["n_samples"] == round_trip_params["n_samples"]
    assert sampled_data_params_test["beam_size"] == round_trip_params["beam_size"]
    assert sampled_data_params_test["batch_size"] == round_trip_params["batch_size"]

    input_data_ground_truth = pd.read_csv(round_trip_params["round_trip_input_data"], sep="\t")
    created_data = pd.read_csv(sampled_data_params_test["round_trip_input_data"], sep="\t")
    assert created_data.equals(input_data_ground_truth)
    os.remove(sampled_data_params_test["round_trip_input_data"])


def test_convert_round_trip_to_input_format(round_trip_raw_prediction_data, round_trip_params):
    round_trip_predictions = round_trip_raw_prediction_data["sampled_smiles"]
    round_trip_targets = round_trip_raw_prediction_data["target_smiles"]

    sampled_smiles, target_smiles = convert_to_input_format(
        round_trip_predictions, round_trip_targets, round_trip_params
    )

    assert len(sampled_smiles) == len(target_smiles)

    n_batches = len(sampled_smiles)
    batch_size, beam_size = sampled_smiles[0].shape

    assert n_batches == 1
    assert batch_size == 3
    assert beam_size == 5


def test_compute_round_trip_accuracy(model_batch_setup, round_trip_converted_prediction_data):
    sampled_smiles = round_trip_converted_prediction_data["sampled_smiles"]
    target_smiles = round_trip_converted_prediction_data["target_smiles"]

    chemformer = model_batch_setup["chemformer"]
    metrics = compute_round_trip_accuracy(chemformer, sampled_smiles, target_smiles)

    assert len(metrics) == 1
    assert round(metrics[0]["accuracy_top_1"], 4) == 0.6667
    assert round(metrics[0]["accuracy_top_3"], 4) == 0.6667
    assert round(metrics[0]["accuracy_top_5"], 4) == 0.6667

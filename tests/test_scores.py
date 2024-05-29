import pytest
import pandas as pd

import omegaconf as oc

from molbart.utils.scores import (
    TanimotoSimilarityScore,
    TopKAccuracyScore,
    FractionInvalidScore,
    FractionUniqueScore,
    ScoreCollection,
)
from molbart.utils import trainer_utils


def test_default_inference_scoring():
    config = oc.OmegaConf.load("molbart/config/inference_score.yaml")
    score_config = config.get("scorers")
    scorers = trainer_utils.instantiate_scorers(score_config)

    scorer_names = set(scorers.names())
    expected = set(["top_k_accuracy", "fraction_invalid", "top1_tanimoto_similarity", "fraction_unique"])

    assert scorer_names.issubset(expected) and expected.issubset(scorer_names)

    sampled_smiles = [["C!O", "CCO", "CCO"], ["c1ccccc1", "c1cccc1", "c1ccccc1"]]
    target_smiles = ["CCO", "c1ccccc1"]

    metrics_scores = scorers.score(sampled_smiles, target_smiles)
    assert round(metrics_scores["fraction_invalid"], 4) == 0.3333
    assert round(metrics_scores["fraction_unique"], 4) == 0.3333
    assert metrics_scores["top1_tanimoto_similarity"] == 1.0
    assert metrics_scores["accuracy_top_1"] == 0.5
    assert metrics_scores["accuracy_top_3"] == 1.0


@pytest.mark.parametrize(
    ("sampled_smiles", "target_smiles", "expected_score"),
    [
        ([["CCO", "CCO", "CCO"], ["c1cc!ccc1", "c1cccc1", "c1ccccc1"]], ["CCO", "c1ccccc1"], 0.3333),
        ([["CCO", "CCO", "CCO"], ["c1ccccc1", "c1cccc1", "c1ccccc1"]], ["CCO", "c1ccccc1"], 0.1667),
        ([["CCO", "C!O", "CCO"], ["c1ccccc1", "c1cccc1", "c1ccccc1"]], ["CCO", "c1ccccc1"], 0.3333),
        ([["CCO", "CCO", "CCO"], ["c1ccccc1", "c1ccccc1", "c1ccccc1"]], ["CCO", "c1ccccc1"], 0.0),
    ],
)
def test_fraction_invalid(sampled_smiles, target_smiles, expected_score):
    scorer = ScoreCollection()
    scorer.load(FractionInvalidScore())

    score = scorer.score(sampled_smiles, target_smiles)["fraction_invalid"]
    assert round(score, 4) == expected_score


@pytest.mark.parametrize(
    ("sampled_smiles", "target_smiles", "expected_score"),
    [
        ([["CCO", "CCO", "CCO"], ["c1cc!ccc1", "c1cccc1", "c1ccccc1"]], ["CCO", "c1ccccc1"], 0.3333),
        ([["CCO", "CCO", "CCO"], ["c1ccccc1", "c1cccc1", "c1ccccc1"]], ["CCO", "c1ccccc1"], 0.3333),
        ([["CCO", "C!O", "COO"], ["c1ccccc1", "c1cccc1", "c1cc(Br)ccc1"]], ["CCO", "c1ccccc1"], 0.6667),
    ],
)
def test_fraction_unique(sampled_smiles, target_smiles, expected_score):
    scorer = ScoreCollection()
    scorer.load(FractionUniqueScore())

    score = scorer.score(sampled_smiles, target_smiles)["fraction_unique"]
    print(round(score, 4))
    assert round(score, 4) == expected_score


def test_accuracy_similarity(round_trip_converted_prediction_data):
    scorer = ScoreCollection()
    scorer.load(TanimotoSimilarityScore(statistics="all"))
    scorer.load(TopKAccuracyScore())

    sampled_smiles = round_trip_converted_prediction_data["sampled_smiles"]
    target_smiles = round_trip_converted_prediction_data["target_smiles"]

    metrics_out = []
    for sampled_batch, target_batch in zip(sampled_smiles, target_smiles):
        metrics = scorer.score(
            sampled_batch,
            target_batch,
        )

        metrics = {key: [val] for key, val in metrics.items()}
        metrics_out.append(pd.DataFrame(metrics))
    metrics_df = pd.concat(metrics_out, axis=0)

    assert all(sim == 1.0 for sim in metrics_df["top1_tanimoto_similarity"].values[0][0])
    assert round(metrics_df["accuracy_top_1"].values[0], 4) == 0.6667
    assert round(metrics_df["accuracy_top_3"].values[0], 4) == 0.6667
    assert round(metrics_df["accuracy_top_5"].values[0], 4) == 0.6667

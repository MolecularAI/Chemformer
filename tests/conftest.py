import pathlib
from argparse import Namespace

import numpy as np
import omegaconf as oc
import pandas as pd
import pytest

import molbart.utils.data_utils as util
from molbart.models import Chemformer
from molbart.data import SynthesisDataModule
from molbart.utils.tokenizers import ChemformerTokenizer, SpanTokensMasker


@pytest.fixture
def example_tokens():
    return [
        ["^", "C", "(", "=", "O", ")", "unknown", "&"],
        ["^", "C", "C", "<SEP>", "C", "Br", "&"],
    ]


@pytest.fixture
def regex_tokens():
    regex = r"\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]"
    return regex.split("|")


@pytest.fixture
def smiles_data():
    return ["CCO.Ccc", "CCClCCl", "C(=O)CBr"]


@pytest.fixture
def mock_random_choice(mocker):
    class ToggleBool:
        def __init__(self):
            self.state = True

        def __call__(self, *args, **kwargs):
            states = []
            for _ in range(kwargs["k"]):
                states.append(self.state)
                self.state = not self.state
            return states

    mocker.patch("molbart.utils.tokenizers.tokenizers.random.choices", side_effect=ToggleBool())


@pytest.fixture
def setup_tokenizer(regex_tokens, smiles_data):
    def wrapper(tokens=None):
        return ChemformerTokenizer(smiles=smiles_data, tokens=tokens, regex_token_patterns=regex_tokens)

    return wrapper


@pytest.fixture
def setup_masker(setup_tokenizer):
    def wrapper(cls=SpanTokensMasker):
        tokenizer = setup_tokenizer()
        return tokenizer, cls(tokenizer)

    return wrapper


@pytest.fixture
def round_trip_params(shared_datadir):
    params = {
        "n_samples": 3,
        "beam_size": 5,
        "batch_size": 3,
        "round_trip_input_data": shared_datadir / "round_trip_input_data.csv",
    }
    return params


@pytest.fixture
def round_trip_namespace_args(shared_datadir):
    args = Namespace()
    args.input_data = shared_datadir / "example_data_uspto.csv"
    args.backward_predictions = shared_datadir / "example_data_backward_sampled_smiles_uspto50k.json"
    args.output_score_data = "temp_metrics.csv"
    args.dataset_part = "test"
    args.working_directory = "tests"
    args.target_column = "products"
    return args


@pytest.fixture
def round_trip_raw_prediction_data(shared_datadir):
    round_trip_df = pd.read_json(shared_datadir / "round_trip_predictions_raw.json", orient="table")
    round_trip_predictions = [np.array(smiles_lst) for smiles_lst in round_trip_df["round_trip_smiles"].values]
    data = {
        "sampled_smiles": round_trip_predictions,
        "target_smiles": round_trip_df["target_smiles"].values,
    }
    return data


@pytest.fixture
def round_trip_converted_prediction_data(shared_datadir):
    round_trip_df = pd.read_json(shared_datadir / "round_trip_predictions_converted.json", orient="table")
    round_trip_predictions = [np.array(smiles_lst) for smiles_lst in round_trip_df["round_trip_smiles"].values]
    data = {
        "sampled_smiles": round_trip_predictions,
        "target_smiles": round_trip_df["target_smiles"].values,
    }
    return data


@pytest.fixture
def model_batch_setup(round_trip_namespace_args):
    config = oc.OmegaConf.load("molbart/config/round_trip_inference.yaml")
    data = pd.read_csv(round_trip_namespace_args.input_data, sep="\t")

    config.d_model = 4
    config.batch_size = 3
    config.n_beams = 3
    config.n_layers = 1
    config.n_heads = 2
    config.d_feedforward = 2
    config.task = "forward_prediction"

    config.datamodule = None
    config.vocabulary_path = "bart_vocab_downstream.json"
    config.n_gpus = 0
    config.device = "cpu"
    config.data_device = "cpu"

    chemformer = Chemformer(config)

    datamodule = SynthesisDataModule(
        reactants=data["reactants"].values,
        products=data["products"].values,
        dataset_path="",
        tokenizer=chemformer.tokenizer,
        batch_size=config.batch_size,
        max_seq_len=util.DEFAULT_MAX_SEQ_LEN,
        reverse=False,
    )

    datamodule.setup()
    dataloader = datamodule.full_dataloader()
    batch_idx, batch_input = next(enumerate(dataloader))

    output_data = {
        "chemformer": chemformer,
        "tokenizer": chemformer.tokenizer,
        "batch_idx": batch_idx,
        "batch_input": batch_input,
        "max_seq_len": util.DEFAULT_MAX_SEQ_LEN,
    }
    return output_data

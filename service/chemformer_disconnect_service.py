import os
from typing import Dict, List

import numpy as np
import omegaconf as oc
from fastapi import FastAPI
from service_utils import get_predictions

from molbart.models import Chemformer
from molbart.retrosynthesis.disconnection_aware import utils
from molbart.retrosynthesis.disconnection_aware.disconnection_atom_mapper import (
    DisconnectionAtomMapper,
)

app = FastAPI()

# Container for data, classes that can be loaded upon startup of the REST API

config = oc.OmegaConf.load("../molbart/config/predict.yaml")

config.batch_size = 64
config.n_gpus = 1

config.model_path = os.environ["CHEMFORMER_DISCONNECTION_MODEL"]
config.model_type = "bart"
config.n_beams = 10
config.n_unique_beams = 10 # Make sure we output unique predictions

config.task = os.environ["CHEMFORMER_TASK"]

config.vocabulary_path = os.environ["CHEMFORMER_VOCAB"]
config.datamodule = None

CONDA_PATH = None
RXNUTILS_ENV_PATH = None

if "CONDA_PATH" in os.environ:
    CONDA_PATH = os.environ["CONDA_PATH"]

if "RXNUTILS_ENV_PATH" in os.environ:
    RXNUTILS_ENV_PATH = os.environ["RXNUTILS_ENV_PATH"]

MODELS = {
    "chemformer_disconnect": Chemformer(config),
    "atom_mapper": DisconnectionAtomMapper(),
}

def _get_n_predictions(predicted_reactants: List[List[str]]):
    return [len(smiles_list) for smiles_list in predicted_reactants]


def _reshape(smiles_list: List[str], n_predictions: List[int]):
    reshaped_smiles_list = []
    counter = 0
    for n_pred in n_predictions:
        all_predictions = [smiles for smiles in smiles_list[counter : counter + n_pred]]
        counter += n_pred
        reshaped_smiles_list.append(all_predictions)
    return reshaped_smiles_list


@app.post("/chemformer-disconnect-api/predict-disconnection")
def predict_disconnection(smiles_list: List[str], bonds_list: List[List[int]], n_beams: int = 10) -> List[Dict]:
    """
    Make prediction with disconnection-Chemformer given list of input SMILES and
    corresponding list of bonds to break [one bond per input SMILES].
    Returns the basic predictions and input product (with new atom-mapping)
    for each bond in each product. Tailored to the multi-step disconnection
    approach in aizynthfinder.

    Args:
        smiles_list: batch of input SMILES to model
        bonds: list of bonds to break for each input SMILES (one bond per molecule)
        n_beams: number of beams in beam search
    """
    # Get input SMILES to the prediction and tag SMILES using the corresponding bonds
    # for that input.
    smiles_atom_map_tagged = [
        MODELS["atom_mapper"].tag_current_bond(smiles, bond_atom_inds)
        for smiles, bond_atom_inds in zip(smiles_list, bonds_list)
    ]

    smiles_tagged_list = utils.get_model_input(
        smiles_atom_map_tagged,
        rxnutils_env_path=RXNUTILS_ENV_PATH,
        conda_path=CONDA_PATH,
    )

    output = []
    predicted_smiles, log_lhs, _ = get_predictions(MODELS["chemformer_disconnect"], smiles_tagged_list, n_beams)
    n_predictions = _get_n_predictions(predicted_smiles)

    # Get atom-mapping of predicted reaction
    mapped_rxns, _ = MODELS["atom_mapper"].predictions_atom_mapping(smiles_list, predicted_smiles)

    reactants_mapped = np.array([mapped_rxn.split(">")[0] for mapped_rxn in mapped_rxns])
    product_new_mapping = np.array([mapped_rxn.split(">")[-1] for mapped_rxn in mapped_rxns])

    output = []
    for item_pred, item_lhs, item_smiles, item_mapped_product, item_bond in zip(
        _reshape(reactants_mapped, n_predictions),
        log_lhs,
        smiles_list,
        _reshape(product_new_mapping, n_predictions),
        bonds_list,
    ):
        output.append(
            {
                "input": item_smiles,
                "output": list(item_pred),
                "lhs": [float(val) for val in item_lhs],
                "product_new_mapping": list(item_mapped_product),
                "current_bond": item_bond,
            }
        )

    return output


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "chemformer_disconnect_service:app",
        host="0.0.0.0",
        port=8023,
        log_level="info",
        reload=False,
    )

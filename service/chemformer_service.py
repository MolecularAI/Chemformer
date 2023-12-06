import os
import numpy as np
from argparse import Namespace
from typing import List, Tuple

from fastapi import FastAPI

from service_utils import (
    get_predictions,
    calculate_llhs, 
    estimate_compound_llhs
)
import molbart.modules.util as util
from molbart.models import Chemformer

app = FastAPI()

# Container for data, classes that can be loaded upon startup of the REST API

args = Namespace()
args.batch_size = 64
args.n_gpus = 1
args.model_path = os.environ["CHEMFORMER_MODEL"]
args.model_type = "bart"
args.n_beams = 10
args.task = os.environ["CHEMFORMER_TASK"]
model_args, data_args = util.get_chemformer_args(args)

kwargs = {
    "vocabulary_path": os.environ["CHEMFORMER_VOCAB"],
    "n_gpus": args.n_gpus,
    "model_path": args.model_path,
    "model_args": model_args,
    "data_args": data_args,
    "n_beams": args.n_beams,
    "train_mode": "eval",
    "datamodule_type": None,
}

global_items = {"chemformer": Chemformer(**kwargs)}


@app.post("/chemformer-api/predict")
def predict(smiles_list: List[str], n_beams: int = 10):

    smiles, log_lhs, original_smiles = get_predictions(
        global_items["chemformer"], smiles_list, n_beams
    )

    output = []
    for item_pred, item_lhs, item_smiles in zip(smiles, log_lhs, original_smiles):
        output.append(
            {
                "input": item_smiles,
                "output": list(item_pred),
                "lhs": [float(val) for val in item_lhs],
            }
        )
    return output


@app.post("/chemformer-api/log_likelihood")
def log_likelihood(reactants: List[str], products: List[str]):

    log_lhs = calculate_llhs(
        global_items["chemformer"], 
        reactants, 
        products
    )
    
    output = []
    for prod_smi, react_smi, llhs in zip(products, reactants, log_lhs):
        output.append(
            {
                "product_smiles": str(prod_smi),
                "reactant_smiles": str(react_smi),
                "log_likelihood": float(llhs),
            }
        )
    return output

@app.post("/chemformer-api/compound_log_likelihood")
def compound_log_likelihood(reactants: List[str], products: List[str], 
                            n_augments: int = 10):

    log_lhs = estimate_compound_llhs(
        global_items["chemformer"], 
        reactants, 
        products, 
        n_augments=n_augments
    )
    
    output = []
    for prod_smi, react_smi, llhs in zip(products, reactants, log_lhs):
        output.append(
            {
                "product_smiles": str(prod_smi),
                "reactant_smiles": str(react_smi),
                "log_likelihood": float(llhs),
            }
        )
    return output


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "chemformer_service:app",
        host="0.0.0.0",
        port=8003,
        log_level="info",
        reload=False,
    )

import os
from argparse import Namespace
import numpy as np
from typing import List, Tuple

from pysmilesutils.augment import SMILESAugmenter
from molbart.modules.data.seq2seq_data import InMemorySynthesisDataModule

import molbart.modules.util as util
from molbart.models import Chemformer

def calculate_llhs(chemformer: Chemformer,  reactants: List[str], 
                  products: List[str]) -> List[float]:
    
    datamodule = InMemorySynthesisDataModule(
        reactants=reactants,
        products=products,
        dataset_path="",
        tokenizer=chemformer.tokenizer,
        batch_size=chemformer.data_args.batch_size,
        max_seq_len=util.DEFAULT_MAX_SEQ_LEN,
        augment_prob=False,        
        reverse=not chemformer.data_args.forward_prediction,
    )

    datamodule.setup()
    llhs = chemformer.log_likelihood(dataloader=datamodule.full_dataloader())
    return llhs 
    

def estimate_compound_llhs(
        chemformer: Chemformer, 
        reactants: List[str], 
        products: List[str], 
        n_augments: int = 10
    ) -> Tuple[np.array, np.array]:

    augmenter = SMILESAugmenter()
    base_log_likelihoods = np.array(
        calculate_llhs(chemformer, reactants, products)
    )

    all_llhs = []
    all_llhs.append(base_log_likelihoods[:, np.newaxis])

    for _ in range(n_augments-1):
        if chemformer.data_args.forward_prediction:
            this_products = augmenter(products)
            this_reactants = reactants
        else:
            this_products = products
            this_reactants = augmenter(reactants)
        
        aug_log_likelihoods = np.array(
            calculate_llhs(chemformer, this_reactants, this_products)
        )

        all_llhs.append(aug_log_likelihoods[:, np.newaxis])
    
    best_log_likelihoods = np.concatenate(all_llhs, axis=1)
    best_log_likelihoods = np.max(best_log_likelihoods, axis=1)

    return best_log_likelihoods

def get_predictions(chemformer: Chemformer, smiles_list: List[str], n_beams: int = 10):

    # Setting both reactants and products to smiles_list since we do 
    # not have the "ground truth" data.
    datamodule = InMemorySynthesisDataModule(
        reactants=smiles_list,
        products=smiles_list,
        tokenizer=chemformer.tokenizer,
        batch_size=chemformer.data_args.batch_size,
        max_seq_len=util.DEFAULT_MAX_SEQ_LEN,
    )
    datamodule.setup()

    chemformer.model.n_unique_beams = n_beams
    chemformer.model.num_beams = n_beams
    smiles, log_lhs, original_smiles = chemformer.predict(
        dataloader=datamodule.full_dataloader()
    )
    return smiles, log_lhs, original_smiles
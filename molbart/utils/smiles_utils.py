from typing import List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem


def canonicalize_smiles(smiles: str) -> str:
    """
    Canonicalize smiles and sort the (possible) multiple molcules.

    Args:
        smiles: Input SMILES string.
    Returns:
        Canonicalized SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return smiles

    smiles_canonical = Chem.MolToSmiles(mol)
    smiles_canonical = ".".join(sorted(smiles_canonical.split(".")))
    return smiles_canonical


def inchi_key(smiles: str):
    """
    Get inchi key of input SMILES.

    Args:
        smiles: Input SMILES string
    Returns:
        Inchi-key of SMILES string or SMILES string if invalid rdkit molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return smiles
    return Chem.MolToInchiKey(mol)


def uniqueify_sampled_smiles(
    sampled_smiles: List[np.ndarray],
    log_lhs: List[np.ndarray],
    n_unique_beams: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Get unique SMILES and corresponding highest log-likelihood of each input.
    For beam_size > 1: Uniqueifying sampled molecules and select
    'n_unique_beams'-top molecules.

    Args:
        sampled_smiles: list of top-k sampled SMILES
        log_lhs: list of top-k log-likelihoods
        n_unique_beams: upper limit on number of unique SMILES to return
    Returns:
        Tuple of lists with unique SMILES and their corresponding highest
        log-likelihoods.
    """
    sampled_smiles_unique = []
    log_lhs_unique = []
    for top_k_smiles, top_k_llhs in zip(sampled_smiles, log_lhs):
        top_k_mols = [Chem.MolFromSmiles(smi) for smi in top_k_smiles]
        top_k_smiles = [Chem.MolToSmiles(mol) for mol in top_k_mols if mol]
        top_k_llhs = [llhs for llhs, mol in zip(top_k_llhs, top_k_mols) if mol]
        top_k_mols = [mol for mol in top_k_mols if mol]

        top_k_unique = pd.DataFrame(
            {
                "smiles": top_k_smiles,
                "log_likelihood": top_k_llhs,
                "molecules": top_k_mols,
            }
        )
        top_k_unique.drop_duplicates(subset=["smiles"], keep="first", inplace=True)

        sampled_smiles_unique.append(list(top_k_unique["smiles"].values[0:n_unique_beams]))
        log_lhs_unique.append(list(top_k_unique["log_likelihood"].values[0:n_unique_beams]))

    return (
        sampled_smiles_unique,
        log_lhs_unique,
    )

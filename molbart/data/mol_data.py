""" Module containing classes for loading molecular data"""
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from rdkit import Chem

from molbart.data.base import MoleculeListDataModule


class ChemblDataModule(MoleculeListDataModule):
    """
    DataModule for Chembl dataset.

    The molecules and the lengths of the sequences
    are loaded from a pickled DataFrame
    """

    def _load_all_data(self) -> None:
        df = pd.read_pickle(self.dataset_path)
        self._all_data = {
            "molecules": df["molecules"].tolist(),
            "lengths": df["lengths"].tolist(),
        }
        self._set_split_indices_from_dataframe(df)

    def _transform_batch(
        self, batch: List[Dict[str, Any]], train: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        smiles_batch = [{"smiles": Chem.MolToSmiles(item["molecules"])} for item in batch]
        return super()._transform_batch(smiles_batch, train)


class ZincDataModule(MoleculeListDataModule):
    """
    DataModule for Zinc dataset.

    The molecules are read as SMILES from a number of
    csv files.
    """

    def _load_all_data(self) -> None:
        path = Path(self.dataset_path)
        if path.is_dir():
            dfs = [pd.read_csv(filename) for filename in path.iterdir()]
            df = pd.concat(dfs, ignore_index=True, copy=False)
        else:
            df = pd.read_csv(path)
        self._all_data = {"smiles": df["smiles"].tolist()}
        self._set_split_indices_from_dataframe(df)

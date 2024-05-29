""" Module containing the default datamodules"""
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from molbart.data.base import ReactionListDataModule

class SynthesisDataModule(ReactionListDataModule):
    """
    DataModule for forward and backard synthesis prediction.

    The reactions are read from a tab seperated DataFrame .csv file.
    Expects the dataset to contain SMILES in two seperate columns named "reactants" and "products".
    The dataset must also contain a columns named "set" with values of "train", "val" and "test".
    validation column can be named "val", "valid" or "validation".

    Supports both loading data from file, and in-memory prediction.

    All rows that are not test or validation, are assumed to be training samples.
    """

    datamodule_name = "synthesis"

    def __init__(
            self, 
            reactants: Optional[List[str]] = None, 
            products: Optional[List[str]] = None, 
            **kwargs
        ):
        super().__init__(**kwargs)

        self._in_memory = False
        if reactants is not None and products is not None:
            self._in_memory = True
            print("Using in-memory datamodule.")
            self._all_data = {"reactants": reactants, "products": products}

    def __repr__(self):
        return self.datamodule_name

    def _get_sequences(self, batch: List[Dict[str, Any]], train: bool) -> Tuple[List[str], List[str]]:
        reactants = [item["reactants"] for item in batch]
        products = [item["products"] for item in batch]
        if train:
            reactants = self._batch_augmenter(reactants)
            products = self._batch_augmenter(products)
        return reactants, products

    def _load_all_data(self) -> None:
        if self._in_memory:
            return
        
        if self.dataset_path.endswith(".csv"):
            df = pd.read_csv(self.dataset_path, sep="\t").reset_index()
            self._all_data = {
                "reactants": df["reactants"].tolist(),
                "products": df["products"].tolist(),
            }
            self._set_split_indices_from_dataframe(df)
        else:
            super()._load_all_data()
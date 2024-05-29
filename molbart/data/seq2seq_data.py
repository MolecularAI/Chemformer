""" Module containing classes to load seq2seq data"""
import pandas as pd
from rdkit import Chem
from typing import Any, Dict, List, Tuple

from molbart.data.base import ReactionListDataModule


class Uspto50DataModule(ReactionListDataModule):
    """
    DataModule for the USPTO-50 dataset

    The reactions as well as a type token are read from
    a pickled DataFrame
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._include_type_token = kwargs.get("include_type_token", False)

    def _get_sequences(self, batch: List[Dict[str, Any]], train: bool) -> Tuple[List[str], List[str]]:
        reactants = [Chem.MolToSmiles(item["reactants"]) for item in batch]
        products = [Chem.MolToSmiles(item["products"]) for item in batch]

        if train:
            reactants = self._batch_augmenter(reactants)
            products = self._batch_augmenter(products)

        if self._include_type_token and not self.reverse:
            reactants = [item["type_tokens"] + smi for item, smi in zip(batch, reactants)]
        if self._include_type_token and self.reverse:
            products = [item["type_tokens"] + smi for item, smi in zip(batch, products)]

        return reactants, products

    def _load_all_data(self) -> None:
        df = pd.read_pickle(self.dataset_path).reset_index()
        self._all_data = {
            "reactants": df["reactants_mol"].tolist(),
            "products": df["products_mol"].tolist(),
            "type_tokens": df["reaction_type"].tolist(),
        }
        self._set_split_indices_from_dataframe(df)


class UsptoMixedDataModule(ReactionListDataModule):
    """
    DataModule for the USPTO-Mixed dataset

    The reactions are read from a pickled DataFrame
    """

    def _get_sequences(self, batch: List[Dict[str, Any]], train: bool) -> Tuple[List[str], List[str]]:
        reactants = [Chem.MolToSmiles(item["reactants"]) for item in batch]
        products = [Chem.MolToSmiles(item["products"]) for item in batch]
        if train:
            reactants = self._batch_augmenter(reactants)
            products = self._batch_augmenter(products)
        return reactants, products

    def _load_all_data(self) -> None:
        df = pd.read_pickle(self.dataset_path).reset_index()
        self._all_data = {
            "reactants": df["reactants_mol"].tolist(),
            "products": df["products_mol"].tolist(),
        }
        self._set_split_indices_from_dataframe(df)


class UsptoSepDataModule(ReactionListDataModule):
    """
    DataModule for the USPTO-Separated dataset

    The reactants, reagents and products are read from
    a pickled DataFrame
    """

    def _get_sequences(self, batch: List[Dict[str, Any]], train: bool) -> Tuple[List[str], List[str]]:
        reactants = [Chem.MolToSmiles(item["reactants"]) for item in batch]
        reagents = [Chem.MolToSmiles(item["reagents"]) for item in batch]
        products = [Chem.MolToSmiles(item["products"]) for item in batch]

        if train:
            reactants = self._batch_augmenter(reactants)
            reagents = self._batch_augmenter(reagents)
            products = self._batch_augmenter(products)

        reactants = [react_smi + ">" + reag_smi for react_smi, reag_smi in zip(reactants, reagents)]

        return reactants, products

    def _load_all_data(self) -> None:
        df = pd.read_pickle(self.dataset_path).reset_index()
        self._all_data = {
            "reactants": df["reactants_mol"].tolist(),
            "products": df["products_mol"].tolist(),
            "reagents": df["reagents_mol"].tolist(),
        }
        self._set_split_indices_from_dataframe(df)


class MolecularOptimizationDataModule(ReactionListDataModule):
    """
    DataModule for a dataset for molecular optimization

    The input and ouput molecules, as well as a the property
    tokens are read from a pickled DataFrame
    """

    def _get_sequences(self, batch: List[Dict[str, Any]], train: bool) -> Tuple[List[str], List[str]]:
        input_smiles = [Chem.MolToSmiles(item["input_mols"]) for item in batch]
        output_smiles = [Chem.MolToSmiles(item["output_mols"]) for item in batch]

        if train:
            input_smiles = self._batch_augmenter(input_smiles)
            output_smiles = self._batch_augmenter(output_smiles)

        input_smiles = [item["prop_tokens"] + smi for item, smi in zip(batch, input_smiles)]

        return input_smiles, output_smiles

    def _load_all_data(self) -> None:
        df = pd.read_pickle(self.dataset_path).reset_index()
        self._all_data = {
            "prop_tokens": df["property_tokens"].tolist(),
            "input_mols": df["input_mols"].tolist(),
            "output_mols": df["output_mols"].tolist(),
        }
        self._set_split_indices_from_dataframe(df)

import random
import functools
import torch
import pandas as pd
import pytorch_lightning as pl
from rdkit import Chem
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset
from pysmilesutils.augment import MolAugmenter


class _AbsDataset(Dataset):
    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, item):
        raise NotImplementedError()

    def split_idxs(self, val_idxs, test_idxs):
        raise NotImplementedError()

    def split(self, val_perc=0.2, test_perc=0.2):
        """ Split the dataset randomly into three datasets

        Splits the dataset into train, validation and test datasets.
        Validation and test dataset have round(len * <val/test>_perc) elements in each
        """

        split_perc = val_perc + test_perc
        if split_perc > 1:
            msg = f"Percentage of dataset to split must not be greater than 1, got {split_perc}"
            raise ValueError(msg)

        dataset_len = len(self)
        val_len = round(dataset_len * val_perc)
        test_len = round(dataset_len * test_perc)

        val_idxs = random.sample(range(dataset_len), val_len)
        test_idxs = random.sample(range(dataset_len), test_len)

        train_dataset, val_dataset, test_dataset = self.split_idxs(val_idxs, test_idxs)

        return train_dataset, val_dataset, test_dataset


# -------------------------------------------------------------------------------------------------------------
# ------------------------------------------- Downstream Datasets ---------------------------------------------
# -------------------------------------------------------------------------------------------------------------


class ReactionDataset(_AbsDataset):
    def __init__(self, reactants, products, items=None, transform=None, aug_prob=0.0):
        super(ReactionDataset, self).__init__()

        if len(reactants) != len(products):
            raise ValueError(f"There must be an equal number of reactants and products")

        self.reactants = reactants
        self.products = products
        self.items = items
        self.transform = transform
        self.aug_prob = aug_prob
        self.aug = MolAugmenter() 

    def __len__(self):
        return len(self.reactants)

    def __getitem__(self, item):
        reactant = self.reactants[item]
        product = self.products[item]
        output = (reactant, product, self.items[item]) if self.items is not None else (reactant, product)
        output = self.transform(*output) if self.transform is not None else output
        return output

    def split_idxs(self, val_idxs, test_idxs):
        """ Splits dataset into train, val and test

        Note: Assumes all remaining indices outside of val_idxs and test_idxs are for training data
        The datasets are returned as ReactionDataset objects, if these should be a subclass 
        the from_reaction_pairs function should be overidden

        Args:
            val_idxs (List[int]): Indices for validation data
            test_idxs (List[int]): Indices for test data

        Returns:
            (ReactionDataset, ReactionDataset, ReactionDataset): Train, val and test datasets
        """

        # Use aug prob of 0.0 for val and test datasets
        val_reacts = [self.reactants[idx] for idx in val_idxs]
        val_prods = [self.products[idx] for idx in val_idxs]
        val_extra = [self.items[idx] for idx in val_idxs] if self.items is not None else None
        val_dataset = ReactionDataset(val_reacts, val_prods, val_extra, self.transform, 0.0)

        test_reacts = [self.reactants[idx] for idx in test_idxs]
        test_prods = [self.products[idx] for idx in test_idxs]
        test_extra = [self.items[idx] for idx in test_idxs] if self.items is not None else None
        test_dataset = ReactionDataset(test_reacts, test_prods, test_extra, self.transform, 0.0)

        train_idxs = set(range(len(self))) - set(val_idxs).union(set(test_idxs))
        train_reacts = [self.reactants[idx] for idx in train_idxs]
        train_prods = [self.products[idx] for idx in train_idxs]
        train_extra = [self.items[idx] for idx in train_idxs] if self.items is not None else None
        train_dataset = ReactionDataset(train_reacts, train_prods, train_extra, self.transform, self.aug_prob)

        return train_dataset, val_dataset, test_dataset

    def _save_idxs(self, df):
        train_idxs = df.index[df["set"] == "train"]
        val_idxs = df.index[df["set"] == "valid"].tolist()
        test_idxs = df.index[df["set"] == "test"].tolist()

        if len(set(val_idxs).intersection(set(test_idxs))) > 0:
            raise ValueError(f"Val idxs and test idxs overlap")
        if len(set(train_idxs).intersection(set(test_idxs))) > 0:
            raise ValueError(f"Train idxs and test idxs overlap")
        if len(set(train_idxs).intersection(set(val_idxs))) > 0:
            raise ValueError(f"Train idxs and val idxs overlap")

        return train_idxs, val_idxs, test_idxs

    def _augment_to_smiles(self, mol, other_mol=None):
        aug = random.random() < self.aug_prob
        mol_aug = self.aug([mol])[0] if aug else mol
        mol_str = Chem.MolToSmiles(mol_aug, canonical=not aug)
        if other_mol is not None:
            other_mol_aug = self.aug([other_mol])[0] if aug else other_mol
            other_mol_str = Chem.MolToSmiles(other_mol_aug, canonical=not aug)
            return mol_str, other_mol_str

        return mol_str


class Uspto50(ReactionDataset):
    def __init__(self, data_path, aug_prob, type_token=False, forward=True):
        path = Path(data_path)
        df = pd.read_pickle(path)
        reactants = df["reactants_mol"].tolist()
        products = df["products_mol"].tolist()
        type_tokens = df["reaction_type"].tolist()

        super().__init__(reactants, products, items=type_tokens, transform=self._prepare_strings, aug_prob=aug_prob)

        self.type_token = type_token
        self.forward = forward
        self.train_idxs, self.val_idxs, self.test_idxs = self._save_idxs(df)

    def _prepare_strings(self, react, prod, type_token):
        react_str = self._augment_to_smiles(react)
        prod_str = self._augment_to_smiles(prod)

        if self.forward:
            react_str = f"{str(type_token)}{react_str}" if self.type_token else react_str
        else:
            prod_str = f"{str(type_token)}{prod_str}" if self.type_token else prod_str

        return react_str, prod_str


class UsptoMixed(ReactionDataset):
    def __init__(self, data_path, aug_prob):
        path = Path(data_path)
        df = pd.read_pickle(path)
        reactants = df["reactants_mol"].tolist()
        products = df["products_mol"].tolist()

        super().__init__(reactants, products, transform=self._prepare_strings, aug_prob=aug_prob)

        self.aug_prob = aug_prob
        self.train_idxs, self.val_idxs, self.test_idxs = self._save_idxs(df)

    def _prepare_strings(self, react, prod):
        react_str = self._augment_to_smiles(react)
        prod_str = self._augment_to_smiles(prod)
        return react_str, prod_str


class UsptoSep(ReactionDataset):
    def __init__(self, data_path, aug_prob):
        path = Path(data_path)
        df = pd.read_pickle(path)
        reactants = df["reactants_mol"].tolist()
        reagents = df["reagents_mol"].tolist()
        products = df["products_mol"].tolist()

        super().__init__(reactants, products, items=reagents, transform=self._prepare_strings, aug_prob=aug_prob)

        self.aug_prob = aug_prob
        self.sep_token = ">"
        self.train_idxs, self.val_idxs, self.test_idxs = self._save_idxs(df)

    def _prepare_strings(self, react, prod, reag):
        if reag.GetNumAtoms() > 0:
            react_str, reag_str = self._augment_to_smiles(react, other_mol=reag)
        else:
            react_str = self._augment_to_smiles(react)
            reag_str = Chem.MolToSmiles(reag)

        react_str = f"{react_str}{self.sep_token}{reag_str}"
        prod_str = self._augment_to_smiles(prod)
        return react_str, prod_str


class MolOpt(ReactionDataset):
    def __init__(self, data_path, aug_prob):
        path = Path(data_path)
        df = pd.read_pickle(path)
        prop_tokens = df["property_tokens"].tolist()
        input_smiles = df["input_mols"].tolist()
        output_smiles = df["output_mols"].tolist()

        super().__init__(
            input_smiles,
            output_smiles,
            items=prop_tokens,
            transform=self._prepare_strings,
            aug_prob=aug_prob
        )

        self.aug_prob = aug_prob
        self.train_idxs, self.val_idxs, self.test_idxs = self._save_idxs(df)

    def _prepare_strings(self, input_smi, output_smi, prop_tokens):
        input_str = self._augment_to_smiles(input_smi)
        input_str = f"{prop_tokens}{input_str}"
        output_str = self._augment_to_smiles(output_smi)
        return input_str, output_str


# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------- Molecule Datasets ----------------------------------------------
# -------------------------------------------------------------------------------------------------------------


class MoleculeDataset(_AbsDataset):
    def __init__(
        self,
        molecules,
        seq_lengths=None,
        transform=None,
        train_idxs=None,
        val_idxs=None,
        test_idxs=None
    ):
        super(MoleculeDataset, self).__init__()

        self.molecules = molecules
        self.seq_lengths = seq_lengths
        self.transform = transform
        self.train_idxs = train_idxs
        self.val_idxs = val_idxs
        self.test_idxs = test_idxs

    def __len__(self):
        return len(self.molecules)

    def __getitem__(self, item):
        molecule = self.molecules[item]
        if self.transform is not None:
            molecule = self.transform(molecule)

        return molecule

    def split_idxs(self, val_idxs, test_idxs):
        val_mols = [self.molecules[idx] for idx in val_idxs]
        val_lengths = [self.seq_lengths[idx] for idx in val_idxs] if self.seq_lengths is not None else None
        val_dataset = MoleculeDataset(val_mols, val_lengths, self.transform)

        test_mols = [self.molecules[idx] for idx in test_idxs]
        test_lengths = [self.seq_lengths[idx] for idx in test_idxs] if self.seq_lengths is not None else None
        test_dataset = MoleculeDataset(test_mols, test_lengths, self.transform)

        train_idxs = set(range(len(self))) - set(val_idxs).union(set(test_idxs))
        train_mols = [self.molecules[idx] for idx in sorted(train_idxs)]
        train_lengths = [self.seq_lengths[idx] for idx in train_idxs] if self.seq_lengths is not None else None
        train_dataset = MoleculeDataset(train_mols, train_lengths, self.transform)

        return train_dataset, val_dataset, test_dataset


class Chembl(MoleculeDataset):
    def __init__(self, data_path):
        path = Path(data_path)
        df = pd.read_pickle(path)

        molecules = df["molecules"].tolist()
        lengths = df["lengths"].tolist()
        train_idxs, val_idxs, test_idxs = self._save_idxs(df)

        super().__init__(
            molecules,
            seq_lengths=lengths,
            train_idxs=train_idxs,
            val_idxs=val_idxs,
            test_idxs=test_idxs
        )

    def _save_idxs(self, df):
        val_idxs = df.index[df["set"] == "val"].tolist()
        test_idxs = df.index[df["set"] == "test"].tolist()

        idxs_intersect = set(val_idxs).intersection(set(test_idxs))
        if len(idxs_intersect) > 0:
            raise ValueError(f"Val idxs and test idxs overlap")

        idxs = set(range(len(df.index)))
        train_idxs = idxs - set(val_idxs).union(set(test_idxs))

        return train_idxs, val_idxs, test_idxs


class ZincSlice(MoleculeDataset):
    def __init__(self, df):
        smiles = df["smiles"].tolist()
        train_idxs, val_idxs, test_idxs = self._save_idxs(df)

        super().__init__(
            smiles,
            train_idxs=train_idxs,
            val_idxs=val_idxs,
            test_idxs=test_idxs,
            transform=lambda smi: Chem.MolFromSmiles(smi)
        )

    def _save_idxs(self, df):
        val_idxs = df.index[df["set"] == "val"].tolist()
        test_idxs = df.index[df["set"] == "test"].tolist()

        idxs_intersect = set(val_idxs).intersection(set(test_idxs))
        if len(idxs_intersect) > 0:
            raise ValueError(f"Val idxs and test idxs overlap")

        idxs = set(range(len(df.index)))
        train_idxs = idxs - set(val_idxs).union(set(test_idxs))

        return train_idxs, val_idxs, test_idxs


class Zinc(ZincSlice):
    def __init__(self, data_path):
        path = Path(data_path)

        # If path is a directory then read every subfile
        if path.is_dir():
            df = self._read_dir_df(path)
        else:
            df = pd.read_csv(path)

        super().__init__(df)

    def _read_dir_df(self, path):
        dfs = [pd.read_csv(f) for f in path.iterdir()]
        zinc_df = pd.concat(dfs, ignore_index=True, copy=False)
        return zinc_df


class ConcatMoleculeDataset(MoleculeDataset):
    """ Dataset class for storing (concatenated) molecules 

    Automatically constructs a dataset which contains rdkit molecules
    Roughly a third of these molecule objects are single molecules,
    another third contain two molecules and the final third contain three molecules.

    The molecules to be concatenated are randomly selected, 
    so the ordering from the original data is not preserved.
    """

    def __init__(
        self, 
        dataset: MoleculeDataset,
        join_token: Optional[str] = ".",
        double_mol_prob: Optional[float] = 0.333,
        triple_mol_prob: Optional[float] = 0.333
    ):
        self.join_token = join_token
        self.double_mol_prob = double_mol_prob
        self.triple_mol_prob = triple_mol_prob

        self.original_dataset = dataset

        concat_idxs = self._construct_concat_idxs(dataset)

        super(ConcatMoleculeDataset, self).__init__(
            concat_idxs, 
            transform=self._process_molecule_idxs,
            train_idxs=dataset.train_idxs,
            val_idxs=dataset.val_idxs,
            test_idxs=dataset.test_idxs
        )

    def _construct_concat_idxs(self, dataset):
        idxs = list(range(len(dataset)))
        random.shuffle(idxs)

        curr = 0
        molecule_idxs = []

        added_prob = self.double_mol_prob + self.triple_mol_prob
        
        while curr <= len(idxs) - 1:
            rand = random.random()

            # Use two molecules
            if rand < self.double_mol_prob and curr <= len(idxs) - 2:
                curr_idxs = [idxs[curr + i] for i in range(2)]
                molecule_idxs.append(curr_idxs)
                curr += 2

            # Or, Use three molecules together
            elif rand < added_prob and curr <= len(idxs) - 3:
                curr_idxs = [idxs[curr + i] for i in range(3)]
                molecule_idxs.append(curr_idxs)
                curr += 3

            # Or, take a single molecule
            else:
                curr_idx = idxs[curr]
                molecule_idxs.append([curr_idx])
                curr += 1

        return molecule_idxs

    def _process_molecule_idxs(self, idxs):
        if len(idxs) == 1:
            molecule = self.original_dataset[idxs[0]]
        else:
            molecule = self._concat_mols_from_idxs(idxs, self.original_dataset)

        return molecule

    def _concat_mols_from_idxs(self, idxs, dataset):
        mols = [dataset[idx] for idx in idxs]
        concat_mol = functools.reduce(lambda m1, m2: Chem.CombineMols(m1, m2), mols)
        return concat_mol

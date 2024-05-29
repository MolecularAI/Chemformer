"""Module containing atom-mapping functionality needed to run disconnection-Chemformer"""
import numpy as np
from rdkit import Chem

RXN_MAPPER_ENV_OK = True
try:
    from rxnmapper import RXNMapper
except ImportError:
    RXN_MAPPER_ENV_OK = False

from typing import Dict, List, Sequence, Tuple


class DisconnectionAtomMapper:
    """Class for handling atom-mapping routines of multi-step disconnection-Chemformer"""

    def __init__(self):
        if RXN_MAPPER_ENV_OK:
            self.rxn_mapper = RXNMapper()

    def mapping_to_index(self, mol: Chem.rdchem.Mol) -> Dict[int, int]:
        """
        Atom-map-num to index mapping.

        Args:
            mol: rdkit Molecule
        Returns
            a dictionary which maps atom-map-number to atom-index"""
        mapping = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum()}
        return mapping

    def predictions_atom_mapping(
        self, smiles_list: List[str], predicted_smiles_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create atom-mapping on the predicted reactions using RXN-mapper.
        Requires RXNMapper to be installed.

        Args:
            rxn_mapper: RXN-mapper model
            smiles_list: batch of input product SMILES to predict atom-mapping on
            predicted_smiles_list: batch of predicted reactant SMILES

        Returns:
            the atom-mapped reactions and the mapping confidence
        """
        if not RXN_MAPPER_ENV_OK:
            raise ImportError("rxnmapper has to be installed in the environment!")
        rxn_smiles_list = []
        for product_smiles_mapped, reactants_smiles in zip(smiles_list, predicted_smiles_list):
            product_smiles = self.remove_atom_mapping(product_smiles_mapped)

            rxn_smiles_list.extend(self._reaction_smiles_lst(product_smiles, reactants_smiles))

        mapped_rxns = self.rxn_mapper.get_attention_guided_atom_maps(rxn_smiles_list, canonicalize_rxns=False)

        atom_map_confidences = np.array([rxnmapper_output["confidence"] for rxnmapper_output in mapped_rxns])
        mapped_rxns = np.array([rxnmapper_output["mapped_rxn"] for rxnmapper_output in mapped_rxns])
        return mapped_rxns, atom_map_confidences

    def propagate_input_mapping_to_reactants(
        self,
        product_input_mapping: str,
        predicted_reactants: str,
        product_new_mapping: str,
    ) -> str:
        """
        Propagate old atom-mapping to reactants using the new atom-mapping.

        Args:
            product_input_mapping: input product.
            predicted_reactants: predicted_reactants without atom-mapping.
            product_new_mapping: product with new mapping from rxn-mapper.
        Returns:
            reactant SMILES with the same atom-mapping as the input product.
        """

        product_input_mapping = self._canonicalize_mapped(product_input_mapping)
        product_new_mapping = self._canonicalize_mapped(product_new_mapping)

        mol_input_mapping = Chem.MolFromSmiles(product_input_mapping)
        mol_new_mapping = Chem.MolFromSmiles(product_new_mapping)

        reactants_mol = Chem.MolFromSmiles(predicted_reactants)
        reactants_map_to_index = self.mapping_to_index(reactants_mol)
        predicted_reactants = self.remove_atom_mapping(predicted_reactants, canonical=False)
        reactants_mol = Chem.MolFromSmiles(predicted_reactants)

        for atom_idx, atom_input in enumerate(mol_input_mapping.GetAtoms()):
            atom_new_mapping = mol_new_mapping.GetAtomWithIdx(atom_idx)

            atom_map_num_input = atom_input.GetAtomMapNum()
            atom_map_num_new_mapping = atom_new_mapping.GetAtomMapNum()

            try:
                atom_reactant = reactants_mol.GetAtomWithIdx(reactants_map_to_index[atom_map_num_new_mapping])
                atom_reactant.SetAtomMapNum(atom_map_num_input)
            except KeyError:
                continue

        return Chem.MolToSmiles(reactants_mol)

    def remove_atom_mapping(self, smiles_mapped: str, canonical: bool = True) -> str:
        """
        Remove atom-mapping from SMILES.

        Args:
            smiles_mapped: SMILES with atom-mapping
            canonical: whether to canonicalize the output SMILES
        Returns:
            SMILES without atom-mapping
        """
        mol = Chem.MolFromSmiles(smiles_mapped)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol, canonical=canonical)

    def tag_current_bond(self, product_smiles: str, bond_inds: Sequence[int]) -> str:
        """
        Remove atom-tagging on all atoms except those in bonds_inds.
        Tag bond_inds atoms as [<atom>:1] where <atom> is any atom.

        Args:
            mol: (product) SMILES with atom-mapping
            bond_inds: atom indices involved in current bond to break
        Returns:
            atom-map tagged SMILES
        """
        mol = Chem.MolFromSmiles(product_smiles)

        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() in bond_inds:
                atom.SetAtomMapNum(1)
            else:
                atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol)

    def _canonicalize_mapped(self, smiles_mapped: str) -> str:
        smiles = self.remove_atom_mapping(smiles_mapped, canonical=False)

        mol_mapped = Chem.MolFromSmiles(smiles_mapped)
        mol_unmapped = Chem.MolFromSmiles(smiles)

        _, canonical_atom_order = tuple(
            zip(*sorted([(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(mol_unmapped))]))
        )

        mol_mapped = Chem.RenumberAtoms(mol_mapped, canonical_atom_order)
        return Chem.MolToSmiles(mol_mapped, canonical=False)

    def _reaction_smiles_lst(self, product_smiles: str, reactants_smiles: np.ndarray) -> List[str]:
        """
        Construct the reaction smiles given product and reactant SMILES.

        Args:
            product_smiles: input product SMILES
            reactants_smiles: list of predicted reactant SMILES
        Returns:
            list of reaction SMILES
        """
        rxn_smiles = [f"{reactants}>>{product_smiles}" for reactants in reactants_smiles]
        return rxn_smiles

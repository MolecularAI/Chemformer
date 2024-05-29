import pytest
from rdkit import Chem

from molbart.retrosynthesis.disconnection_aware.disconnection_atom_mapper import (
    DisconnectionAtomMapper,
)
from molbart.retrosynthesis.disconnection_aware.utils import (
    verify_disconnection,
)


@pytest.mark.parametrize(
    ("reactants_smiles", "expected"),
    [
        (
            "[Cl:2].[CH:1]1=[CH:7][CH:6]=[CH:5][CH:4]=[CH:3]1",
            {2: 0, 1: 1, 7: 2, 6: 3, 5: 4, 4: 5, 3: 6},
        ),
        (
            "[Cl:6].[CH:1]1=[CH:17][CH:2]=[CH:5][CH:24]=[CH:3]1",
            {6: 0, 1: 1, 17: 2, 2: 3, 5: 4, 24: 5, 3: 6},
        ),
    ],
)
def test_mapping_to_index(reactants_smiles, expected):
    mapper = DisconnectionAtomMapper()
    mapping2idx = mapper.mapping_to_index(Chem.MolFromSmiles(reactants_smiles))
    assert mapping2idx == expected


def test_remove_atom_mapping():
    mapper = DisconnectionAtomMapper()
    smiles = "[CH:1]1=[CH:7][CH:6]=[CH:5][CH:4]=[CH:3]1"
    assert mapper.remove_atom_mapping(smiles) == "c1ccccc1"


@pytest.mark.parametrize(
    ("reactants", "product_new_mapping", "product_old_mapping", "expected"),
    [
        (
            "[Cl:2].[CH:1]1=[CH:7][CH:6]=[CH:5][CH:4]=[CH:3]1",
            "[Cl:2][C:1]1=[CH:7][CH:6]=[CH:5][CH:4]=[CH:3]1",
            "[Cl:5][C:3]1=[CH:15][CH:1]=[CH:2][CH:7]=[CH:16]1",
            "[Cl:5].[cH:1]1[cH:2][cH:7][cH:16][cH:3][cH:15]1",
        ),
        (
            "[CH:1]1=[CH:7][CH:6]=[CH:5][CH:4]=[CH:3]1",
            "[Cl:2][C:1]1=[CH:7][CH:6]=[CH:5][CH:4]=[CH:3]1",
            "[Cl:5][C:3]1=[CH:15][CH:1]=[CH:7][CH:8]=[CH:16]1",
            "[cH:1]1[cH:7][cH:8][cH:16][cH:3][cH:15]1",
        ),
    ],
)
def test_input_mapping_to_reactants(reactants, product_new_mapping, product_old_mapping, expected):
    mapper = DisconnectionAtomMapper()
    assert mapper.propagate_input_mapping_to_reactants(product_old_mapping, reactants, product_new_mapping) == expected


@pytest.mark.parametrize(
    ("product_mapping", "bond_atom_inds", "expected"),
    [
        (
            "[Cl:5][C:3]1=[CH:15][CH:1]=[CH:2][CH:6]=[CH:16]1",
            [1, 15],
            "Clc1ccc[cH:1][cH:1]1",
        ),
        (
            "[Cl:5][C:3]1=[CH:15][CH:1]=[CH:2][CH:6]=[CH:16]1",
            [5, 3],
            "c1cc[c:1]([Cl:1])cc1",
        ),
    ],
)
def test_tag_current_bond(product_mapping, bond_atom_inds, expected):
    mapper = DisconnectionAtomMapper()
    assert mapper.tag_current_bond(product_mapping, bond_atom_inds) == expected


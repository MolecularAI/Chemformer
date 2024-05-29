import pandas as pd
import pytest

from molbart.data.base import (
    ChemistryDataset,
    MoleculeListDataModule,
    ReactionListDataModule,
)
from molbart.data.datamodules import SynthesisDataModule
from molbart.utils.tokenizers import ReplaceTokensMasker


@pytest.fixture
def create_smiles_file(tmpdir):
    filename = str(tmpdir / "smiles_temp.txt")

    def wrapper():
        with open(filename, "w") as fileobj:
            fileobj.write(
                "\n".join(
                    [
                        "O",
                        "CC(=O)O",
                        "CC(=O)C",
                        "c1ccccc1",
                        "Cc1ccccc1",
                        "Oc1ccccc1",
                        "C1CCOOC1",
                        "CC(C)(C)O",
                        "CC(C)(Cl)O",
                        "CCN",
                    ]
                )
            )
        return filename

    return wrapper


@pytest.fixture
def create_synthesis_data_file(tmpdir):
    filename = str(tmpdir / "synthesis_data_tmp.csv")

    def wrapper():
        products = [
            "CC(C)(C)OC(=O)N1CC[C@H](N)[C@H](F)C1",
            "Nc1ncc(Br)nc1N1CCOCC1",
            "COC(=O)c1cc(Br)sc1NC(=O)NC(=O)C(Cl)(Cl)Cl",
            "O=S(=O)(c1ccccc1)N1CCNCC1",
            "O=Cc1cc(Br)ccc1OCc1ccccc1",
            "C[Si](C)(C)CCOCn1ccc2cc(Br)ccc21",
        ]
        reactants = [
            "CC(C)(C)OC(=O)N1CC[C@H](NCc2ccccc2)[C@H](F)C1",
            "Nc1ncc(Br)nc1Br.C1COCCN1",
            "COC(=O)c1ccsc1NC(=O)NC(=O)C(Cl)(Cl)Cl.BrBr",
            "O=S(=O)(Cl)c1ccccc1.C1CNCCN1",
            "BrCc1ccccc1.O=Cc1cc(Br)ccc1O",
            "C[Si](C)(C)CCOCCl.Brc1ccc2[nH]ccc2c1",
        ]

        data = pd.DataFrame(
            {
                "products": products,
                "reactants": reactants,
                "set": ["train", "test", "val", "train", "train", "test"],
            }
        )

        data.to_csv(filename, sep="\t", index=False)
        return filename

    return wrapper


@pytest.fixture
def create_reactions_file(tmpdir):
    filename = str(tmpdir / "rxns_temp.txt")

    def wrapper():
        with open(filename, "w") as fileobj:
            fileobj.write(
                "\n".join(
                    [
                        "O>>Cl",
                        "CC(=O)O>>CC(=O)C",
                        "CC(=O)C>>CC(=O)O",
                        "c1ccccc1>>c1ccccc1",
                        "Cc1ccccc1>>Brc1ccccc1",
                        "Oc1ccccc1>>Brc1ccccc1",
                        "C1CCOOC1>>C1CCOOC1",
                        "CC(C)(C)O>>CC(C)(C)O",
                        "CC(C)(Cl)O>>CC(C)(Cl)O",
                        "CCN>>CCO",
                    ]
                )
            )
        return filename

    return wrapper


def test_dataset():
    data = ChemistryDataset({"a": [1, 2, 3], "b": [True, False, True]})

    assert len(data) == 3
    assert data[1] == {"a": 2, "b": False}


def test_dataset_with_len():
    data = ChemistryDataset({"a": [1, 2, 3], "b": [True, False, True]})

    with pytest.raises(KeyError):
        _ = data.seq_lengths

    data = ChemistryDataset({"lengths": [1, 2, 3], "b": [True, False, True]})

    assert data.seq_lengths == [1, 2, 3]


def test_create_mol_datamodule(create_smiles_file, setup_tokenizer):
    dataset_path = create_smiles_file()
    dm = MoleculeListDataModule(
        dataset_path=dataset_path,
        tokenizer=setup_tokenizer(),
        batch_size=2,
        max_seq_len=100,
    )

    dm.setup()

    assert len(dm.train_dataloader()) == 3
    assert len(dm.test_dataloader()) == 1
    assert len(dm.val_dataloader()) == 1
    assert len(dm.full_dataloader()) == 5


def test_create_synthesis_datamodule(create_synthesis_data_file, setup_tokenizer):
    dm = SynthesisDataModule(
        dataset_path=create_synthesis_data_file(),
        tokenizer=setup_tokenizer(),
        batch_size=1,
        max_seq_len=100,
    )

    dm.setup()

    print(
        [
            len(dm.train_dataloader()),
            len(dm.test_dataloader()),
            len(dm.val_dataloader()),
            len(dm.full_dataloader()),
        ]
    )

    assert len(dm.train_dataloader()) == 3
    assert len(dm.test_dataloader()) == 2
    assert len(dm.val_dataloader()) == 1
    assert len(dm.full_dataloader()) == 6


def test_create_mol_datamodule_test_idxs(create_smiles_file, setup_tokenizer):
    dataset_path = create_smiles_file()
    dm = MoleculeListDataModule(
        dataset_path=dataset_path,
        tokenizer=setup_tokenizer(),
        batch_size=2,
        max_seq_len=100,
        test_idxs=[0, 1, 2, 3],
    )

    dm.setup()

    # Random sampler for training cannot handle empty sets
    assert len(dm.train_dataloader()) == 3
    assert len(dm.test_dataloader()) == 2
    assert len(dm.val_dataloader()) == 0
    assert len(dm.full_dataloader()) == 5


def test_create_mol_datamodule_val_idxs(create_smiles_file, setup_tokenizer):
    dataset_path = create_smiles_file()
    dm = MoleculeListDataModule(
        dataset_path=dataset_path,
        tokenizer=setup_tokenizer(),
        batch_size=2,
        max_seq_len=100,
        val_idxs=[0, 1, 2, 3],
    )

    dm.setup()

    # Random sampler for training cannot handle empty sets
    assert len(dm.train_dataloader()) == 3
    assert len(dm.test_dataloader()) == 0
    assert len(dm.val_dataloader()) == 2
    assert len(dm.full_dataloader()) == 5


def test_create_mol_datamodule_test_val_idxs(create_smiles_file, setup_tokenizer):
    dataset_path = create_smiles_file()
    dm = MoleculeListDataModule(
        dataset_path=dataset_path,
        tokenizer=setup_tokenizer(),
        batch_size=2,
        max_seq_len=100,
        test_idxs=[4, 5, 6, 7, 8, 9],
        val_idxs=[0, 1, 2, 3],
    )

    dm.setup()

    # Random sampler for training cannot handle empty sets
    with pytest.raises(ValueError):
        dm.train_dataloader()
    assert len(dm.test_dataloader()) == 3
    assert len(dm.val_dataloader()) == 2
    assert len(dm.full_dataloader()) == 5


@pytest.mark.parametrize(
    ("task", "expect_mask_token"),
    [
        ("aug", False),
        ("mask", True),
        ("mask_aug", True),
    ],
)
def test_mol_datamodule_collation(create_smiles_file, setup_masker, task, expect_mask_token):
    dataset_path = create_smiles_file()
    tokenizer, masker = setup_masker(ReplaceTokensMasker)
    dm = MoleculeListDataModule(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        batch_size=10,
        max_seq_len=100,
        task=task,
        masker=masker,
        augment_prob=0.5,
    )
    dm.setup()

    batch = next(iter(dm.full_dataloader()))

    for expected_key in [
        "encoder_input",
        "encoder_pad_mask",
        "decoder_input",
        "decoder_pad_mask",
        "target",
        "target_mask",
        "target_smiles",
    ]:
        assert expected_key in batch

    assert tuple(batch["encoder_input"].shape) == (13, 10)
    assert tuple(batch["encoder_pad_mask"].shape) == (13, 10)
    assert tuple(batch["decoder_input"].shape) == (12, 10)
    assert tuple(batch["decoder_pad_mask"].shape) == (12, 10)
    assert tuple(batch["target"].shape) == (12, 10)
    assert tuple(batch["target_mask"].shape) == (12, 10)
    assert len(batch["target_smiles"]) == 10

    # Check for mask tokens
    mask_id = tokenizer[tokenizer.special_tokens["mask"]]
    found_mask = False
    for lst in batch["encoder_input"].numpy().T.tolist():
        if mask_id in lst:
            found_mask = True
    if expect_mask_token:
        assert found_mask
    else:
        assert not found_mask


def test_mol_datamodule_collation_overlap(create_smiles_file, setup_masker):
    dataset_path = create_smiles_file()
    tokenizer, masker = setup_masker(ReplaceTokensMasker)
    common_arg = {
        "dataset_path": dataset_path,
        "tokenizer": tokenizer,
        "batch_size": 10,
        "max_seq_len": 100,
        "masker": masker,
        "augment_prob": 0.5,
    }
    dm_mask = MoleculeListDataModule(task="mask", **common_arg)
    dm_mask.setup()
    dm_aug = MoleculeListDataModule(task="aug", **common_arg)
    dm_aug.setup()
    dm_aug_mask = MoleculeListDataModule(task="aug_mask", **common_arg)
    dm_aug_mask.setup()

    batch_mask = next(iter(dm_mask.full_dataloader()))
    batch_aug = next(iter(dm_aug.full_dataloader()))
    batch_aug_mask = next(iter(dm_aug_mask.full_dataloader()))

    assert batch_mask["encoder_input"].tolist() != batch_aug["encoder_input"].tolist()
    assert batch_mask["encoder_input"].tolist() != batch_aug_mask["encoder_input"].tolist()
    assert batch_aug["encoder_input"].tolist() != batch_aug_mask["encoder_input"].tolist()

    assert batch_mask["target"].tolist() != batch_aug["target"].tolist()
    assert batch_mask["target"].tolist() != batch_aug_mask["target"].tolist()
    assert batch_aug["target"].tolist() == batch_aug_mask["target"].tolist()


def test_mol_datamodule_unified_collation(create_smiles_file, setup_masker):
    dataset_path = create_smiles_file()
    tokenizer, masker = setup_masker(ReplaceTokensMasker)
    dm = MoleculeListDataModule(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        batch_size=10,
        max_seq_len=100,
        task="mask",
        masker=masker,
        augment_prob=0.0,
        unified_model=True,
    )
    dm.setup()

    batch = next(iter(dm.full_dataloader()))

    for expected_key in [
        "encoder_input",
        "encoder_pad_mask",
        "decoder_input",
        "decoder_pad_mask",
        "target",
        "target_mask",
        "target_smiles",
        "attention_mask",
    ]:
        assert expected_key in batch

    assert tuple(batch["encoder_input"].shape) == (12, 10)
    assert tuple(batch["encoder_pad_mask"].shape) == (12, 10)
    assert tuple(batch["decoder_input"].shape) == (9, 10)
    assert tuple(batch["decoder_pad_mask"].shape) == (9, 10)
    assert tuple(batch["target"].shape) == (21, 10)
    assert tuple(batch["target_mask"].shape) == (21, 10)
    assert tuple(batch["attention_mask"].shape) == (21, 21)
    assert len(batch["target_smiles"]) == 10


def test_rxn_datamodule_collation(create_reactions_file, setup_tokenizer):
    dataset_path = create_reactions_file()
    dm = ReactionListDataModule(
        dataset_path=dataset_path,
        tokenizer=setup_tokenizer(),
        batch_size=10,
        max_seq_len=100,
    )
    dm.setup()

    batch = next(iter(dm.full_dataloader()))

    for expected_key in [
        "encoder_input",
        "encoder_pad_mask",
        "decoder_input",
        "decoder_pad_mask",
        "target",
        "target_mask",
        "target_smiles",
    ]:
        assert expected_key in batch

    assert tuple(batch["encoder_input"].shape) == (11, 10)
    assert tuple(batch["encoder_pad_mask"].shape) == (11, 10)
    assert tuple(batch["decoder_input"].shape) == (10, 10)
    assert tuple(batch["decoder_pad_mask"].shape) == (10, 10)
    assert tuple(batch["target"].shape) == (10, 10)
    assert tuple(batch["target_mask"].shape) == (10, 10)
    assert len(batch["target_smiles"]) == 10


def test_rxn_datamodule_reverse(create_reactions_file, setup_tokenizer):
    dataset_path = create_reactions_file()
    dm = ReactionListDataModule(
        dataset_path=dataset_path,
        tokenizer=setup_tokenizer(),
        batch_size=10,
        max_seq_len=100,
    )
    dm.setup()
    dm_reverse = ReactionListDataModule(
        dataset_path=dataset_path,
        tokenizer=setup_tokenizer(),
        batch_size=10,
        max_seq_len=100,
        reverse=True,
    )
    dm_reverse.setup()

    batch = next(iter(dm.full_dataloader()))
    batch_reverse = next(iter(dm_reverse.full_dataloader()))

    assert batch["encoder_input"][1:, :].tolist() != batch_reverse["decoder_input"].tolist()
    assert batch["decoder_input"].tolist() != batch_reverse["encoder_input"][1:, :].tolist()

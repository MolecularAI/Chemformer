import pickle
import argparse
from pathlib import Path
from rdkit import Chem

from molbart.tokeniser import MolEncTokeniser
from molbart.data.datasets import Chembl, MolOptDataset


MOL_OPT_TOKENS_PATH = "mol_opt_tokens.txt"
PROP_PRED_TOKENS_PATH = "prop_pred_tokens.txt"
NUM_UNUSED_TOKENS = 200
REGEX = "\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]"


def build_mol_dataset(args):
    dataset = Chembl(args.data_path)
    return dataset


def build_mol_opt_dataset(args):
    dataset = MolOptDataset(args.mol_opt_data_path)
    return dataset


def read_extra_tokens(paths):
    extra_tokens = []
    for path in paths:
        p = Path(path)
        if p.is_file():
            text = p.read_text()
            tokens = text.split("\n")
            tokens = [token for token in tokens if token != ""]
            print(f"Read {len(tokens)} tokens from {path}")
            extra_tokens.extend(tokens)

    return extra_tokens


def build_unused_tokens(num_tokens):
    tokens = []
    for i in range(num_tokens):
        token = f"<UNUSED_{str(i)}>"
        tokens.append(token)

    return tokens


def build_tokeniser(smiles, regex, extra_tokens):
    tokeniser = MolEncTokeniser.from_smiles(smiles, regex, extra_tokens=extra_tokens)
    return tokeniser


def write_tokeniser(args, tokeniser):
    write_path = Path(args.tokeniser_path)
    write_path.parent.mkdir(parents=True, exist_ok=True)
    file_handle = write_path.open("wb")
    pickle.dump(tokeniser, file_handle)
    file_handle.close()


def main(args):
    print("Reading molecule dataset...")
    mol_dataset = build_mol_dataset(args)
    print("Completed reading dataset.")

    print("Reading extra tokens...")
    paths = [args.mol_opt_tokens_path, args.prop_pred_tokens_path]
    extra_tokens = read_extra_tokens(paths)
    unused_tokens = build_unused_tokens(NUM_UNUSED_TOKENS)
    print("Completed reading extra tokens.")

    print("Constructing SMILES strings...")
    mol_smiles = [Chem.MolToSmiles(mol_dataset[idx]) for idx in range(len(mol_dataset))]
    print("Completed SMILES construction.")

    print("Building tokeniser...")
    tokeniser = build_tokeniser(mol_smiles, REGEX, extra_tokens + unused_tokens)
    print("Completed building tokeniser.")

    print("Writing tokeniser...")
    tokeniser.save_vocab(args.tokeniser_path)
    print("Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Script for training the tokeniser on a dataset.")

    parser.add_argument("--data_path", type=str)
    parser.add_argument("--mol_opt_tokens_path", type=str, default=MOL_OPT_TOKENS_PATH)
    parser.add_argument("--prop_pred_tokens_path", type=str, default=PROP_PRED_TOKENS_PATH)
    parser.add_argument("--tokeniser_path", type=str)

    args = parser.parse_args()
    main(args)

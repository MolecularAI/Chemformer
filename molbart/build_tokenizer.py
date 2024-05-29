from pathlib import Path

import hydra
import pandas as pd

from molbart.utils.tokenizer import ChemformerTokenizer
from molbart.utils.data_utils import REGEX


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


@hydra.main(version_base=None, config_path="config", config_name="build_tokeniser")
def main(args):
    print("Reading molecule dataset...")
    mol_dataset = pd.read_pickle(args.data_path)
    smiles = mol_dataset[args.smiles_column].values.tolist()
    print("Completed reading dataset.")

    print("Reading extra tokens...")
    paths = [args.mol_opt_tokens_path, args.prop_pred_tokens_path]
    extra_tokens = read_extra_tokens(paths)
    unused_tokens = build_unused_tokens(args.num_unused_tokens)
    print("Completed reading extra tokens.")

    print("Building tokenizer...")
    tokenizer = ChemformerTokenizer(
        smiles=smiles,
        tokens=extra_tokens + unused_tokens,
        regex_token_patterns=REGEX.split("|"),
    )
    print("Completed building tokenizer.")

    print("Writing tokenizer...")
    tokenizer.save_vocabulary(args.tokeniser_path)
    print("Complete.")


if __name__ == "__main__":
    main()

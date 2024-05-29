import argparse
from argparse import Namespace

import pandas as pd

ENV_OK = True
try:
    from rxnutils.chem.disconnection_sites.tag_converting import convert_atom_map_tag
except ImportError:
    ENV_OK = False

from typing import Optional


def main(args: Optional[Namespace] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_smiles", nargs="+")
    parser.add_argument("--output")
    parser.add_argument("--column_in", default="products_atom_map_tagged")
    parser.add_argument("--column_out", default="products_tagged")

    args = parser.parse_args(args)

    if args.output:
        data = pd.DataFrame(
            {
                args.column_in: args.input_smiles,
            }
        )

        products_tagged_col = data[args.column_in].apply(convert_atom_map_tag)

        data = data.assign(**{args.column_out: products_tagged_col})

        with open(args.output, "w") as file:
            file.write(" ".join(data[args.column_out].values))
    else:
        tagged_smiles = [convert_atom_map_tag(smiles) for smiles in args.input_smiles]
        print(" ".join(tagged_smiles))

    return


if __name__ == "__main__":
    main()

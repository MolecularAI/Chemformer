import argparse

import molbart.modules.util as util
from molbart.models import Chemformer

DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_BEAMS = 10
DEFAULT_NUM_UNIQUE_BEAMS = 10


def main(args):
    util.seed_everything(1)

    print("Running model inference and scoring.")

    if args.dataset_type not in [
        "uspto_mixed",
        "uspto_50",
        "uspto_sep",
        "uspto_50_with_type",
        "synthesis",
    ]:
        raise ValueError(f"Unknown dataset: {args.dataset_type}")

    model_args, data_args = util.get_chemformer_args(args)

    kwargs = {
        "vocabulary_path": args.vocabulary_path,
        "n_gpus": args.n_gpus,
        "model_path": args.model_path,
        "model_args": model_args,
        "data_args": data_args,
        "n_beams": args.n_beams,
        "train_mode": "eval",
    }

    chemformer = Chemformer(**kwargs)

    chemformer.score_model(
        n_unique_beams=args.n_unique_beams,
        dataset=args.dataset_part,
        i_chunk=args.i_chunk,
        n_chunks=args.n_chunks,
        outfile=args.output_score_data,
        outfile_sampled_smiles=args.output_sampled_smiles,
    )
    print("Model inference and scoring done.")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path")
    parser.add_argument("--model_path")
    parser.add_argument(
        "--dataset_part",
        default="test",
        help="Which dataset split to run inference on.",
        choices=["full", "train", "val", "test"],
    )
    parser.add_argument("--dataset_type", default=util.DEFAULT_DATASET_TYPE)
    parser.add_argument(
        "--model_type", choices=["bart", "unified"], default=util.DEFAULT_MODEL
    )
    parser.add_argument("--task", choices=["forward_prediction", "backward_prediction"])
    parser.add_argument("--vocabulary_path")

    # Output args
    parser.add_argument(
        "-o",
        "--output_score_data",
        help="Path to .csv file to which model score results should be written.",
    )
    parser.add_argument(
        "-os",
        "--output_sampled_smiles",
        help="Path to .json file to which sampled smiles should be written.",
    )

    # Model args
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--n_beams", type=int, default=DEFAULT_NUM_BEAMS)
    parser.add_argument("--n_unique_beams", type=int)
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--n_chunks", type=int, default=1)
    parser.add_argument("-i", "--i_chunk", type=int, default=0)

    args = parser.parse_args()
    main(args)

import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

import molbart.util as util
from molbart.decoder import DecodeSampler
from molbart.data.datamodules import FineTuneReactionDataModule


DEFAULT_BATCH_SIZE = 64
DEFAULT_AUG_PROB = 0.0
DEFAULT_TRAIN_TOKENS = None
DEFAULT_NUM_BUCKETS = None
DEFAULT_NUM_BEAMS = 10
DEFAULT_VOCAB_PATH = "bart_vocab_downstream.txt"


def build_trainer(args, limit_test_batches=1.0):
    logger = TensorBoardLogger("tb_logs", name=f"eval_{args.model_type}_{args.dataset}")
    trainer = Trainer(
        logger=logger,
        limit_test_batches=limit_test_batches,
        precision=16,
        gpus=1
    )
    return trainer


def main(args):
    util.seed_everything(1)

    if args.dataset not in ["uspto_mixed", "uspto_50", "uspto_sep", "uspto_50_with_type"]:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if args.task == "forward_prediction":
        forward_pred = True
    elif args.task == "backward_prediction":
        forward_pred = False
    else:
        raise ValueError(f"Unknown model type {args.task}")

    print("Building tokeniser...")
    tokeniser = util.load_tokeniser(args.vocab_path, args.chem_token_start_idx)
    print("Finished tokeniser.")

    print("Reading dataset...")
    dataset = util.build_dataset(args, forward=forward_pred)
    print("Finished dataset.")

    print("Building data module...")
    dm = util.build_reaction_datamodule(args, dataset, tokeniser, forward=forward_pred)
    print("Finished datamodule.")

    sampler = DecodeSampler(tokeniser, args.max_seq_len)

    print("Loading model...")
    if args.model_type == "bart":
        model = util.load_bart(args, sampler)
    elif args.model_type == "unified":
        model = util.load_unified(args, sampler)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    model.num_beams = args.num_beams
    print("Finished model.")

    print("Building trainer...")
    trainer = build_trainer(args)
    print("Finished trainer.")

    print("Evaluating model...")
    results = trainer.test(model, datamodule=dm)
    util.print_results(args, results[0])
    print("Finished evaluation.")

    print("Printing unknown tokens...")
    tokeniser.print_unknown_tokens()
    print("Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Program level args
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--vocab_path", type=str, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--chem_token_start_idx", type=int, default=util.DEFAULT_CHEM_TOKEN_START)

    # Model args
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_seq_len", type=int, default=util.DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--num_beams", type=int, default=DEFAULT_NUM_BEAMS)
    parser.add_argument("--train_tokens", type=int, default=DEFAULT_TRAIN_TOKENS)
    parser.add_argument("--num_buckets", type=int, default=DEFAULT_NUM_BUCKETS)
    parser.add_argument("--aug_prob", type=float, default=DEFAULT_AUG_PROB)

    args = parser.parse_args()
    main(args)

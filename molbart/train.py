import os
import argparse

import molbart.util as util
from molbart.models.pre_train import BARTModel, UnifiedModel
from molbart.decoder import DecodeSampler


# Default training hyperparameters
DEFAULT_BATCH_SIZE = 128
DEFAULT_ACC_BATCHES = 1
DEFAULT_MASK_PROB = 0.10
DEFAULT_MASK_SCHEME = "span"
DEFAULT_LR = 1.0
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_EPOCHS = 10
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_TRAIN_TOKENS = None
DEFAULT_NUM_BUCKETS = 12
DEFAULT_LIMIT_VAL_BATCHES = 1.0
DEFAULT_SCHEDULE = "transformer"
DEFAULT_WARM_UP_STEPS = 8000
DEFAULT_TASK = "mask_aug"
DEFAULT_AUGMENT = True


def build_model(args, sampler, vocab_size, total_steps, pad_token_idx):
    # These args don't affect the model directly but will be saved by lightning as hparams
    # Tensorboard doesn't like None so we need to convert to string
    augment = "None" if args.augment is None else args.augment
    train_tokens = "None" if args.train_tokens is None else args.train_tokens
    num_buckets = "None" if args.num_buckets is None else args.num_buckets
    extra_args = {
        "batch_size": args.batch_size,
        "acc_batches": args.acc_batches,
        "mask_prob": args.mask_prob,
        "epochs": args.epochs,
        "clip_grad": args.clip_grad,
        "train_tokens": train_tokens,
        "num_buckets": num_buckets,
        "limit_val_batches": args.limit_val_batches,
        "augment": augment,
        "task": args.task,
        "mask_scheme": args.mask_scheme,
        "model_type": args.model_type
    }

    if args.model_type == "bart":
        model = BARTModel(
            sampler,
            pad_token_idx,
            vocab_size,
            args.d_model,
            args.num_layers,
            args.num_heads,
            args.d_feedforward,
            args.lr,
            args.weight_decay,
            args.activation,
            total_steps,
            args.max_seq_len,
            schedule=args.schedule,
            warm_up_steps=args.warm_up_steps,
            dropout=util.DEFAULT_DROPOUT,
            **extra_args
        )
    elif args.model_type == "unified":
        model = UnifiedModel(
            sampler,
            pad_token_idx,
            vocab_size,
            args.d_model,
            args.num_layers,
            args.num_heads,
            args.d_feedforward,
            args.lr,
            args.weight_decay,
            args.activation,
            total_steps,
            args.max_seq_len,
            schedule=args.schedule,
            warm_up_steps=args.warm_up_steps,
            dropout=util.DEFAULT_DROPOUT,
            **extra_args
        )
    else:
        raise ValueError(f"Unknown model type {args.model_type}")

    return model


def main(args):
    util.seed_everything(37)

    if args.dataset == "zinc" and args.train_tokens is not None:
        raise ValueError("train_tokens arg must be None when using zinc dataset.")

    if args.gpus > 1 and args.train_tokens is not None:
        raise ValueError("train_tokens arg must be None when training on multiple gpus.")

    print("Building tokeniser...")
    tokeniser = util.load_tokeniser(args.vocab_path, args.chem_token_start_idx)
    tokeniser.mask_prob = args.mask_prob
    tokeniser.mask_scheme = args.mask_scheme
    print("Finished tokeniser.")

    print("Reading dataset...")
    dataset = util.build_dataset(args)
    print("Finished dataset.")

    print("Building data module...")
    dm = util.build_molecule_datamodule(args, dataset, tokeniser)
    num_available_cpus = len(os.sched_getaffinity(0))
    num_workers = num_available_cpus // args.gpus
    dm._num_workers = num_workers
    print(f"Using {str(num_workers)} workers for data module.")
    print("Finished data module.")

    vocab_size = len(tokeniser)
    train_steps = util.calc_train_steps(args, dm)
    print(f"Train steps: {train_steps}")

    sampler = DecodeSampler(tokeniser, args.max_seq_len)
    pad_token_idx = tokeniser.vocab[tokeniser.pad_token]

    print("Building model...")
    model = build_model(args, sampler, vocab_size, train_steps, pad_token_idx)
    print("Finished model.")

    print("Building trainer...")
    trainer = util.build_trainer(args)
    print("Finished trainer.")

    print("Fitting data module to model")
    trainer.fit(model, dm)
    print("Finished training.")

    print("Printing unknown tokens...")
    tokeniser.print_unknown_tokens()
    print("Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Program level args
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--vocab_path", type=str, default=util.DEFAULT_VOCAB_PATH)
    parser.add_argument("--chem_token_start_idx", type=int, default=util.DEFAULT_CHEM_TOKEN_START)
    parser.add_argument("--log_dir", type=str, default=util.DEFAULT_LOG_DIR)
    parser.add_argument("--deepspeed_config_path", type=str, default=util.DEFAULT_DEEPSPEED_CONFIG_PATH)

    # Model and training args
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--acc_batches", type=int, default=DEFAULT_ACC_BATCHES)
    parser.add_argument("--max_seq_len", type=int, default=util.DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--mask_prob", type=float, default=DEFAULT_MASK_PROB)
    parser.add_argument("--mask_scheme", type=str, default=DEFAULT_MASK_SCHEME)
    parser.add_argument("--d_model", type=int, default=util.DEFAULT_D_MODEL)
    parser.add_argument("--num_layers", type=int, default=util.DEFAULT_NUM_LAYERS)
    parser.add_argument("--num_heads", type=int, default=util.DEFAULT_NUM_HEADS)
    parser.add_argument("--d_feedforward", type=int, default=util.DEFAULT_D_FEEDFORWARD)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--activation", type=str, default=util.DEFAULT_ACTIVATION)
    parser.add_argument("--clip_grad", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--train_tokens", type=int, default=DEFAULT_TRAIN_TOKENS)
    parser.add_argument("--num_buckets", type=int, default=DEFAULT_NUM_BUCKETS)
    parser.add_argument("--limit_val_batches", type=float, default=DEFAULT_LIMIT_VAL_BATCHES)
    parser.add_argument("--gpus", type=int, default=util.DEFAULT_GPUS)
    parser.add_argument("--num_nodes", type=int, default=util.DEFAULT_NUM_NODES)
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--schedule", type=str, default=DEFAULT_SCHEDULE)
    parser.add_argument("--warm_up_steps", type=int, default=DEFAULT_WARM_UP_STEPS)

    parser.add_argument("--augment", dest="augment", action="store_true")
    parser.add_argument("--no_augment", dest="augment", action="store_false")
    parser.set_defaults(augment=DEFAULT_AUGMENT)

    args = parser.parse_args()
    main(args)

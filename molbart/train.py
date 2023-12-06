import argparse
import os

import molbart.modules.util as util
from molbart.models.transformer_models import BARTModel, UnifiedModel
from molbart.modules.decoder import DecodeSampler
from molbart.modules.tokenizer import (
    ChemformerTokenizer,
    ReplaceTokensMasker,
    SpanTokensMasker,
)

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
    augment = (
        "None" if args.augmentation_strategy is None else args.augmentation_strategy
    )
    train_tokens = "None" if args.train_tokens is None else args.train_tokens
    n_buckets = "None" if args.n_buckets is None else args.n_buckets
    extra_args = {
        "batch_size": args.batch_size,
        "acc_batches": args.acc_batches,
        "mask_prob": args.mask_prob,
        "epochs": args.n_epochs,
        "clip_grad": args.clip_grad,
        "train_tokens": train_tokens,
        "num_buckets": n_buckets,
        "limit_val_batches": args.limit_val_batches,
        "augment": augment,
        "task": args.task,
        "mask_scheme": args.mask_scheme,
        "model_type": args.model_type,
    }

    if args.model_type == "bart":
        model = BARTModel(
            sampler,
            pad_token_idx,
            vocab_size,
            args.d_model,
            args.n_layers,
            args.n_heads,
            args.d_feedforward,
            args.learning_rate,
            args.weight_decay,
            args.activation,
            total_steps,
            args.max_seq_len,
            schedule=args.schedule,
            warm_up_steps=args.warm_up_steps,
            dropout=util.DEFAULT_DROPOUT,
            **extra_args,
        )
    elif args.model_type == "unified":
        model = UnifiedModel(
            sampler,
            pad_token_idx,
            vocab_size,
            args.d_model,
            args.n_layers,
            args.n_heads,
            args.d_feedforward,
            args.learning_rate,
            args.weight_decay,
            args.activation,
            total_steps,
            args.max_seq_len,
            schedule=args.schedule,
            warm_up_steps=args.warm_up_steps,
            dropout=util.DEFAULT_DROPOUT,
            **extra_args,
        )
    else:
        raise ValueError(f"Unknown model type {args.model_type}")

    return model


def main(args):
    util.seed_everything(37)

    if args.dataset_type == "zinc" and args.train_tokens is not None:
        raise ValueError("train_tokens arg must be None when using zinc dataset.")

    if args.n_gpus > 1 and args.train_tokens is not None:
        raise ValueError(
            "train_tokens arg must be None when training on multiple gpus."
        )

    print("Building tokeniser...")
    tokeniser = ChemformerTokenizer(filename=args.vocabulary_path)
    if args.mask_scheme == "replace":
        masker = ReplaceTokensMasker(tokenizer=tokeniser, mask_prob=args.mask_prob)
    else:
        masker = SpanTokensMasker(tokenizer=tokeniser, mask_prob=args.mask_prob)
    print("Finished tokeniser.")

    print("Building data module...")
    dm = util.build_molecule_datamodule(args, tokeniser, masker=masker)
    n_available_cpus = len(os.sched_getaffinity(0))
    n_workers = n_available_cpus // args.n_gpus
    dm._num_workers = n_workers
    print(f"Using {str(n_workers)} workers for data module.")
    print("Finished data module.")

    vocab_size = len(tokeniser)
    train_steps = util.calc_train_steps(args, dm)
    print(f"Train steps: {train_steps}")

    sampler = DecodeSampler(tokeniser, args.max_seq_len)
    pad_token_idx = tokeniser["pad"]

    print("Building model...")
    model = build_model(args, sampler, vocab_size, train_steps, pad_token_idx)
    print("Finished model.")

    print("Building trainer...")
    trainer = util.build_trainer(args)
    print("Finished trainer.")

    print("Fitting data module to model")
    trainer.fit(model, dm)
    print("Finished training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Program level args
    parser.add_argument("--dataset_type", choices=["chembl", "zinc"])
    parser.add_argument("--data_path")
    parser.add_argument(
        "--model_type", choices=["bart", "unified"], default=util.DEFAULT_MODEL
    )
    parser.add_argument("--vocabulary_path", default=util.DEFAULT_VOCAB_PATH)
    parser.add_argument("--output_directory", default=util.DEFAULT_LOG_DIR)
    parser.add_argument(
        "--deepspeed_config_path", type=str, default=util.DEFAULT_DEEPSPEED_CONFIG_PATH
    )

    # Model and training args
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--acc_batches", type=int, default=DEFAULT_ACC_BATCHES)
    parser.add_argument("--max_seq_len", type=int, default=util.DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--mask_prob", type=float, default=DEFAULT_MASK_PROB)
    parser.add_argument(
        "--mask_scheme", choices=["span", "replace"], default=DEFAULT_MASK_SCHEME
    )
    parser.add_argument("--d_model", type=int, default=util.DEFAULT_D_MODEL)
    parser.add_argument("--n_layers", type=int, default=util.DEFAULT_NUM_LAYERS)
    parser.add_argument("--n_heads", type=int, default=util.DEFAULT_NUM_HEADS)
    parser.add_argument("--d_feedforward", type=int, default=util.DEFAULT_D_FEEDFORWARD)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--n_epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--activation", default=util.DEFAULT_ACTIVATION)
    parser.add_argument("--clip_grad", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--train_tokens", type=int, default=DEFAULT_TRAIN_TOKENS)
    parser.add_argument("--n_buckets", type=int, default=DEFAULT_NUM_BUCKETS)
    parser.add_argument(
        "--limit_val_batches", type=float, default=DEFAULT_LIMIT_VAL_BATCHES
    )
    parser.add_argument("--n_gpus", type=int, default=util.DEFAULT_GPUS)
    parser.add_argument("--n_nodes", type=int, default=util.DEFAULT_NUM_NODES)
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--schedule", type=str, default=DEFAULT_SCHEDULE)
    parser.add_argument("--warm_up_steps", type=int, default=DEFAULT_WARM_UP_STEPS)

    parser.add_argument(
        "--augmentation_strategy", dest="augmentation_strategy", action="store_true"
    )
    parser.add_argument(
        "--no_augment", dest="augmentation_strategy", action="store_false"
    )
    parser.set_defaults(augmentation_strategy=DEFAULT_AUGMENT)

    args = parser.parse_args()
    main(args)

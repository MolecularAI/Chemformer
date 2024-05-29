import os

import hydra

import molbart.utils.data_utils as util
import molbart.utils.trainer_utils as trainer_utils
from molbart.models.transformer_models import BARTModel, UnifiedModel
from molbart.utils.samplers.beam_search_samplers import DecodeSampler
from molbart.utils.tokenizer import ChemformerTokenizer, ReplaceTokensMasker, SpanTokensMasker


def build_model(args, sampler, vocab_size, total_steps, pad_token_idx):
    # These args don't affect the model directly but will be saved by lightning as hparams
    # Tensorboard doesn't like None so we need to convert to string
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
        "augment_prob": args.augmentation_probability,
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


@hydra.main(version_base=None, config_path="config", config_name="pretrain")
def main(args):
    util.seed_everything(args.seed)

    if args.dataset_type == "zinc" and args.train_tokens is not None:
        raise ValueError("train_tokens arg must be None when using zinc dataset.")

    if args.n_gpus > 1 and args.train_tokens is not None:
        raise ValueError("train_tokens arg must be None when training on multiple gpus.")

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
    train_steps = trainer_utils.calc_train_steps(args, dm)
    print(f"Train steps: {train_steps}")

    sampler = DecodeSampler(tokeniser, args.max_seq_len)
    pad_token_idx = tokeniser["pad"]

    print("Building model...")
    model = build_model(args, sampler, vocab_size, train_steps, pad_token_idx)
    print("Finished model.")

    print("Building trainer...")
    trainer = trainer_utils.build_trainer(args)
    print("Finished trainer.")

    print("Fitting data module to model")
    trainer.fit(model, dm)
    print("Finished training.")


if __name__ == "__main__":
    main()

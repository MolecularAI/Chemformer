import argparse
import time

import molbart.modules.util as util
from molbart.models import Chemformer

# Default training hyperparameters
DEFAULT_BATCH_SIZE = 64
DEFAULT_ACC_BATCHES = 8
DEFAULT_LR = 0.001
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_EPOCHS = 50
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_SCHEDULE = "cycle"
DEFAULT_AUGMENT = "None"
DEFAULT_AUG_PROB = 0.5
DEFAULT_WARM_UP_STEPS = 8000
DEFAULT_TRAIN_TOKENS = None
DEFAULT_NUM_BUCKETS = 24
DEFAULT_LIMIT_VAL_BATCHES = 1.0
DEFAULT_DOWNSTREAM_VOCAB_PATH = "bart_vocab_downstream.txt"
DEFAULT_MODEL_TYPE = "bart"
CHECK_VAL_EVERY_N_EPOCH = 10
CHECKPOINT_EVERY_N_STEP = 50000


def main(args):
    util.seed_everything(73)
    print("Fine-tuning CHEMFORMER.")
    model_args, data_args = util.get_chemformer_args(args)

    kwargs = {
        "vocabulary_path": args.vocabulary_path,
        "n_gpus": args.n_gpus,
        "model_path": args.model_path,
        "model_args": model_args,
        "data_args": data_args,
        "resume_training": args.resume,
    }

    chemformer = Chemformer(**kwargs)

    print("Training model...")
    t0 = time.time()
    chemformer.fit()
    t_fit = time.time() - t0
    print(f"Training complete, time: {t_fit}")

    print("Writing logged metrics to csv.")
    chemformer.save_logged_data()
    print("Done writing.")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Program level arguments
    parser.add_argument(
        "--data_path", help="The path and filename of the data file to use"
    )
    parser.add_argument(
        "--task", choices=["forward_prediction", "backward_prediction", "mol_opt"]
    )
    parser.add_argument(
        "--model_path",
        help="The path and filename to the checkpoint file of the pretrained model",
    )
    parser.add_argument(
        "--dataset_type",
        help="The datamodule to to use, example 'uspto_50' (see molbart.util.build_seq2seq_datamodule",
    )
    parser.add_argument(
        "--model_type", choices=["bart", "unified"], default=DEFAULT_MODEL_TYPE
    )
    parser.add_argument("--vocabulary_path", default=DEFAULT_DOWNSTREAM_VOCAB_PATH)

    parser.add_argument("-o", "--output_directory", default=util.DEFAULT_LOG_DIR)
    parser.add_argument(
        "--deepspeed_config_path", default=util.DEFAULT_DEEPSPEED_CONFIG_PATH
    )

    # Model and training arguments
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--acc_batches", type=int, default=DEFAULT_ACC_BATCHES)
    parser.add_argument("-lr", "--learning_rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--n_epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--clip_grad", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--schedule", default=DEFAULT_SCHEDULE)
    parser.add_argument("--augmentation_strategy", default=DEFAULT_AUGMENT)
    parser.add_argument(
        "-aug_prob", "--augmentation_probability", type=float, default=DEFAULT_AUG_PROB
    )
    parser.add_argument("--warm_up_steps", type=int, default=DEFAULT_WARM_UP_STEPS)
    parser.add_argument("--train_tokens", type=int, default=DEFAULT_TRAIN_TOKENS)
    parser.add_argument("--n_buckets", type=int, default=DEFAULT_NUM_BUCKETS)
    parser.add_argument(
        "--limit_val_batches", type=float, default=DEFAULT_LIMIT_VAL_BATCHES
    )
    parser.add_argument(
        "--check_val_every_n_epoch", type=int, default=CHECK_VAL_EVERY_N_EPOCH
    )
    parser.add_argument(
        "--checkpoint_every_n_step",
        type=int,
        default=CHECKPOINT_EVERY_N_STEP,
        help="Step is understood as the number of optimizer steps taken, thus if --acc_batches is more than one, it will be saved every acc_batches*num_minibatches",
    )
    parser.add_argument("--n_gpus", type=int, default=util.DEFAULT_GPUS)
    parser.add_argument("--n_nodes", type=int, default=util.DEFAULT_NUM_NODES)

    # Arguments for model from random initialization
    parser.add_argument("--d_model", type=int, default=util.DEFAULT_D_MODEL)
    parser.add_argument("--n_layers", type=int, default=util.DEFAULT_NUM_LAYERS)
    parser.add_argument("--n_heads", type=int, default=util.DEFAULT_NUM_HEADS)
    parser.add_argument("--d_feedforward", type=int, default=util.DEFAULT_D_FEEDFORWARD)

    args = parser.parse_args()
    main(args)

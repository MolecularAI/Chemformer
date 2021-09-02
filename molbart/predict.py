import torch
import pickle
import argparse
import pandas as pd
from rdkit import Chem
from pathlib import Path

import molbart.util as util
from molbart.decoder import DecodeSampler
from molbart.models.pre_train import BARTModel
from molbart.data.datasets import ReactionDataset
from molbart.data.datamodules import FineTuneReactionDataModule


DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_BEAMS = 10


class SmilesError(Exception):
    def __init__(self, idx, smi):
        message = f"RDKit could not parse smiles {smi} at index {idx}"
        super().__init__(message)


def build_dataset(args):
    text = Path(args.reactants_path).read_text()
    smiles = text.split("\n")
    smiles = [smi for smi in smiles if smi != "" and smi is not None]
    dataset = ReactionDataset(smiles, smiles)
    return dataset


def build_datamodule(args, dataset, tokeniser, max_seq_len):
    test_idxs = range(len(dataset))
    dm = FineTuneReactionDataModule(
        dataset,
        tokeniser,
        args.batch_size,
        max_seq_len,
        val_idxs=[],
        test_idxs=test_idxs
    )
    return dm


def predict(model, test_loader):
    device = "cuda:0" if util.use_gpu else "cpu"
    model = model.to(device)
    model.eval()

    smiles = []
    log_lhs = []
    original_smiles = []

    for b_idx, batch in enumerate(test_loader):
        device_batch = {
            key: val.to(device) if type(val) == torch.Tensor else val for key, val in batch.items()
        }
        with torch.no_grad():
            smiles_batch, log_lhs_batch = model.sample_molecules(device_batch, sampling_alg="beam")

        smiles.extend(smiles_batch)
        log_lhs.extend(log_lhs_batch)
        original_smiles.extend(batch["target_smiles"])

    return smiles, log_lhs, original_smiles


def write_predictions(args, smiles, log_lhs, original_smiles):
    num_data = len(smiles)
    beam_width = len(smiles[0])
    beam_outputs = [[[]] * num_data for _ in range(beam_width)]
    beam_log_lhs = [[[]] * num_data for _ in range(beam_width)]

    for b_idx, (smiles_beams, log_lhs_beams) in enumerate(zip(smiles, log_lhs)):
        for beam_idx, (smi, log_lhs) in enumerate(zip(smiles_beams, log_lhs_beams)):
            beam_outputs[beam_idx][b_idx] = smi
            beam_log_lhs[beam_idx][b_idx] = log_lhs

    df_data = {
        "original_smiles": original_smiles
    }
    for beam_idx, (outputs, log_lhs) in enumerate(zip(beam_outputs, beam_log_lhs)):
        df_data["prediction_" + str(beam_idx)] = beam_outputs[beam_idx]
        df_data["log_likelihood_" + str(beam_idx)] = beam_log_lhs[beam_idx]

    df = pd.DataFrame(data=df_data)
    df.to_pickle(Path(args.products_path))


def main(args):
    print("Building tokeniser...")
    tokeniser = util.load_tokeniser(args.vocab_path, args.chem_token_start_idx)
    print("Finished tokeniser.")

    print("Reading dataset...")
    dataset = build_dataset(args)
    print("Finished dataset.")

    sampler = DecodeSampler(tokeniser, util.DEFAULT_MAX_SEQ_LEN)
    pad_token_idx = tokeniser.vocab[tokeniser.pad_token]

    print("Loading model...")
    model = util.load_bart(args, sampler)
    model.num_beams = args.num_beams
    sampler.max_seq_len = model.max_seq_len
    print("Finished model.")

    print("Building data loader...")
    dm = build_datamodule(args, dataset, tokeniser, model.max_seq_len)
    dm.setup()
    test_loader = dm.test_dataloader()
    print("Finished loader.")

    print("Evaluating model...")
    smiles, log_lhs, original_smiles = predict(model, test_loader)
    write_predictions(args, smiles, log_lhs, original_smiles)
    print("Finished evaluation.")

    print("Printing unknown tokens...")
    tokeniser.print_unknown_tokens()
    print("Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Program level args
    parser.add_argument("--reactants_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--products_path", type=str)
    parser.add_argument("--vocab_path", type=str, default=util.DEFAULT_VOCAB_PATH)
    parser.add_argument("--chem_token_start_idx", type=int, default=util.DEFAULT_CHEM_TOKEN_START)

    # Model args
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_beams", type=int, default=DEFAULT_NUM_BEAMS)

    args = parser.parse_args()
    main(args)

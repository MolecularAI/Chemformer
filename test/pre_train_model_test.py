import pytest
import torch
import random

from molbart.decoder import DecodeSampler
from molbart.tokeniser import MolEncTokeniser
from molbart.models.pre_train import BARTModel

regex = "\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]"

# Use dummy SMILES strings
react_data = [
    "CCO.C",
    "CCCl",
    "C(=O)CBr"
]

# Use dummy SMILES strings
prod_data = [
    "cc",
    "CCl",
    "CBr"
]

model_args = {
    "d_model": 5,
    "num_layers": 2,
    "num_heads": 1,
    "d_feedforward": 32,
    "lr": 0.0001,
    "weight_decay": 0.0,
    "activation": "gelu",
    "num_steps": 1000,
    "max_seq_len": 40
}

random.seed(a=1)
torch.manual_seed(1)


def build_tokeniser():
    tokeniser = MolEncTokeniser.from_smiles(react_data + prod_data, regex, mask_scheme="replace")
    return tokeniser


def test_pos_emb_shape():
    tokeniser = build_tokeniser()
    pad_token_idx = tokeniser.vocab[tokeniser.pad_token]
    sampler = DecodeSampler(tokeniser, model_args["max_seq_len"])
    model = BARTModel(sampler, pad_token_idx, len(tokeniser), **model_args)

    pos_embs = model._positional_embs()

    assert pos_embs.shape[0] == model_args["max_seq_len"]
    assert pos_embs.shape[1] == model.d_model


def test_construct_input_shape():
    tokeniser = build_tokeniser()
    pad_token_idx = tokeniser.vocab[tokeniser.pad_token]
    sampler = DecodeSampler(tokeniser, model_args["max_seq_len"])
    model = BARTModel(sampler, pad_token_idx, len(tokeniser), **model_args)

    token_output = tokeniser.tokenise(react_data, sents2=prod_data, pad=True)
    tokens = token_output["original_tokens"]
    sent_masks = token_output["sentence_masks"]

    token_ids = torch.tensor(tokeniser.convert_tokens_to_ids(tokens)).transpose(0, 1)
    sent_masks = torch.tensor(sent_masks).transpose(0, 1)

    emb = model._construct_input(token_ids, sent_masks)

    assert emb.shape[0] == max([len(ts) for ts in tokens])
    assert emb.shape[1] == 3
    assert emb.shape[2] == model_args["d_model"]


def test_bart_forward_shape():
    tokeniser = build_tokeniser()
    pad_token_idx = tokeniser.vocab[tokeniser.pad_token]
    sampler = DecodeSampler(tokeniser, model_args["max_seq_len"])
    model = BARTModel(sampler, pad_token_idx, len(tokeniser), **model_args)

    react_token_output = tokeniser.tokenise(react_data, mask=True, pad=True)
    react_tokens = react_token_output["masked_tokens"]
    react_pad_mask = react_token_output["masked_pad_masks"]
    react_ids = torch.tensor(tokeniser.convert_tokens_to_ids(react_tokens)).T
    react_mask = torch.tensor(react_pad_mask).T

    prod_token_output = tokeniser.tokenise(prod_data, pad=True)
    prod_tokens = prod_token_output["original_tokens"]
    prod_pad_mask = prod_token_output["original_pad_masks"]
    prod_ids = torch.tensor(tokeniser.convert_tokens_to_ids(prod_tokens)).T
    prod_mask = torch.tensor(prod_pad_mask).T

    batch_input = {
        "encoder_input": react_ids,
        "encoder_pad_mask": react_mask,
        "decoder_input": prod_ids,
        "decoder_pad_mask": prod_mask
    }

    output = model(batch_input)
    model_output = output["model_output"]
    token_output = output["token_output"]

    exp_seq_len = 4  # From expected tokenised length of prod data
    exp_batch_size = len(prod_data)
    exp_dim = model_args["d_model"]
    exp_vocab_size = len(tokeniser)

    assert tuple(model_output.shape) == (exp_seq_len, exp_batch_size, exp_dim)
    assert tuple(token_output.shape) == (exp_seq_len, exp_batch_size, exp_vocab_size)


def test_bart_encode_shape():
    tokeniser = build_tokeniser()
    pad_token_idx = tokeniser.vocab[tokeniser.pad_token]
    sampler = DecodeSampler(tokeniser, model_args["max_seq_len"])
    model = BARTModel(sampler, pad_token_idx, len(tokeniser), **model_args)

    react_token_output = tokeniser.tokenise(react_data, mask=True, pad=True)
    react_tokens = react_token_output["masked_tokens"]
    react_pad_mask = react_token_output["masked_pad_masks"]
    react_ids = torch.tensor(tokeniser.convert_tokens_to_ids(react_tokens)).T
    react_mask = torch.tensor(react_pad_mask).T

    batch_input = {
        "encoder_input": react_ids,
        "encoder_pad_mask": react_mask
    }

    output = model.encode(batch_input)

    exp_seq_len = 9  # From expected tokenised length of react data
    exp_batch_size = len(react_data)
    exp_dim = model_args["d_model"]

    assert tuple(output.shape) == (exp_seq_len, exp_batch_size, exp_dim)


def test_bart_decode_shape():
    tokeniser = build_tokeniser()
    pad_token_idx = tokeniser.vocab[tokeniser.pad_token]
    sampler = DecodeSampler(tokeniser, model_args["max_seq_len"])
    model = BARTModel(sampler, pad_token_idx, len(tokeniser), **model_args)

    react_token_output = tokeniser.tokenise(react_data, mask=True, pad=True)
    react_tokens = react_token_output["masked_tokens"]
    react_pad_mask = react_token_output["masked_pad_masks"]
    react_ids = torch.tensor(tokeniser.convert_tokens_to_ids(react_tokens)).T
    react_mask = torch.tensor(react_pad_mask).T

    encode_input = {
        "encoder_input": react_ids,
        "encoder_pad_mask": react_mask
    }
    memory = model.encode(encode_input)

    prod_token_output = tokeniser.tokenise(prod_data, pad=True)
    prod_tokens = prod_token_output["original_tokens"]
    prod_pad_mask = prod_token_output["original_pad_masks"]
    prod_ids = torch.tensor(tokeniser.convert_tokens_to_ids(prod_tokens)).T
    prod_mask = torch.tensor(prod_pad_mask).T

    batch_input = {
        "decoder_input": prod_ids,
        "decoder_pad_mask": prod_mask,
        "memory_input": memory,
        "memory_pad_mask": react_mask
    }
    output = model.decode(batch_input)

    exp_seq_len = 4  # From expected tokenised length of prod data
    exp_batch_size = len(react_data)
    exp_vocab_size = len(tokeniser)

    assert tuple(output.shape) == (exp_seq_len, exp_batch_size, exp_vocab_size)


def test_calc_token_acc():
    tokeniser = build_tokeniser()
    pad_token_idx = tokeniser.vocab[tokeniser.pad_token]
    sampler = DecodeSampler(tokeniser, model_args["max_seq_len"])
    model = BARTModel(sampler, pad_token_idx, len(tokeniser), **model_args)

    react_token_output = tokeniser.tokenise(react_data[1:], pad=True)
    react_tokens = react_token_output["original_tokens"]
    react_pad_mask = react_token_output["original_pad_masks"]
    target_ids = torch.tensor(tokeniser.convert_tokens_to_ids(react_tokens)).T[1:, :]
    target_mask = torch.tensor(react_pad_mask).T[1:, :]

    # 9 is expected seq len of react data when padded
    token_output = torch.rand([8, len(react_data[1:]), len(tokeniser)])

    """
    Expected outputs 
    CCCl
    C(=O)CBr

    Vocab:
    0 <PAD>
    3 &
    6 C
    7 O
    8 .
    9 Cl
    10 (
    11 =
    12 )
    13 Br
    """

    # Batch element 0
    token_output[0, 0, 6] += 1
    token_output[1, 0, 6] -= 1
    token_output[2, 0, 9] += 1
    token_output[3, 0, 3] += 1
    token_output[4, 0, 0] += 1
    token_output[5, 0, 0] -= 1

    # Batch element 1
    token_output[0, 1, 6] += 1
    token_output[1, 1, 10] += 1
    token_output[2, 1, 11] += 1
    token_output[3, 1, 7] += 1
    token_output[4, 1, 12] -= 1
    token_output[5, 1, 6] += 1
    token_output[6, 1, 13] -= 1
    token_output[7, 1, 3] += 1

    batch_input = {
        "target": target_ids,
        "target_mask": target_mask
    }
    model_output = {
        "token_output": token_output
    }
    token_acc = model._calc_token_acc(batch_input, model_output)

    exp_token_acc = (3 + 6) / (4 + 8)

    assert exp_token_acc == token_acc

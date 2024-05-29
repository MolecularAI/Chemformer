from molbart.data.util import BatchEncoder
from molbart.utils.tokenizers import ReplaceTokensMasker


def test_encoder_no_masking(setup_tokenizer):
    smiles_list = ["CC(=O)O", "O"]
    tokenizer = setup_tokenizer()
    encoder = BatchEncoder(tokenizer, masker=False, max_seq_len=10000)

    id_tensor, mask_tensor = encoder(smiles_list)

    assert tuple(id_tensor.shape) == (9, 2)
    assert tuple(mask_tensor.shape) == (9, 2)

    assert id_tensor[:, 0].tolist()[1:-1] == [tokenizer[token] for token in smiles_list[0]]
    assert id_tensor[0, 0] == tokenizer[tokenizer.special_tokens["start"]]
    assert id_tensor[-1, 0] == tokenizer[tokenizer.special_tokens["end"]]

    assert id_tensor[1, 1] == tokenizer["O"]
    assert id_tensor[0, 1] == tokenizer[tokenizer.special_tokens["start"]]
    assert id_tensor[2, 1] == tokenizer[tokenizer.special_tokens["end"]]
    assert id_tensor[3:, 1].tolist() == [tokenizer[tokenizer.special_tokens["pad"]]] * 6

    assert mask_tensor[:, 0].sum() == 0
    assert mask_tensor[:, 1].tolist() == [0, 0, 0] + [1] * 6


def test_encoder_with_masking(setup_masker, mock_random_choice):
    smiles_list = ["CC(=O)O", "O"]
    tokenizer, masker = setup_masker(ReplaceTokensMasker)
    encoder = BatchEncoder(tokenizer, masker, 10000)

    id_tensor, mask_tensor = encoder(smiles_list, mask=False)

    expected_tokens = [tokenizer[token] for token in smiles_list[0]]
    assert id_tensor[:, 0].tolist()[1:-1] == expected_tokens
    assert id_tensor[1, 1] == tokenizer["O"]

    id_tensor, mask_tensor = encoder(smiles_list, mask=True)

    assert id_tensor[:, 0].tolist()[2:-1:2] == expected_tokens[1::2]
    mask_tokens = [tokenizer[tokenizer.special_tokens["mask"]]] * 4
    assert id_tensor[:, 0].tolist()[1:-1:2] == mask_tokens
    assert id_tensor[0, 0] == tokenizer[tokenizer.special_tokens["start"]]
    assert id_tensor[-1, 0] == tokenizer[tokenizer.special_tokens["end"]]

    assert id_tensor[1, 1] == tokenizer["O"]
    assert id_tensor[0, 1] == tokenizer[tokenizer.special_tokens["start"]]
    assert id_tensor[2, 1] == tokenizer[tokenizer.special_tokens["end"]]
    assert id_tensor[3:, 1].tolist() == [tokenizer[tokenizer.special_tokens["pad"]]] * 6

    assert mask_tensor[:, 0].sum() == 0
    assert mask_tensor[:, 1].tolist() == [0, 0, 0] + [1] * 6


def test_encoder_with_size_limit(setup_tokenizer):
    smiles_list = ["CC(=O)O", "O"]
    tokenizer = setup_tokenizer()
    encoder = BatchEncoder(tokenizer, masker=False, max_seq_len=7)

    id_tensor, mask_tensor = encoder(smiles_list)

    assert tuple(id_tensor.shape) == (7, 2)
    assert tuple(mask_tensor.shape) == (7, 2)

    assert id_tensor[:, 0].tolist()[1:] == [tokenizer[token] for token in smiles_list[0][:-1]]
    assert id_tensor[0, 0] == tokenizer[tokenizer.special_tokens["start"]]
    assert id_tensor[-1, 0] != tokenizer[tokenizer.special_tokens["end"]]

    assert id_tensor[1, 1] == tokenizer["O"]
    assert id_tensor[0, 1] == tokenizer[tokenizer.special_tokens["start"]]
    assert id_tensor[2, 1] == tokenizer[tokenizer.special_tokens["end"]]
    assert id_tensor[3:, 1].tolist() == [tokenizer[tokenizer.special_tokens["pad"]]] * 4

    assert mask_tensor[:, 0].sum() == 0
    assert mask_tensor[:, 1].tolist() == [0, 0, 0] + [1] * 4

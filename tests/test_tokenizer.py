from molbart.utils.tokenizers import ChemformerTokenizer, ReplaceTokensMasker


def test_create_vocab(setup_tokenizer):
    tokenizer = setup_tokenizer()
    expected = {
        "<PAD>": 0,
        "?": 1,
        "^": 2,
        "&": 3,
        "<MASK>": 4,
        "<SEP>": 5,
        "C": 6,
        "O": 7,
        ".": 8,
        "c": 9,
        "Cl": 10,
        "(": 11,
        "=": 12,
        ")": 13,
        "Br": 14,
    }

    vocab = tokenizer.vocabulary

    assert expected == vocab
    assert tokenizer.chem_token_idxs == [6, 7, 8, 9, 10, 11, 12, 13, 14]


def test_add_non_chem_tokens(setup_tokenizer):
    tokenizer = setup_tokenizer(tokens=["<RESERVED>"])
    expected = {
        "<PAD>": 0,
        "?": 1,
        "^": 2,
        "&": 3,
        "<MASK>": 4,
        "<SEP>": 5,
        "<RESERVED>": 6,
        "C": 7,
        "O": 8,
        ".": 9,
        "c": 10,
        "Cl": 11,
        "(": 12,
        "=": 13,
        ")": 14,
        "Br": 15,
    }

    assert expected == tokenizer.vocabulary
    assert tokenizer.chem_token_idxs == [7, 8, 9, 10, 11, 12, 13, 14, 15]


def test_save_and_load(setup_tokenizer, tmpdir):
    test_smiles = ["C.CCCcc1(Br)cccC"]
    filename = str(tmpdir / "vocab.json")
    tokenizer = setup_tokenizer()

    tokenizer.save_vocabulary(filename)

    tokenizer2 = ChemformerTokenizer(filename=filename)

    assert tokenizer(test_smiles)[0].tolist() == tokenizer2(test_smiles)[0].tolist()
    assert tokenizer2.chem_token_idxs == [6, 7, 8, 9, 10, 11, 12, 13, 14]


"""
TODO: these should be moved to wherever the padding is moved
def test_pad_seqs_padding():
    seqs = [[1, 2], [2, 3, 4, 5], []]
    padded, _ = MolEncTokeniser._pad_seqs(seqs, " ")
    expected = [[1, 2, " ", " "], [2, 3, 4, 5], [" ", " ", " ", " "]]

    assert padded == expected


def test_pad_seqs_mask():
    seqs = [[1, 2], [2, 3, 4, 5], []]
    _, mask = MolEncTokeniser._pad_seqs(seqs, " ")
    expected_mask = [[0, 0, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]]

    assert expected_mask == mask
"""


def test_mask_tokens_empty_mask(setup_masker, example_tokens):
    _, masker = setup_masker()
    masked, token_mask = masker(example_tokens, empty_mask=True)
    expected_sum = 0
    mask_sum = sum([sum(m) for m in token_mask])

    assert masked == example_tokens
    assert expected_sum == mask_sum


def test_mask_tokens_replace(setup_masker, mock_random_choice, example_tokens):
    _, masker = setup_masker(ReplaceTokensMasker)
    masked, token_mask = masker(example_tokens)

    expected_masks = [
        [False, True, False, True, False, True, False, False],
        [False, True, False, True, False, True, False],
    ]

    assert expected_masks == token_mask


def test_mask_tokens_span(setup_masker, mock_random_choice, mocker, example_tokens):
    patched_poisson = mocker.patch("molbart.utils.tokenizers.tokenizers.torch.poisson")
    patched_poisson.return_value.long.return_value.item.side_effect = [3, 3, 2, 3]
    _, masker = setup_masker()
    masked, token_mask = masker(example_tokens)

    expected_masks = [
        [False, True, False, True, False],
        [False, True, True, False],
    ]

    assert expected_masks == token_mask


def test_convert_tokens_to_ids(regex_tokens, smiles_data, example_tokens):
    tokeniser = ChemformerTokenizer(smiles=smiles_data[2:3], regex_token_patterns=regex_tokens)
    ids = tokeniser.convert_tokens_to_ids(example_tokens)
    expected_ids = [[2, 6, 7, 8, 9, 10, 1, 3], [2, 6, 6, 5, 6, 11, 3]]

    assert expected_ids == [item.tolist() for item in ids]


def test_tokenise_one_sentence(setup_tokenizer, smiles_data):
    tokeniser = setup_tokenizer()
    tokens = tokeniser.tokenize(smiles_data)
    expected = [
        ["^", "C", "C", "O", ".", "C", "c", "c", "&"],
        ["^", "C", "C", "Cl", "C", "Cl", "&"],
        ["^", "C", "(", "=", "O", ")", "C", "Br", "&"],
    ]

    assert expected == tokens

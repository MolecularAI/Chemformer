import torch

from molbart.modules.decoder import BeamSearchSampler, DecodeSampler


def test_transpose_list():
    list_ = [[1, 2], [3, 4], [5, 6]]
    transposed = DecodeSampler._transpose_list(list_)

    expected = [[1, 3, 5], [2, 4, 6]]

    assert transposed == expected


def test_sort_beams():
    mols = [["b", "z", "g"], ["d", "y", "a"]]
    log_lhs = [[0.3, -0.5, -0.3], [0.3, -0.4, 0.6]]

    sorted_mols, sorted_log_lhs = DecodeSampler._sort_beams(mols, log_lhs)

    exp_mols = [["b", "g", "z"], ["a", "d", "y"]]
    exp_log_lhs = [[0.3, -0.3, -0.5], [0.6, 0.3, -0.4]]

    assert exp_mols == sorted_mols
    assert exp_log_lhs == sorted_log_lhs


def test_greedy_calls_decode(mocker):
    mocked_tokenizer = mocker.patch("molbart.modules.tokenizer.ChemformerTokenizer")
    max_seq_len = 3
    batch_size = 4
    num_tokens = 50

    sampler = DecodeSampler(mocked_tokenizer, max_seq_len)
    sampler.begin_token_id = 1
    sampler.end_token_id = 2
    sampler.pad_token_id = 0

    decode_fn = mocker.MagicMock(
        side_effect=[
            torch.rand((i + 1, batch_size, num_tokens)) for i in range(max_seq_len)
        ]
    )

    mols, log_lhs = sampler.greedy_decode(decode_fn, batch_size)

    expected_calls = max_seq_len - 1
    assert len(decode_fn.call_args_list) == expected_calls


def test_greedy_chooses_max(mocker):
    mocked_tokenizer = mocker.patch("molbart.modules.tokenizer.ChemformerTokenizer")
    max_seq_len = 3
    batch_size = 1

    sampler = DecodeSampler(mocked_tokenizer, max_seq_len)
    sampler.begin_token_id = 1
    sampler.end_token_id = 2
    sampler.pad_token_id = 0

    ret_vals = [
        torch.tensor([[[0.1, 0.1, 0.1, 0.6, 0.1]]]),
        torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0]], [[0.1, 0.3, 0.4, 0.2, 0.0]]]),
    ]

    decode_fn = mocker.MagicMock(side_effect=ret_vals)

    mols, log_lhs = sampler.greedy_decode(decode_fn, batch_size)

    token_ids = decode_fn.call_args_list[-1][0][0]
    token_1, token_2 = tuple(token_ids.tolist())

    exp_token_1 = [1]
    exp_token_2 = [3]

    assert token_1 == exp_token_1
    assert token_2 == exp_token_2


def test_greedy_stops_at_end_token(mocker):
    mocked_tokenizer = mocker.patch("molbart.modules.tokenizer.ChemformerTokenizer")
    max_seq_len = 3
    batch_size = 1

    sampler = DecodeSampler(mocked_tokenizer, max_seq_len)
    sampler.begin_token_id = 1
    sampler.end_token_id = 2
    sampler.pad_token_id = 0

    ret_vals = [
        torch.tensor([[[0.1, 0.1, 0.1, 0.6, 0.1]]]),
        torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0]], [[0.1, 0.3, 0.4, 0.2, 0.0]]]),
        torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0, 0.0, 0.0]],
                [[0.1, 0.1, 0.1, 0.6, 0.1]],
            ]
        ),
    ]

    decode_fn = mocker.MagicMock(side_effect=ret_vals)

    mols, log_lhs = sampler.greedy_decode(decode_fn, batch_size)

    expected_calls = 2
    assert len(decode_fn.call_args_list) == expected_calls


def test_greedy_lls(mocker):
    mocked_tokenizer = mocker.patch("molbart.modules.tokenizer.ChemformerTokenizer")
    max_seq_len = 3
    batch_size = 1

    sampler = DecodeSampler(mocked_tokenizer, max_seq_len)
    sampler.begin_token_id = 1
    sampler.end_token_id = 2
    sampler.pad_token_id = 0

    ret_vals = [
        torch.tensor([[[0.1, 0.1, 0.1, 0.6, 0.1]]]),
        torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0]], [[0.1, 0.3, 0.4, 0.2, 0.0]]]),
        torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0, 0.0, 0.0]],
                [[0.1, 0.1, 0.1, 0.6, 0.1]],
            ]
        ),
    ]

    decode_fn = mocker.MagicMock(side_effect=ret_vals)

    mols, log_lhs = sampler.greedy_decode(decode_fn, batch_size)

    expected_log_lhs = 0.6 + 0.4
    assert log_lhs[0] == expected_log_lhs


def test_beam_calls_decode(mocker):
    mocked_tokenizer = mocker.patch("molbart.modules.tokenizer.ChemformerTokenizer")
    max_seq_len = 3
    batch_size = 4
    num_beams = 5
    num_tokens = 50

    sampler = DecodeSampler(mocked_tokenizer, max_seq_len)
    sampler.begin_token_id = 1
    sampler.end_token_id = 2
    sampler.pad_token_id = 0

    decode_fn = mocker.MagicMock(
        side_effect=lambda ts, ps: torch.rand((ts.shape[0] + 1, batch_size, num_tokens))
    )
    sampler._sort_beams = mocker.MagicMock(return_value=(None, None))

    mols, log_lhs = sampler.beam_decode(decode_fn, batch_size, k=5)

    expected_calls = ((max_seq_len - 2) * num_beams) + 1
    assert len(decode_fn.call_args_list) == expected_calls


def test_beam_chooses_correct_tokens(mocker):
    mocked_tokenizer = mocker.patch("molbart.modules.tokenizer.ChemformerTokenizer")
    max_seq_len = 4
    batch_size = 1
    num_beams = 2

    sampler = DecodeSampler(mocked_tokenizer, max_seq_len)
    sampler.begin_token_id = 1
    sampler.end_token_id = 2
    sampler.pad_token_id = 0

    ret_vals = [
        torch.tensor([[[0.1, 0.1, 0.0, 0.6, 0.5]]]),
        torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.3, 0.7]]]),
        torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0]], [[0.1, 0.0, 0.0, 0.0, 0.9]]]),
        torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0, 0.0, 0.0]],
                [[0.1, 0.3, 0.4, 0.2, 0.0]],
            ]
        ),
        torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0, 1.0, 0.0]],
            ]
        ),
    ]

    decode_fn = mocker.MagicMock(side_effect=ret_vals)
    sampler._sort_beams = mocker.MagicMock(return_value=(None, None))

    mols, log_lhs = sampler.beam_decode(decode_fn, batch_size, k=num_beams)

    token_ids = mocked_tokenizer.convert_ids_to_tokens.call_args_list
    beam_1_token_ids = token_ids[0][0][0]
    beam_2_token_ids = token_ids[1][0][0]

    exp_tokens_1 = [[1, 3, 4, 2]]
    exp_tokens_2 = [[1, 4, 4, 2]]

    assert beam_1_token_ids.tolist() == exp_tokens_1
    assert beam_2_token_ids.tolist() == exp_tokens_2


def test_beam_stops_at_end_token(mocker):
    mocked_tokenizer = mocker.patch("molbart.modules.tokenizer.ChemformerTokenizer")
    max_seq_len = 4
    batch_size = 1
    num_beams = 2

    sampler = DecodeSampler(mocked_tokenizer, max_seq_len)
    sampler.begin_token_id = 1
    sampler.end_token_id = 2
    sampler.pad_token_id = 0

    ret_vals = [
        torch.tensor([[[0.1, 0.1, 0.0, 0.6, 0.5]]]),
        torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.9, 0.1, 0.2]]]),
        torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0]], [[0.1, 0.0, 0.9, 0.0, 0.1]]]),
        torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0, 0.0, 0.0]],
                [[0.1, 0.3, 0.1, 0.0, 0.7]],
            ]
        ),
        torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0, 1.0, 0.0]],
            ]
        ),
    ]

    decode_fn = mocker.MagicMock(side_effect=ret_vals)
    sampler._sort_beams = mocker.MagicMock(return_value=(None, None))

    mols, log_lhs = sampler.beam_decode(decode_fn, batch_size, k=num_beams)

    token_ids = mocked_tokenizer.convert_ids_to_tokens.call_args_list
    beam_1_token_ids = token_ids[0][0][0]
    beam_2_token_ids = token_ids[1][0][0]

    exp_tokens_1 = [[1, 3, 2, 0]]
    exp_tokens_2 = [[1, 4, 2, 0]]

    assert beam_1_token_ids.tolist() == exp_tokens_1
    assert beam_2_token_ids.tolist() == exp_tokens_2


def test_beam_lls(mocker):
    mocked_tokenizer = mocker.patch("molbart.modules.tokenizer.ChemformerTokenizer")
    max_seq_len = 4
    batch_size = 1
    num_beams = 2

    sampler = DecodeSampler(mocked_tokenizer, max_seq_len)
    sampler.begin_token_id = 1
    sampler.end_token_id = 2
    sampler.pad_token_id = 0

    ret_vals = [
        torch.tensor([[[0.1, 0.1, 0.0, 0.6, 0.5]]]),
        torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.9, 0.1, 0.2]]]),
        torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0]], [[0.1, 0.0, 0.9, 0.0, 0.1]]]),
        torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0, 0.0, 0.0]],
                [[0.1, 0.3, 0.1, 0.0, 0.7]],
            ]
        ),
        torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0, 1.0, 0.0]],
            ]
        ),
    ]

    decode_fn = mocker.MagicMock(side_effect=ret_vals)
    sampler._sort_beams = mocker.MagicMock(return_value=(None, None))

    mols, log_lhs = sampler.beam_decode(decode_fn, batch_size, k=num_beams)

    log_lhs = sampler._sort_beams.call_args_list[0][0][1][0]
    beam_1_lls = round(log_lhs[0], 2)
    beam_2_lls = round(log_lhs[1], 2)

    exp_lls_1 = 0.6 + 0.9
    exp_lls_2 = 0.5 + 0.9

    assert beam_1_lls == exp_lls_1
    assert beam_2_lls == exp_lls_2

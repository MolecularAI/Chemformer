""" Module containing helper routines for the DataModules """
from typing import Any, List, Optional, Tuple

import torch
from molbart.utils.tokenizers import ChemformerTokenizer, ListOfStrList, TokensMasker


class BatchEncoder:
    """
    Encodes a sequence for the Chemformer model

    This procedure includes:
        1. Tokenization
        2. Optional masking
        3. Padding
        4. Optional adding separation token to the end
        5. Checking of sequence lengths and possibly truncation
        6. Conversion to pytorch.Tensor

    Encoding is carried out by

    .. code-block::

        id_tensor, mask_tensor = encoder(batch, mask=True)

    where `batch` is a list of strings to be encoded and `mask` is
    a flag that can be used to toggled the masking.

    :param tokenizer: the tokenizer to use
    :param masker: the masker to use
    :param max_seq_len: the maximum allowed list length
    """

    def __init__(
        self,
        tokenizer: ChemformerTokenizer,
        masker: Optional[TokensMasker],
        max_seq_len: int,
    ):
        self._tokenizer = tokenizer
        self._masker = masker
        self._max_seq_len = max_seq_len

    def __call__(
        self,
        batch: List[str],
        mask: bool = False,
        add_sep_token: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self._tokenizer.tokenize(batch)
        if mask and self._masker is not None:
            tokens, _ = self._masker(tokens)
        tokens, pad_mask = self._pad_seqs(tokens, self._tokenizer.special_tokens["pad"])

        if add_sep_token:
            sep_token = self._tokenizer.special_tokens["sep"]
            tokens = [itokens + [sep_token] for itokens in tokens]
            pad_mask = [imasks + [0] for imasks in pad_mask]

        tokens, pad_mask = self._check_seq_len(tokens, pad_mask)
        id_data = self._tokenizer.convert_tokens_to_ids(tokens)
        id_tensor = torch.stack(id_data).transpose(0, 1)
        mask_tensor = torch.tensor(pad_mask, dtype=torch.bool).transpose(0, 1)
        return id_tensor, mask_tensor

    def _check_seq_len(self, tokens: ListOfStrList, mask: List[List[int]]) -> Tuple[ListOfStrList, List[List[int]]]:
        """Warn user and shorten sequence if the tokens are too long, otherwise return original"""

        seq_len = max([len(ts) for ts in tokens])
        if seq_len > self._max_seq_len:
            print(f"WARNING -- Sequence length {seq_len} is larger than maximum sequence size")

            tokens_short = [ts[: self._max_seq_len] for ts in tokens]
            mask_short = [ms[: self._max_seq_len] for ms in mask]

            return tokens_short, mask_short

        return tokens, mask

    @staticmethod
    def _pad_seqs(seqs: List[Any], pad_token: Any) -> Tuple[List[Any], List[int]]:
        pad_length = max([len(seq) for seq in seqs])
        padded = [seq + ([pad_token] * (pad_length - len(seq))) for seq in seqs]
        masks = [([0] * len(seq)) + ([1] * (pad_length - len(seq))) for seq in seqs]
        return padded, masks


def build_attention_mask(enc_length: int, dec_length: int) -> torch.Tensor:
    """
    Building the attention mask for the unified model

    :param enc_length: the length of the encoder
    :param dec_length: the length of the decoder
    :return: the mask tensor
    """
    seq_len = enc_length + dec_length
    enc_mask = torch.zeros((seq_len, enc_length))
    upper_dec_mask = torch.ones((enc_length, dec_length))
    lower_dec_mask = torch.ones((dec_length, dec_length)).triu_(1)
    dec_mask = torch.cat((upper_dec_mask, lower_dec_mask), dim=0)
    mask = torch.cat((enc_mask, dec_mask), dim=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask


def build_target_mask(enc_length: int, dec_length: int, batch_size: int) -> torch.Tensor:
    """
    Build the target mask for the unified model

    :param enc_length: the length of the encoder
    :param dec_length: the length of the decoder
    :param batch_size: the batch size
    :return: the mask tensor
    """
    # Take one and add one because we shift the target left one token
    # So the first token of the target output will be at the same position as the separator token of the input,
    # And the separator token is not present in the output
    enc_mask = [1] * (enc_length - 1)
    dec_mask = [0] * (dec_length + 1)
    mask = [enc_mask + dec_mask] * batch_size
    mask = torch.tensor(mask, dtype=torch.bool).T
    return mask

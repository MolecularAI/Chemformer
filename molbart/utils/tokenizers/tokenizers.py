""" Module containing tokeniser and maskers """
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from pysmilesutils.tokenize import SMILESTokenizer

StrList = List[str]
ListOfStrList = List[StrList]
BoolList = List[bool]
ListOfBoolList = List[BoolList]


class ChemformerTokenizer(SMILESTokenizer):
    """
    Tokenizer for the Chemformer.

    There are a few different features that sets this apart from the `SMILESTokenizer`:
       * It reserves two extra special tokens, "mask" and "sep"
       * It distinguish between chemical and non-chemical tokens

    :param smiles: A list of SMILES that are used to create the vocabulary for the tokenizer. Defaults to None.
    :param tokens:  A list of tokens (strings) that the tokenizer uses when tokenizing SMILES. Defaults to None.
    :param regex_token_patterns: A list of regular expressions that the tokenizer uses when tokenizing SMILES.
    :param beginning_of_smiles_token: Token that is added to beginning of SMILES. Defaults to "^".
    :param end_of_smiles_token: Token that is added to the end of SMILES. Defaults to "&".
    :param padding_token: Token used for padding. Defalts to " ".
    :param unknown_token: Token that is used for unknown ids when decoding encoded data. Defaults to "?".
    :param mask_token: Token that is used by the Masker
    :param sep_token: Token that is used to separate sentences, currently unused
    :param filename: if given and `smiles` is None, load the vocabulary from disc
    :raises: ValueError: If the `encoding_type` is invalid.
    """

    def __init__(
        self,
        smiles: List[str] = None,
        tokens: List[str] = None,
        regex_token_patterns: List[str] = None,
        beginning_of_smiles_token: str = "^",
        end_of_smiles_token: str = "&",
        padding_token: str = "<PAD>",
        unknown_token: str = "?",
        mask_token: str = "<MASK>",
        sep_token: str = "<SEP>",
        filename: str = None,
    ) -> None:
        self._mask_token = mask_token
        self._sep_token = sep_token
        self._chem_start_idx = 6  # Default, number of special tokens + 1
        self._chem_token_idxs: Optional[List[int]] = None
        super().__init__(
            smiles=smiles,
            tokens=tokens,
            regex_token_patterns=regex_token_patterns,
            beginning_of_smiles_token=beginning_of_smiles_token,
            end_of_smiles_token=end_of_smiles_token,
            padding_token=padding_token,
            unknown_token=unknown_token,
            encoding_type="index",
            filename=filename,
        )

    @property
    def chem_token_idxs(self) -> List[int]:
        """Returns the indices of the vocabulary that are chemical tokens"""
        if self._chem_token_idxs is None:
            self._chem_token_idxs = list(range(self._chem_start_idx, len(self.vocabulary)))
        return self._chem_token_idxs

    @property
    def special_tokens(self) -> Dict[str, str]:
        """Returns a dictionary of non-character tokens"""
        return {
            "start": self._beginning_of_smiles_token,
            "end": self._end_of_smiles_token,
            "pad": self._padding_token,
            "unknown": self._unknown_token,
            "mask": self._mask_token,
            "sep": self._sep_token,
        }

    def add_tokens(self, tokens: List[str], regex: bool = False, smiles=None) -> None:
        """Adds tokens to the classes list of tokens.

        The new tokens are added to the front of the token list and take priority over old tokens. Note that that the
        vocabulary of the tokenizer is not updated after the tokens are added,
        and must be updated by calling `create_vocabulary_from_smiles`.

        If `regex` is False, the tokens are interpreted as non-chemical tokens, which distinguish
        them for processing by e.g. the masker.

        :param tokens: List of tokens to be added.
        :param regex: If `True` the input tokens are treated as
                regular expressions and are added to the list of regular expressions
                instead of token list. Defaults to False.
        :param smiles: If a list of smiles is provided, the vocabulary will be created, defaults to None

        :raises ValueError: If any of the tokens supplied are already in the list
                of tokens.
        """
        super().add_tokens(tokens, regex, smiles)
        if not regex:
            self._chem_start_idx += len(tokens)
            self._chem_token_idxs = None

    def _reset_vocabulary(self) -> Dict[str, int]:
        """Create a new tokens vocabulary.

        :return: New tokens vocabulary
        """
        dict_ = {
            self._padding_token: 0,
            self._unknown_token: 1,
            self._beginning_of_smiles_token: 2,
            self._end_of_smiles_token: 3,
            self._mask_token: 4,
            self._sep_token: 5,
        }
        for token in self._tokens:
            dict_.setdefault(token, len(dict_))
        return dict_

    def _state_properties(self) -> Dict[str, Any]:
        """Return properties to reconstruct the internal state of the tokenizer"""
        dict_ = super()._state_properties()
        dict_["chem_start_idx"] = self._chem_start_idx
        return dict_

    def _update_state(self, dict_: Dict[str, Any]) -> None:
        """Update the internal state with properties loaded from disc"""
        super()._update_state(dict_)
        self._chem_start_idx = dict_["chem_start_idx"]
        self._chem_token_idxs = None


class TokensMasker:
    """Base-class for different masking strategies"""

    def __init__(
        self,
        tokenizer: ChemformerTokenizer,
        mask_prob=0.15,
        **kwargs,
    ) -> None:
        self._tokenizer = tokenizer
        self._mask_prob = mask_prob

    def __call__(self, tokens: ListOfStrList, empty_mask=False) -> Tuple[ListOfStrList, ListOfBoolList]:
        return self.mask_tokens(tokens, empty_mask)

    def mask_tokens(self, tokens: ListOfStrList, empty_mask=False) -> Tuple[ListOfStrList, ListOfBoolList]:
        """
        Mask tokenized string.

        The `tokens` parameter should be the output from the `tokenize` function
        of the tokenizer.

        :param tokens: the tokenized strings
        :param empty_mask: if True, do no masking
        :return: the masked tokens and the masks
        """
        if empty_mask:
            mask = [[False] * len(list_) for list_ in tokens]
            return tokens, mask

        masked_tokens = []
        token_masks = []

        start = self._tokenizer.special_tokens["start"]
        end = self._tokenizer.special_tokens["end"]
        for ts in tokens:
            masked, token_mask = self._apply_mask(ts[1:-1])
            masked_tokens.append([start] + masked + [end])
            token_masks.append([False] + token_mask + [False])

        return masked_tokens, token_masks

    def _apply_mask(self, tokens: StrList) -> Tuple[StrList, BoolList]:
        raise NotImplementedError("You need to use one of the sub-classes to mask tokens")


class ReplaceTokensMasker(TokensMasker):
    """
    Class that encapsulate a replacement masking strategy

    It will replace a token with a mask token or a random chemical token with
    a probability determined by the `show_mask_token_prob` argument.

    :param tokenizer: the tokenizer
    :param mask_prob: Probability of token being masked when masking is enabled
    :param show_mask_token_prob: probability of a masked token being replaced with mask token when scheme is "replace"
    """

    def __init__(
        self,
        tokenizer: ChemformerTokenizer,
        mask_prob=0.15,
        **kwargs,
    ) -> None:
        super().__init__(tokenizer, mask_prob)
        self._show_mask_token_prob = kwargs.get("show_mask_token_prob", 1.0)

    def _apply_mask(self, tokens: StrList) -> Tuple[StrList, BoolList]:
        mask_bools = [True, False]
        weights = [self._mask_prob, 1 - self._mask_prob]
        token_mask = random.choices(mask_bools, weights=weights, k=len(tokens))
        masked = [self._mask_token(tokens[idx]) if mask else tokens[idx] for idx, mask in enumerate(token_mask)]
        return masked, token_mask

    def _mask_token(self, token: str) -> str:
        rand = random.random()
        if rand < self._show_mask_token_prob:
            return self._tokenizer.special_tokens["mask"]

        if rand < self._show_mask_token_prob + ((1 - self._show_mask_token_prob) / 2):
            token_idx = random.choice(self._tokenizer.chem_token_idxs)
            return self._tokenizer.decoder_vocabulary[token_idx]

        return token


class SpanTokensMasker(TokensMasker):
    """
    Class that encapsulate a masking strategy that masks a span of tokens

    The length of the span is determined by a Poisson distribution with mean
    equal to the `span_lambda` argument

    :param tokenizer: the tokenizer
    :param mask_prob: Probability of token being masked when masking is enabled
    :param span_lambda: mean for Poisson distribution when sampling a span of tokens
    """

    def __init__(
        self,
        tokenizer: ChemformerTokenizer,
        mask_prob=0.15,
        **kwargs,
    ) -> None:
        super().__init__(tokenizer, mask_prob)
        self._span_lambda = kwargs.get("span_lambda", 3.0)

    def _apply_mask(self, tokens: StrList) -> Tuple[StrList, BoolList]:
        curr_idx = 0
        masked = []
        token_mask = []

        mask_bools = [True, False]
        weights = [self._mask_prob, 1 - self._mask_prob]
        sampled_mask = random.choices(mask_bools, weights=weights, k=len(tokens))

        while curr_idx < len(tokens):
            # If mask, sample from a poisson dist to get length of mask
            if sampled_mask[curr_idx]:
                mask_len = torch.poisson(torch.tensor(self._span_lambda)).long().item()
                masked.append(self._tokenizer.special_tokens["mask"])
                token_mask.append(True)
                curr_idx += mask_len

            # Otherwise don't mask
            else:
                masked.append(tokens[curr_idx])
                token_mask.append(False)
                curr_idx += 1

        return masked, token_mask

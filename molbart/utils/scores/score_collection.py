""" Module containing classes used to score the reaction routes.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import yaml
from omegaconf import ListConfig, OmegaConf

from molbart.utils.base_collection import BaseCollection
from molbart.utils.scores import BaseScore
from molbart.utils.scores.scores import __name__ as score_module

if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional


class ScoreCollection(BaseCollection):
    """
    Store score objects for the chemformer model.

    The scores can be obtained by name

    .. code-block::

        scores = ScoreCollection()
        score = scores['TopKAccuracy']
    """

    _collection_name = "scores"

    def __init__(self) -> None:
        super().__init__()
        self._logger = logging.Logger("score-collection")

    def __repr__(self) -> str:
        if self.selection:
            return f"{self._collection_name} ({', '.join(self.selection)})"

        return f"{self._collection_name} ({', '.join(self.items)})"

    def load(self, score: BaseScore) -> None:  # type: ignore
        """
        Add a pre-initialized score object to the collection

        Args:
            score: the item to add
        """
        if not isinstance(score, BaseScore):
            raise ValueError("Only objects of classes inherited from " "molbart.scores.BaseScore can be added")
        self._items[repr(score)] = score
        self._logger.info(f"Loaded score: {repr(score)}")

    def load_from_config(self, scores_config: ListConfig) -> None:
        """
        Load one or several scores from a configuration dictionary

        The keys are the name of score class. If a score is not
        defined in the ``molbart.utils.scores.scores`` module, the module
        name can be appended, e.g. ``mypackage.scoring.AwesomeScore``.

        The values of the configuration is passed directly to the score
        class along with the ``config`` parameter.

        Args:
            scores_config: Config of scores
        """
        for item in scores_config:
            if isinstance(item, str):
                cls = self.load_dynamic_class(item, score_module)
                obj = cls()
                config_str = ""
            else:
                item = [(key, item[key]) for key in item.keys()][0]
                name, kwargs = item

                x = yaml.load(OmegaConf.to_yaml(kwargs), Loader=yaml.SafeLoader)
                kwargs = self._unravel_list_dict(x)

                cls = self.load_dynamic_class(name, score_module)
                obj = cls(**kwargs)
                config_str = f" with configuration '{kwargs}'"
            self._items[repr(obj)] = obj
            print(f"Loaded score: '{repr(obj)}'{config_str}")

    def score(self, sampled_smiles: List[List[str]], target_smiles: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Apply all scorers in collection to the given sampled and target SMILES.

        Args:
            sampled_smiles: top-N SMILES sampled by a model, such as Chemformer.
            target_smiles: ground truth SMILES.
        Returns:
            A dictionary with all the scores.
        """
        scores = []
        for score in self._items.values():
            scores.append(score(sampled_smiles, target_smiles))
        return self._unravel_list_dict(scores)

""" Module containing classes used to score the reaction routes.
"""
from __future__ import annotations

import logging

from omegaconf import ListConfig, OmegaConf
from typing import Any, Dict
import yaml

from molbart.data import _AbsDataModule
from molbart.data.datamodules import __name__ as data_module
from molbart.utils import data_utils
from molbart.utils.tokenizers import ChemformerTokenizer
from molbart.utils.base_collection import BaseCollection


class DataCollection(BaseCollection):
    """
    Store datamodule object for the chemformer model.

    The datamodule can be obtained by name

    .. code-block::

        datamodule = DataCollection()
    """

    _collection_name = "data"

    def __init__(self, config: OmegaConf, tokenizer: ChemformerTokenizer) -> None:
        super().__init__()
        self._logger = logging.Logger("data-collection")
        self._config = config
        self._tokenizer = tokenizer

    def __repr__(self) -> str:
        if self.selection:
            return f"{self._collection_name} ({', '.join(self.selection)})"

        return f"{self._collection_name} ({', '.join(self.items)})"

    def load(self, datamodule: _AbsDataModule) -> None:  # type: ignore
        """
        Load a datamodule object to the collection

        Args:
            datamodule: the item to add
        """
        if not isinstance(datamodule, _AbsDataModule):
            raise ValueError("Only objects of classes inherited from " "molbart.data._AbsDataModule can be added")
        self._items[repr(datamodule)] = datamodule
        self._logger.info(f"Loaded datamodule: {repr(datamodule)}")

    def load_from_config(self, datamodule_config: ListConfig) -> None:
        """
        Load a datamodule from a configuration dictionary

        The keys are the name of score class. If a score is not
        defined in the ``molbart.data.datamodules`` module, the module
        name can be appended, e.g. ``mypackage.data.AwesomeDataModule``.

        The values of the configuration is passed directly to the datamodule
        class along with the ``config`` parameter.

        Args:
            datamodule_config: Config of the datamodule
        """
        for item in datamodule_config:
            if isinstance(item, str):
                cls = self.load_dynamic_class(item, data_module)
                kwargs = self._set_datamodule_kwargs()
            else:
                item = [(key, item[key]) for key in item.keys()][0]
                name, kwargs = item
                cls = self.load_dynamic_class(name, data_module)

                x = yaml.load(OmegaConf.to_yaml(kwargs), Loader=yaml.SafeLoader)
                kwargs = self._unravel_list_dict(x)
                kwargs.update(self._set_datamodule_kwargs())
                
            obj = cls(**kwargs)
            config_str = f" with configuration '{kwargs}'"
            self._items[repr(obj)] = obj
            print(f"Loaded datamodule: '{repr(obj)}'{config_str}")

    def get_datamodule(self, datamodule_config: ListConfig) -> _AbsDataModule:
        """
        Return the datamodule which has been loaded from the config file
        """
        self.load_from_config(datamodule_config)
        return self.objects()[0]

    def _set_datamodule_kwargs(self) -> Dict[str, Any]:
        """
        Returns a dictionary with kwargs which are general to the _AbsDataModule. 
        These are specified as single parameters in the config file
        """
        reverse = self._config.task == "backward_prediction"
        kwargs = {
            "reverse": reverse,
            "max_seq_len": self._config.get("max_seq_len", data_utils.DEFAULT_MAX_SEQ_LEN),
            "tokenizer": self._tokenizer,
            "augment_prob": self._config.get("augmentation_probability"),
            "augment_prob": self._config.get("augmentation_probability"),
            "unified_model": self._config.model_type == "unified",
            "dataset_path": self._config.data_path,
            "batch_size": self._config.batch_size,
            "train_token_batch_size": self._config.get("train_tokens"),
            "num_buckets": self._config.get("n_buckets"),
            "unified_model": self._config.model_type == "unified",
            "i_chunk": self._config.get("i_chunk", 0),
            "n_chunks": self._config.get("n_chunks", 1),
        }
        return kwargs
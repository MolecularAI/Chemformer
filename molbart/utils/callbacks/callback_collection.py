""" Module containing classes used to score the reaction routes.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import yaml
from omegaconf import ListConfig, OmegaConf
from pytorch_lightning.callbacks import Callback

from molbart.utils.base_collection import BaseCollection
from molbart.utils.callbacks.callbacks import __name__ as callback_module

if TYPE_CHECKING:
    from typing import List


class CallbackCollection(BaseCollection):
    """
    Store callback objects for the chemformer model.

    The callbacks can be obtained by name

    .. code-block::

        callbacks = CallbackCollection()
        callback = callbacks['LearningRateMonitor']
    """

    _collection_name = "callbacks"

    def __init__(self) -> None:
        super().__init__()
        self._logger = logging.Logger("callback-collection")

    def __repr__(self) -> str:
        if self.selection:
            return f"{self._collection_name} ({', '.join(self.selection)})"

        return f"{self._collection_name} ({', '.join(self.items)})"

    def load(self, callback: Callback) -> None:  # type: ignore
        """
        Add a pre-initialized callback object to the collection

        Args:
            callback: the item to add
        """
        if not isinstance(callback, Callback):
            raise ValueError(
                "Only objects of classes inherited from " "pytorch_lightning.callbacks.Callbacks can be added"
            )
        self._items[repr(callback)] = callback
        self._logger.info(f"Loaded callback: {repr(callback)}")

    def load_from_config(self, callbacks_config: ListConfig) -> None:
        """
        Load one or several callbacks from a configuration dictionary

        The keys are the name of callback class. If a callback is not
        defined in the ``molbart.utils.callbacks.callbacks`` module, the module
        name can be appended, e.g. ``mypackage.callbacks.AwesomeCallback``.

        The values of the configuration is passed directly to the callback
        class along with the ``config`` parameter.

        Args:
            callbacks_config: Config of callbacks
        """
        for item in callbacks_config:
            if isinstance(item, str):
                cls = self.load_dynamic_class(item, callback_module)
                obj = cls()
                config_str = ""
            else:
                item = [(key, item[key]) for key in item.keys()][0]
                name, kwargs = item

                x = yaml.load(OmegaConf.to_yaml(kwargs), Loader=yaml.SafeLoader)
                kwargs = self._unravel_list_dict(x)

                cls = self.load_dynamic_class(name, callback_module)
                obj = cls(**kwargs)
                config_str = f" with configuration '{kwargs}'"

            self._items[repr(obj)] = obj
            print(f"Loaded callback: '{repr(obj)}'{config_str}")

""" Module containing classes used to score the reaction routes.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from omegaconf import DictConfig

if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional


class BaseCollection:
    """
    Base class for collection classes (callback collection, score collection, etc.).
    """

    _collection_name = "base"

    def __init__(self) -> None:
        self._items: Dict[str, Any] = {}

    def __repr__(self) -> str:
        return f"{self._collection_name} ({', '.join(self.names)})"

    def load_from_config(self, config: DictConfig) -> None:
        """
        Load one or several items (e.g. score, callback, etc.) from a configuration dictionary

        The keys are the name of item class. If an item is not
        defined in the ``molbart.utils.items.items`` module, the module
        name can be appended, e.g. ``mypackage.item.AwesomeItem``.
        """
        raise NotImplementedError("BaseCollection.load_from_config() not implemented.")

    def names(self) -> List[str]:
        """Return a list of the names of all the loaded items"""
        return list(self._items.keys())

    def objects(self) -> List[Any]:
        """Return a list of all the loaded items"""
        return list(self._items.values())

    @staticmethod
    def load_dynamic_class(
        name_spec: str,
        default_module: Optional[str] = None,
        exception_cls: Any = ValueError,
    ) -> Any:
        """
        Load an object from a dynamic specification.

        The specification can be either:
            ClassName, in-case the module name is taken from the `default_module` argument
        or
            package_name.module_name.ClassName, in-case the module is taken as `package_name.module_name`

        Args:
            name_spec: the class specification
            default_module: the default module
            exception_cls: the exception class to raise on exception
        Returns
            the loaded class
        """
        if "." not in name_spec:
            name = name_spec
            if not default_module:
                raise exception_cls("Must provide default_module argument if not given in name_spec")
            module_name = default_module
        else:
            module_name, name = name_spec.rsplit(".", maxsplit=1)

        try:
            loaded_module = importlib.import_module(module_name)
        except ImportError:
            raise exception_cls(f"Unable to load module: {module_name}")

        if not hasattr(loaded_module, name):
            raise exception_cls(f"Module ({module_name}) does not have a class called {name}")

        return getattr(loaded_module, name)

    @staticmethod
    def _unravel_list_dict(input_data: List[Dict]):
        output = {}
        for data in input_data:
            for key, value in data.items():
                output[key] = value
        return output

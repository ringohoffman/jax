# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Logical type interface for JAX C++ PyTree C-extensions (pytree.pyi)."""

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from typing import Any, ClassVar, Final, TypeVar, overload

_T = TypeVar("_T")
_Children = TypeVar("_Children", bound=Iterable[Any])
_AuxData = TypeVar("_AuxData", bound=Hashable)
_KeyLeafPair = TypeVar("_KeyLeafPair", bound=tuple[Any, Any])
_KeyLeafPairs = TypeVar("_KeyLeafPairs", bound=Iterable[tuple[Any, Any]])


class KeyPath(Hashable):
  """Base class representing a single step in a PyTree key path."""
  def __eq__(self, other: object) -> bool: ...
  def __hash__(self) -> int: ...


class SequenceKey(KeyPath):
  """Key path entry for a sequence element at index ``idx``."""
  idx: int
  def __init__(self, idx: int) -> None: ...
  def __str__(self) -> str: ...
  def __repr__(self) -> str: ...
  __match_args__: ClassVar[Final[tuple[str]]] = ("idx",)


class DictKey(KeyPath):
  """Key path entry for a dict key."""
  key: Any
  def __init__(self, key: Any) -> None: ...
  def __str__(self) -> str: ...
  def __repr__(self) -> str: ...
  __match_args__: ClassVar[Final[tuple[str]]] = ("key",)


class GetAttrKey(KeyPath):
  """Key path entry for an attribute name (e.g. object attribute)."""
  name: str
  def __init__(self, name: str) -> None: ...
  def __str__(self) -> str: ...
  def __repr__(self) -> str: ...
  __match_args__: ClassVar[Final[tuple[str]]] = ("name",)


class FlattenedIndexKey(KeyPath):
  """Key path entry for a flattened index position."""
  key: int
  def __init__(self, key: int) -> None: ...
  def __str__(self) -> str: ...
  def __repr__(self) -> str: ...
  __match_args__: ClassVar[Final[tuple[str]]] = ("key",)


class PyTreeDef:
  """C++ compiled container topology metadata skeleton for a PyTree."""

  @property
  def num_leaves(self) -> int:
    """Number of terminal leaf nodes in the PyTree structure."""
    ...

  @property
  def num_nodes(self) -> int:
    """Total number of container nodes in the PyTree structure."""
    ...

  def unflatten(self, leaves: Iterable[Any], /) -> Any:
    """Reconstruct original PyTree by filling the structure with `leaves`."""
    ...

  def flatten_up_to(self, tree: object | None) -> list[Any]:
    """Flatten `tree` up to the depth defined by this PyTreeDef."""
    ...

  def compose(self, inner: PyTreeDef, /) -> PyTreeDef:
    """Compose this outer PyTreeDef with an inner PyTreeDef."""
    ...

  def children(self) -> list[PyTreeDef]:
    """Return child PyTreeDefs for sub-containers."""
    ...

  def walk(
      self,
      f_node: Callable[[Any, Any], Any],
      f_leaf: Callable[[_T], Any] | None,
      leaves: Iterable[Any],
      /,
  ) -> Any:
    """Traverse PyTree, applying `f_node` at nodes and `f_leaf` at leaves."""
    ...

  def node_data(self) -> tuple[type[Any], Any] | None:
    """Returns None if leaf-pytree, else (type, node_metadata)."""
    ...

  @staticmethod
  def from_node_data_and_children(
      registry: PyTreeRegistry,
      node_data: tuple[type[Any], Any] | None,
      children: Iterable[PyTreeDef],
  ) -> PyTreeDef:
    """Reconstruct a PyTreeDef from node_data and child PyTreeDefs."""
    ...

  def __eq__(self, other: object, /) -> bool: ...
  def __ne__(self, other: object, /) -> bool: ...
  def __hash__(self) -> int: ...
  def __repr__(self) -> str: ...


class PyTreeRegistry:
  """Registry mapping Python container classes to flatten/unflatten hooks."""

  def __init__(
      self,
      enable_none: bool = True,
      enable_tuple: bool = True,
      enable_namedtuple: bool = True,
      enable_list: bool = True,
      enable_dict: bool = True,
  ) -> None: ...

  def flatten(
      self,
      tree: object | None,
      leaf_predicate: Callable[[Any], bool] | None = None,
  ) -> tuple[list[Any], PyTreeDef]:
    """Flatten a PyTree into (list_of_leaves, treedef)."""
    ...

  @overload
  def flatten_with_path(
      self,
      tree: Mapping[Any, _T],
      leaf_predicate: Callable[[KeyPath, Any], bool] | None = None,
  ) -> tuple[list[tuple[KeyPath, _T]], PyTreeDef]: ...

  @overload
  def flatten_with_path(
      self,
      tree: Sequence[_T],
      leaf_predicate: Callable[[KeyPath, Any], bool] | None = None,
  ) -> tuple[list[tuple[KeyPath, _T]], PyTreeDef]: ...

  @overload
  def flatten_with_path(
      self,
      tree: object | None,
      leaf_predicate: Callable[[KeyPath, Any], bool] | None = None,
  ) -> tuple[list[tuple[KeyPath, Any]], PyTreeDef]: ...

  def flatten_one_level(
      self, tree: object | None
  ) -> tuple[Iterable[Any], Any] | None: ...

  def flatten_one_level_with_keys(
      self, tree: object | None
  ) -> tuple[Iterable[_KeyLeafPair], Any] | None: ...

  def register_node(
      self,
      type: type[_T],
      to_iterable: Callable[[_T], tuple[_Children, _AuxData]],
      from_iterable: Callable[[_AuxData, _Children], _T],
      to_iterable_with_keys: (
          Callable[[_T], tuple[_KeyLeafPairs, _AuxData]] | None
      ) = None,
  ) -> Any: ...

  def register_dataclass_node(
      self,
      type: type[Any],
      data_fields: Sequence[str],
      meta_fields: Sequence[str],
      /,
  ) -> Any: ...

  def is_node(self, type: type[Any]) -> bool: ...


_default_registry: PyTreeRegistry
default_registry: Callable[[], PyTreeRegistry]

def treedef_tuple(
    registry: PyTreeRegistry, arg0: Sequence[PyTreeDef], /
) -> PyTreeDef: ...

def all_leaves(registry: PyTreeRegistry, arg1: Iterable[Any], /) -> bool: ...


__all__ = [
    "KeyPath",
    "SequenceKey",
    "DictKey",
    "GetAttrKey",
    "FlattenedIndexKey",
    "PyTreeDef",
    "PyTreeRegistry",
    "default_registry",
    "treedef_tuple",
    "all_leaves",
]

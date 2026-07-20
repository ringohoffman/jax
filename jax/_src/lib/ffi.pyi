# Copyright 2026 The JAX Authors.

"""Python bindings for XLA FFI type stub."""

import enum
from typing import Any
import numpy
from typing_extensions import CapsuleType

class Buffer:
  @property
  def dtype(self) -> numpy.dtype[Any]: ...
  @property
  def ndim(self) -> int: ...
  @property
  def shape(self) -> tuple[int, ...]: ...
  @property
  def writeable(self) -> bool: ...
  def __array__(
      self, dtype: object | None = ..., copy: object | None = ...
  ) -> numpy.ndarray[Any, Any]: ...

class ExecutionStage(enum.Enum):
  INSTANTIATE = 0
  PREPARE = 1
  INITIALIZE = 2
  EXECUTE = 3

class ExecutionContext:
  @property
  def stage(self) -> ExecutionStage: ...
  @property
  def stream(self) -> int: ...

# Copyright 2026 The JAX Authors.

"""XLA client bindings type stub for jax._src.lib.xla_client."""

from typing import Any

_version: int
_ifrt_version: int

class Device:
  id: int
  process_index: int
  platform: str
  device_kind: str
  def memory_stats(self) -> dict[str, int] | None: ...

class Client:
  def devices((self)) -> list[Device]: ...
  def local_devices(self) -> list[Device]: ...
  def process_index(self) -> int: ...
  def device_count(self) -> int: ...

class _XlaModule:
  def collect_garbage(self) -> None: ...

_xla: _XlaModule

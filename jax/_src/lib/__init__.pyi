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

"""Type stub for jax._src.lib."""

from types import ModuleType
from typing import Any

from jax._src.lib import ffi as ffi
from jax._src.lib import guard_lib as guard_lib
from jax._src.lib import jax_jit as jax_jit
from jax._src.lib import pytree as pytree
from jax._src.lib import xla_client as xla_client
from jax._src.lib.xla_client import Device as Device

jaxlib_extension_version: int
ifrt_version: int
cuda_versions: ModuleType | None
cuda_path: str | None

__all__ = [
    "Device",
    "ffi",
    "guard_lib",
    "ifrt_version",
    "jax_jit",
    "jaxlib_extension_version",
    "pytree",
    "xla_client",
]

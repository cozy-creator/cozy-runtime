from __future__ import annotations

from types import TracebackType
from typing import Any, ClassVar, Self

__version__ = "0.0.0-proof"


class _Version:
    cuda = "12.9"


version = _Version()


class _Context:
    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


class device(_Context):
    def __init__(self, name: object) -> None:
        self.type = str(name)

    def __str__(self) -> str:
        return self.type


class _Future:
    value = False

    @classmethod
    def get_swap_module_params_on_conversion(cls) -> bool:
        return cls.value

    @classmethod
    def set_swap_module_params_on_conversion(cls, value: bool) -> None:
        cls.value = value


__future__ = _Future()


class _Cuda:
    @staticmethod
    def is_initialized() -> bool:
        return False

    @staticmethod
    def get_device_capability() -> tuple[int, int]:
        return (0, 0)

    @staticmethod
    def get_device_name() -> str:
        return "none"

    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def device_count() -> int:
        return 0


cuda = _Cuda()


class _Storage:
    _cdata = 1


class Tensor:
    shape = (4,)
    dtype = "bfloat16"
    device = device("meta")
    nbytes = 8

    def untyped_storage(self) -> _Storage:
        return _Storage()

    def storage_offset(self) -> int:
        return 0


class Parameter(Tensor):
    pass


class Module:
    training = False
    _non_persistent_buffers_set: ClassVar[set[str]] = set()

    def __init__(self) -> None:
        self._parameters: dict[str, Parameter] = {}
        self._buffers: dict[str, Tensor] = {}
        self.weight = Parameter()
        self.register_parameter("weight", self.weight)

    def register_parameter(self, name: str, value: Parameter) -> None:
        self._parameters[name] = value
        setattr(self, name, value)

    def register_buffer(self, name: str, value: Tensor, persistent: bool = True) -> None:
        del persistent
        self._buffers[name] = value
        setattr(self, name, value)

    def state_dict(self) -> dict[str, Parameter]:
        return {"weight": self.weight}

    def named_parameters(self, remove_duplicate: bool = False) -> tuple[tuple[str, Parameter], ...]:
        del remove_duplicate
        return (("weight", self.weight),)

    def named_buffers(self, remove_duplicate: bool = False) -> tuple[()]:
        del remove_duplicate
        return ()

    def named_modules(self) -> tuple[tuple[str, Self], ...]:
        return (("", self),)

    @staticmethod
    def compile(*args: object, **kwargs: object) -> None:
        del args, kwargs


class _NN:
    Module = Module
    Parameter = Parameter


nn = _NN()


class TorchFunctionMode(_Context):
    pass


class _Overrides:
    TorchFunctionMode = TorchFunctionMode


overrides = _Overrides()


def compile(*args: Any, **kwargs: Any) -> None:
    del args, kwargs

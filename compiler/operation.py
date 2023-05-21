import enum
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from compiler.codegen import DataType


class Operation(ABC):
    @abstractmethod
    def generate_code(self, f, state):
        raise NotImplementedError()


# TODO rena me to space
class MemoryKind(enum.Enum):
    L3 = enum.auto()
    L2 = enum.auto()
    L1 = enum.auto()

    @property
    def addressable(self):
        return self in (MemoryKind.L2, MemoryKind.L1)

    @property
    def ram(self):
        return self == MemoryKind.L3


@dataclass
class Pointer:
    code: str
    kind: MemoryKind

    def __str__(self):
        return self.code

    def offset(self, offset) -> 'Pointer':
        assert int(offset) == offset
        offset = int(offset)
        return Pointer(f"{self.code} + {offset}", self.kind)


# TODO rename lock to fence
#   and add separate mutex, for IMA exclusion?
@dataclass
class Lock:
    index: int

    def __str__(self):
        return f"LOCKS[{self.index}]"


@dataclass
class ProfileInfo:
    core: str
    name: str


@dataclass
class Profile:
    index: int
    info: ProfileInfo

    def __str__(self):
        return f"PROFILES[{self.index}]"


@dataclass
class CyclesInfo:
    core: str
    kind: str
    name: str


@dataclass
class Cycles:
    index: int
    info: CyclesInfo

    def __str__(self):
        return f"CYCLES[{self.index}]"


@dataclass
class OperationCopy(Operation):
    dest: Pointer
    src: Pointer
    size_bytes: int

    def generate_code(self, f, state):
        kinds = (self.dest.kind, self.src.kind)

        if kinds == (MemoryKind.L2, MemoryKind.L3):
            f.writeln(f"pi_cl_ram_copy_blocking(ram, {self.src}, {self.dest}, {self.size_bytes}, 1);")
            return
        if kinds == (MemoryKind.L3, MemoryKind.L2):
            f.writeln(f"pi_cl_ram_copy_blocking(ram, {self.dest}, {self.src}, {self.size_bytes}, 0);")
            return

        if kinds == (MemoryKind.L1, MemoryKind.L2):
            f.writeln(f"memcpy_cl_l2_l1_blocking({self.src}, {self.dest}, {self.size_bytes}, 1);")
            return
        if kinds == (MemoryKind.L2, MemoryKind.L1):
            f.writeln(f"memcpy_cl_l2_l1_blocking({self.dest}, {self.src}, {self.size_bytes}, 0);")
            return

        if all(k.addressable for k in kinds):
            # TODO figure out DMA for within same memory?
            f.writeln(f"memcpy({self.dest}, {self.src}, {self.size_bytes});")
            return

        raise ValueError(f"Cannot copy [{self.dest}] <- [{self.src}]")


@dataclass
class OperationCopy2D(Operation):
    upper: Pointer
    lower: Pointer
    dir_down: bool

    size_bytes: int

    # stride and length apply to upper
    stride_bytes: int
    length_bytes: int

    def generate_code(self, f, state):
        assert self.upper.kind == MemoryKind.L3 and self.lower.kind == MemoryKind.L2
        f.writeln(
            f"pi_cl_ram_copy_2d_blocking(ram, {self.upper}, {self.lower}, {self.size_bytes}, {self.stride_bytes}, {self.length_bytes}, {int(self.dir_down)});")


@dataclass
class OperationMatmul(Operation):
    b: int
    k: int
    c: int

    input: Pointer
    weight: Pointer
    output: Pointer

    profile: Profile

    def generate_code(self, f, state):
        assert self.b != 1, "Matmul does not work for b=1"
        f.writeln(
            f"run_matmul("
            f"{self.b}, {self.k}, {self.c}, "
            f"{self.weight}, {self.input}, {self.output}, "
            f"&{self.profile}"
            f");"
        )


@dataclass
class OperationLockIncrement(Operation):
    lock: Lock

    def generate_code(self, f, state):
        f.writeln(f"{self.lock}++;")


@dataclass
class OperationLockWait(Operation):
    lock: Lock
    value: int

    def generate_code(self, f, state):
        f.writeln(f"while({self.lock} < {self.value});")


@dataclass
class OperationRecordCycles(Operation):
    cycles: Cycles

    def generate_code(self, f, state):
        f.writeln(f"{self.cycles} = cycles(); // {self.cycles.info}")


@dataclass
class OperationPad(Operation):
    def generate_code(self, f, state):
        f.writeln()


@dataclass
class OperationComment(Operation):
    text: str

    def generate_code(self, f, state):
        for line in self.text.splitlines():
            f.writeln(f"// {line}")

@dataclass
class Tensor:
    dtype: DataType
    shape: Tuple[int, ...]

    offset_elem: int
    strides_elem: Tuple[int, ...]

    def __post_init__(self):
        assert isinstance(self.dtype, DataType)
        assert isinstance(self.shape, tuple)
        assert isinstance(self.offset_elem, int)
        assert isinstance(self.strides_elem, tuple)
        assert len(self.shape) == len(self.strides_elem)

    @staticmethod
    def simple(dtype: DataType, shape: Tuple[int, ...]):
        return Tensor(dtype, shape, 0, simple_strides(shape))

    @property
    def has_simple_strides(self):
        return self.strides_elem == simple_strides(self.shape)

    @property
    def rank(self):
        return len(self.shape)

    @property
    def offset_bytes(self) -> int:
        return map_int(self.offset_elem * self.dtype.size_bytes)

    def stride_bytes(self, axis: int) -> int:
        return map_int(self.strides_elem[axis] * self.dtype.size_bytes)

    @property
    def size_elem(self) -> int:
        return math.prod(self.shape)

    @property
    def size_bytes(self) -> int:
        return map_int(self.size_elem)

    def shape_bytes(self, axis: int) -> int:
        return map_int(self.shape[axis] * self.dtype.size_bytes)

    def transpose(self, a: int, b: int) -> 'Tensor':
        new_shape = list(self.shape)
        new_strides_elem = list(self.strides_elem)

        new_shape[a], new_shape[b] = new_shape[b], new_shape[a]
        new_strides_elem[a], new_strides_elem[b] = new_strides_elem[b], new_strides_elem[a]

        return Tensor(self.dtype, tuple(new_shape), self.offset_elem, tuple(new_strides_elem))

    def __getitem__(self, item):
        assert isinstance(item, tuple)
        assert (len(item) == self.rank)

        new_offset_elem = self.offset_elem
        new_shape = []
        new_strides_elem = []

        for (i, s) in enumerate(item):
            if isinstance(s, int):
                new_offset_elem += self.strides_elem[i] * s
            elif isinstance(s, slice):
                start = s.start if s.start is not None else 0
                stop = s.stop if s.stop is not None else self.shape[i]

                assert s.step is None
                assert 0 <= start <= stop <= self.shape[i]

                new_offset_elem += self.strides_elem[i] * start
                new_strides_elem.append(self.strides_elem[i])
                new_shape.append(stop - start)
            else:
                assert False, f"Invalid axis slice {s}"

        return Tensor(self.dtype, tuple(new_shape), new_offset_elem, tuple(new_strides_elem))


def map_int(f: float) -> int:
    assert float(int(f)) == f, f"Cannot map {f} to int"
    return int(f)


@dataclass
class Buffer:
    shape: tuple
    dtype: DataType
    const: bool

    # whether the buffer is stored transposed in device memory
    transposed: bool

    pointer_l3: Optional[Pointer]

    pointer_l3_expected: Optional[Pointer] = None
    sim_value: Optional[np.array] = None

    def __post_init__(self):
        if self.transposed:
            assert len(self.shape) == 2

    @property
    def size_bytes(self):
        return map_int(math.prod(self.shape) * self.dtype.size_bytes)

    def tensor(self) -> Tensor:
        # TODO is this right?
        simple = Tensor.simple(self.dtype, self.shape)
        return simple

        # if self.transposed:
        #     return simple.transpose(0, 1)
        # else:
        #     return simple


def simple_strides(shape):
    curr = 1
    strides = []
    for size in reversed(shape):
        strides.append(curr)
        curr *= size
    return tuple(reversed(strides))


def main():
    tensor = Tensor.simple(DataType.Int8, (256, 32))
    print(tensor)

    print(tensor[:, :])
    print(tensor[:, :16])
    print(tensor[:16, :])
    print(tensor[4, :16])


if __name__ == '__main__':
    main()

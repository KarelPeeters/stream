import enum
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

from compiler.data_type import DataType


class Operation(ABC):
    @abstractmethod
    def generate_code(self, core: Optional[int], f):
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

    def generate_code(self, core: Optional[int], f):
        kinds = (self.dest.kind, self.src.kind)

        if kinds == (MemoryKind.L2, MemoryKind.L3):
            assert core is None
            f.writeln(f"pi_ram_copy(ram, {self.src}, {self.dest}, {self.size_bytes}, 1);")
            return
        if kinds == (MemoryKind.L3, MemoryKind.L2):
            assert core is None
            f.writeln(f"pi_ram_copy(ram, {self.dest}, {self.src}, {self.size_bytes}, 0);")
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

    def generate_code(self, core: Optional[int], f):
        assert self.upper.kind == MemoryKind.L3 and self.lower.kind == MemoryKind.L2
        assert core is None

        f.writeln(
            f"pi_ram_copy_2d(ram, {self.upper}, {self.lower}, {self.size_bytes}, {self.stride_bytes}, {self.length_bytes}, {int(self.dir_down)});")


@dataclass
class OperationCopy3D(Operation):
    upper: Pointer
    lower: Pointer
    dir_down: bool

    size_0: int
    size_1: int
    size_2: int
    stride_0: int
    stride_1: int

    def generate_code(self, core: Optional[int], f):
        assert self.upper.kind == MemoryKind.L3 and self.lower.kind == MemoryKind.L2
        assert core is None

        shape_str = f".size_0 = {self.size_0}, .size_1 = {self.size_1}, .size_2 = {self.size_2}, .stride_0 = {self.stride_0}, .stride_1 = {self.stride_1}"
        f.writeln(
            f"pi_ram_copy_3d(ram, {self.upper}, {self.lower}, (ShapeCopy3D_t){{{shape_str}}}, {int(self.dir_down)});")


@dataclass
class OperationMatmul(Operation):
    b: int
    k: int
    c: int

    input: Pointer
    weight: Pointer
    output: Pointer

    profile: Profile

    def generate_code(self, core: Optional[int], f):
        assert self.b != 1, "Matmul does not work for b=1"
        f.writeln(
            f"run_matmul("
            f"{self.b}, {self.k}, {self.c}, "
            f"{self.weight}, {self.input}, {self.output}, "
            f"&{self.profile}"
            f");"
        )


@dataclass
class OperationConvPadded(Operation):
    b: int
    k: int
    c: int
    oh: int
    ow: int
    fh: int
    fw: int

    out_stride_b: int
    out_stride_h: int
    out_stride_w: int

    input: Pointer
    weight: Pointer
    output: Pointer

    profile: Profile

    def generate_code(self, core: Optional[int], f):
        # TODO some assert for conv param limitations?
        f.writeln(
            f"run_conv("
            f"{self.b}, {self.k}, {self.c}, {self.oh}, {self.ow}, {self.fh}, {self.fw}, "
            f"{self.out_stride_b}, {self.out_stride_h}, {self.out_stride_w}, "
            f"{self.weight}, {self.input}, {self.output}, "
            f"&{self.profile}"
            f");"
        )


@dataclass
class OperationMemClear(Operation):
    base: Pointer
    size_bytes: int

    def generate_code(self, core: Optional[int], f):
        f.writeln(f"memset({self.base}, 0, {self.size_bytes});")


# TODO just replace with set/wait
@dataclass
class OperationLockIncrement(Operation):
    lock: Lock

    def generate_code(self, core: Optional[int], f):
        f.writeln(f"{self.lock}++;")


@dataclass
class OperationLockWait(Operation):
    lock: Lock
    value: int

    def generate_code(self, core: Optional[int], f):
        f.writeln(f"while({self.lock} < {self.value});")


@dataclass
class OperationRecordCycles(Operation):
    cycles: Cycles

    def generate_code(self, core: Optional[int], f):
        f.writeln(f"{self.cycles} = cycles(); // {self.cycles.info}")


@dataclass
class OperationPad(Operation):
    def generate_code(self, core: Optional[int], f):
        f.writeln()


@dataclass
class OperationComment(Operation):
    text: str

    def generate_code(self, core: Optional[int], f):
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
        return map_int(self.size_elem * self.dtype.size_bytes)

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
                assert 0 <= start <= stop <= self.shape[i], f"Failed to slice {self} with {item}"

                new_offset_elem += self.strides_elem[i] * start
                new_strides_elem.append(self.strides_elem[i])
                new_shape.append(stop - start)
            else:
                assert False, f"Invalid axis slice {s}"

        return Tensor(self.dtype, tuple(new_shape), new_offset_elem, tuple(new_strides_elem))

    def remove_unit_dims(self) -> 'Tensor':
        left_shape = []
        left_strides = []

        for (s, d) in zip(self.shape, self.strides_elem):
            if s > 1:
                left_shape.append(s)
                left_strides.append(d)

        return Tensor(self.dtype, tuple(left_shape), self.offset_elem, tuple(left_strides))

    def combine_adjacent_dims(self) -> 'Tensor':
        if self.rank <= 1:
            return self

        rev_shape = []
        rev_strides_elem = []

        next_size = self.shape[-1]
        next_stride = self.strides_elem[-1]

        for dim in reversed(range(0, self.rank - 1)):
            curr_size = self.shape[dim]
            curr_stride = self.strides_elem[dim]

            if curr_stride == next_size * next_stride:
                # merge
                next_size *= curr_size
                # next_stride stays the same
            else:
                # push
                rev_shape.append(next_size)
                rev_strides_elem.append(next_stride)
                next_size = curr_size
                next_stride = curr_stride

        # push final
        rev_shape.append(next_size)
        rev_strides_elem.append(next_stride)

        # reverse
        shape = tuple(reversed(rev_shape))
        strides_elem = tuple(reversed(rev_strides_elem))

        return Tensor(dtype=self.dtype, shape=tuple(shape), offset_elem=self.offset_elem, strides_elem=strides_elem)

    def simplify_for_copy(self):
        return self.remove_unit_dims().combine_adjacent_dims()


def map_int(f: float) -> int:
    assert float(int(f)) == f, f"Cannot map {f} to int"
    return int(f)


def simple_strides(shape):
    curr = 1
    strides = []
    for size in reversed(shape):
        strides.append(curr)
        curr *= size
    return tuple(reversed(strides))


def main():
    # tensor = Tensor.simple(DataType.Int8, (256, 32))
    # print(tensor)
    #
    # print(tensor[:, :])
    # print(tensor[:, :16])
    # print(tensor[:16, :])
    # print(tensor[4, :16])

    # tensor = Tensor(DataType.Int8, (4, 16, 64, 64), 0, (16 * 64 * 64, 64 * 64, 64, 1))
    # print(tensor)
    # print(tensor.simplify_for_copy())

    padded = Tensor.simple(DataType.Int8, (1, 16, 64 + 2, 64 + 2))


if __name__ == '__main__':
    main()

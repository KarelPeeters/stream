import enum
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from compiler.codegen import DataType


class Operation(ABC):
    @abstractmethod
    def generate_code(self, f, state):
        raise NotImplementedError()


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


@dataclass
class Lock:
    index: int

    def __str__(self):
        return f"LOCKS[{self.index}]"


@dataclass
class Profile:
    index: int

    def __str__(self):
        return f"PROFILES[{self.index}]"


@dataclass
class Cycles:
    index: int
    name: str

    def __str__(self):
        return f"CYCLES[{self.index}]"


@dataclass
class OperationCopy(Operation):
    dest: Pointer
    src: Pointer
    size_bytes: int

    def generate_code(self, f, state):
        # TODO how to allow async copies?
        if self.dest.kind.ram and self.src.kind.addressable:
            f.writeln(f"pi_cl_ram_write_blocking(ram, {self.dest}, {self.src}, {self.size_bytes});")
            return
        if self.dest.kind.addressable and self.src.kind.ram:
            f.writeln(f"pi_cl_ram_read_blocking(ram, {self.src}, {self.dest}, {self.size_bytes});")
            return
        if self.dest.kind.addressable and self.src.kind.addressable:
            f.writeln(f"memcpy({self.dest}, {self.src}, {self.size_bytes});")
            return

        raise ValueError(f"Cannot copy [{self.dest}] <- [{self.src}]")


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
        f.writeln(f"{self.cycles} = cycles(); // {self.cycles.name}")


@dataclass
class OperationPad(Operation):
    def generate_code(self, f, state):
        f.writeln()


@dataclass
class Buffer:
    shape: tuple
    dtype: DataType
    const: bool

    # TODO remove this 1-to-1 correspondence between buffers and pointers in various memory levels
    pointer_l1: Optional[Pointer]
    pointer_l2: Optional[Pointer]
    pointer_l3: Optional[Pointer]

    pointer_l3_expected: Optional[Pointer] = None
    sim_value: Optional[np.array] = None

    @property
    def size_bytes(self):
        size_f = math.prod(self.shape) * self.dtype.size_bytes
        assert float(int(size_f)) == size_f
        return int(size_f)

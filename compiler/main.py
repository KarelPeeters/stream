import enum
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np
import onnx
from networkx import DiGraph
from onnx import ModelProto
from zigzag.classes.cost_model.cost_model import CostModelEvaluation
from zigzag.classes.mapping.combined_mapping import Mapping

from compiler.codegen import DataType, array_to_str
from compiler.ima_simulate import random_ima_input, random_ima_weight, ima_matmul
from stream.classes.hardware.architecture.accelerator import Accelerator
from stream.classes.workload.computation_node import ComputationNode


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


@dataclass
class Pointer:
    symbol: str
    kind: MemoryKind

    def __str__(self):
        return self.symbol


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
        # TODO generate async copy instead?
        # TODO support L3 here too?
        assert self.dest.kind.addressable and self.src.kind.addressable
        f.writeln(f"memcpy({self.dest}, {self.src}, {self.size_bytes});")


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


class Output:
    def __init__(self, f):
        self.f = f
        self.indent = ""

    def write(self, s: str):
        if s.strip() == "":
            self.f.write(s)
        else:
            self.f.write(self.indent + s)

    def writeln(self, s: Optional[str] = None):
        if s is None:
            self.f.write("\n")
        else:
            self.write(s)
            self.f.write("\n")

    def __enter__(self):
        self.indent += "    "
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.indent = self.indent[:-4]


@dataclass
class Buffer:
    shape: tuple
    dtype: DataType

    # TODO remove this 1-to-1 correspondence between buffers and pointers in L1 and L2
    pointer_l1: Optional[Pointer]
    pointer_l2: Optional[Pointer]
    pointer_l2_expected: Optional[Pointer] = None

    input: bool = False
    constant: bool = False
    sim_value: Optional[np.array] = None

    @property
    def size_bytes(self):
        size_f = math.prod(self.shape) * self.dtype.size_bytes
        assert float(int(size_f)) == size_f
        return int(size_f)

    @property
    def computed(self) -> bool:
        return not (self.input or self.constant)


class State:
    def __init__(self, onnx_model: ModelProto, core_count: int, sim: bool):
        self.onnx_model = onnx_model
        self.sim = sim

        self.buffers: Dict[str, Buffer] = {}
        self.operations_per_core: List[List[Operation]] = [[] for _ in range(core_count)]

        self.next_lock = 0
        self.next_profile = 0
        self.cycle_names = []
        self.meta_frozen = False

        # define inputs and constants
        for input in onnx_model.graph.input:
            shape = tuple(d.dim_value for d in list(input.type.tensor_type.shape.dim))
            buffer = self.define_buffer(input.name, shape, DataType.Int8)

            buffer.input = True
            buffer.sim_value = random_ima_input(shape)
        for const in onnx_model.graph.initializer:
            shape = tuple(const.dims)
            buffer = self.define_buffer(const.name, shape, DataType.Int4)

            buffer.constant = True
            buffer.sim_value = random_ima_weight(shape)

        # built-in meta vars
        self.cycles_start_init = self.new_cycles("start init")
        self.cycles_end_init = self.new_cycles("end init")

    def get_buffer(self, name):
        return self.buffers[name]

    def define_buffer(self, name, shape: tuple, dtype: DataType) -> Buffer:
        if name in self.buffers:
            raise KeyError(f"Buffer {name} already defined")

        pointer_l1 = Pointer(f"DATA_{name_to_c_upper(name)}_L1", MemoryKind.L1)
        pointer_l2 = Pointer(f"DATA_{name_to_c_upper(name)}_L2", MemoryKind.L2)

        buffer = Buffer(shape=shape, dtype=dtype, pointer_l1=pointer_l1, pointer_l2=pointer_l2)
        self.buffers[name] = buffer
        return buffer

    def push_operation(self, core: int, operation: Operation):
        self.operations_per_core[core].append(operation)

    # TODO reduce code duplication here
    def new_lock(self) -> Lock:
        assert not self.meta_frozen
        index = self.next_lock
        self.next_lock += 1
        return Lock(index)

    def new_cycles(self, name: str) -> Cycles:
        assert not self.meta_frozen
        index = len(self.cycle_names)
        self.cycle_names.append(name)
        return Cycles(index, name)

    def new_profile(self) -> Profile:
        assert not self.meta_frozen
        index = self.next_profile
        self.next_profile += 1
        return Profile(index)

    def freeze_meta(self):
        self.meta_frozen = True


def generate_includes(f):
    f.writeln('#include "pmsis.h"')
    f.writeln('#include <bsp/bsp.h>')
    f.writeln('#include "run_layer.h"')
    f.writeln('#include "layer_weights.h"')


def name_to_c_upper(name: str):
    return name.replace(".", "_").replace("/", "_").replace("::", "_").replace(":", "_").upper()


def visit_node(state: State, cn: ComputationNode, zcme: CostModelEvaluation):
    input_name, weight_name, _ = cn.input_names
    output_name, = cn.output_names

    input = state.get_buffer(input_name)
    weight = state.get_buffer(weight_name)

    output_loop_dims = cn.operand_tensors['O'].loop_ranges
    output_shape = []
    for start, end in output_loop_dims:
        assert start == 0
        output_shape.append(end)

    output = state.define_buffer(output_name, tuple(output_shape), dtype=DataType.Int8)

    loop_ranges = cn.loop_ranges
    temporal = zcme.temporal_mapping
    spatial = zcme.spatial_mapping

    dim_size = cn.loop_dim_size

    core = cn.get_core_allocation()

    if cn.equation == 'O[b][k]+=A[b][c]*B[c][k]':
        # copy inputs
        state.push_operation(core, OperationCopy(input.pointer_l1, input.pointer_l2, input.size_bytes))
        state.push_operation(core, OperationCopy(weight.pointer_l1, weight.pointer_l2, weight.size_bytes))

        # real operation
        state.push_operation(core, OperationMatmul(
            b=dim_size["B"], k=dim_size["K"], c=dim_size["C"],
            weight=weight.pointer_l1, input=input.pointer_l1, output=output.pointer_l1,
            profile=state.new_profile(),
        ))

        # copy output
        state.push_operation(core, OperationCopy(output.pointer_l2, output.pointer_l1, output.size_bytes))

        # simulate
        if state.sim:
            output.sim_value = ima_matmul(input.sim_value, weight.sim_value)
    else:
        raise ValueError(f"Unrecognised equation {cn.equation}")


def generate_meta(f, state: State):
    f.writeln(f"volatile u32 LOCKS[{state.next_lock}] = {{}};")
    f.writeln(f"u32 CYCLES[{len(state.cycle_names)}] = {{}};")
    f.writeln(f"struct Profile PROFILES[{state.next_profile}] = {{}};")


def generate_func_init(f, state: State):
    f.writeln("void generated_init() {")

    with f:
        OperationRecordCycles(state.cycles_start_init).generate_code(f, state)
        # put init code here?
        # TODO maybe switch to list of operations for init too?
        OperationRecordCycles(state.cycles_end_init).generate_code(f, state)

    f.writeln("}")


def generate_func_final(f, state: State):
    f.writeln("void generated_final() {")
    with f:
        for name, buffer in state.buffers.items():
            if buffer.pointer_l2_expected is not None:
                f.writeln(
                    f"verify_output({buffer.pointer_l2_expected}, {buffer.pointer_l2}, {buffer.size_bytes}, \"{name}\");")

        f.writeln()

        # print named cycle counters
        for index, name in enumerate(state.cycle_names):
            cycles = Cycles(index, name)
            f.writeln(f"printf(\"== profile == %d: {name}\\n\", {cycles});")

    f.writeln("}")


def generate_func_core(f, state: State, core):
    f.writeln(f"void generated_core_{core}() {{")
    with f:
        for operation in state.operations_per_core[core]:
            operation.generate_code(f, state)
    f.writeln("}")


def generate_data(f, d, state: State):
    for name, buffer in state.buffers.items():
        name = name_to_c_upper(name)

        init = ""
        if buffer.sim_value is not None:
            sim_init = f" = DATA_{name}"
            values_str = array_to_str(buffer.sim_value.flatten(), buffer.dtype)
            print(f"#define DATA_{name} {values_str}", file=d)

            if buffer.computed:
                pointer_l2_expected = Pointer(f"DATA_{name_to_c_upper(name)}_EXPECTED_L2", MemoryKind.L2)
                buffer.pointer_l2_expected = pointer_l2_expected
                f.writeln(f"PI_L2 i8 {pointer_l2_expected}[{buffer.size_bytes}]{sim_init};")
            else:
                init = sim_init

        f.writeln(f"PI_L2 i8 {buffer.pointer_l2}[{buffer.size_bytes}]{init};")
        f.writeln(f"PI_L1 i8 {buffer.pointer_l1}[{buffer.size_bytes}];")


def generate_code(state: State):
    source_path = r"\\wsl.localhost\Ubuntu\home\karel\new-attempt\pulp-sdk\applications\custom\generated.c"
    data_path = r"\\wsl.localhost\Ubuntu\home\karel\new-attempt\pulp-sdk\applications\custom\layer_weights.h"

    with open(source_path, "w") as source_file, open(data_path, "w") as data_file:
        f = Output(source_file)

        generate_includes(f)
        f.writeln()
        generate_meta(f, state)
        f.writeln()
        generate_data(f, data_file, state)
        f.writeln()

        generate_func_init(f, state)
        f.writeln()
        generate_func_final(f, state)
        f.writeln()

        for core in range(len(state.operations_per_core)):
            generate_func_core(f, state, core)
            f.writeln()


def compiler(onnx_path, scme, node_hw_performances):
    print("compiler")
    print(onnx_path)
    print(scme)
    print(node_hw_performances)

    assert onnx_path.endswith(".onnx")
    onnx_model = onnx.load(onnx_path, load_external_data=False)
    # noinspection PyUnresolvedReferences
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    workload: DiGraph = scme.workload
    accelerator: Accelerator = scme.accelerator

    np.random.seed(0)
    state = State(onnx_model, len(accelerator.cores), sim=True)

    for cn in workload:
        cn: ComputationNode
        print(cn)

        core = cn.core_allocation

        key_node = next(other for other in node_hw_performances.keys() if other.id[0] == cn.id[0])
        zcme = next(m for c, m in node_hw_performances[key_node].items() if c.id == core)
        print(f"  zcme: {zcme}")

        mapping: Mapping = zcme.mapping
        mapping_temporal = mapping.temporal_mapping
        mapping_spatial = mapping.spatial_mapping

        print(f"  mapping: {mapping}")

        # dependency edges: (node_from, node_to)
        print(f"  incoming edge nodes: {list(a for a, _ in workload.in_edges(cn))}")
        print(f"  outgoing edge nodes: {list(b for _, b in workload.out_edges(cn))}")

        print(f"  inputs: {cn.input_names}")
        print(f"  outputs: {cn.output_names}")

        visit_node(state, cn, zcme)

    # insert some additional profiling
    for core in range(len(state.operations_per_core)):
        if len(state.operations_per_core[core]) > 0:
            cycles_start = state.new_cycles(f"start core_{core}")
            cycles_end = state.new_cycles(f"end core_{core}")
            state.operations_per_core[core].insert(0, OperationRecordCycles(cycles_start))
            state.operations_per_core[core].append(OperationRecordCycles(cycles_end))

    # insert some dummy locking stuff
    # lock0 = state.new_lock()
    # lock1 = state.new_lock()
    #
    # state.operations_per_core[0][0:0] = [
    #     OperationLockWait(lock1, 1),
    #     OperationLockIncrement(lock0),
    #     OperationLockWait(lock1, 2),
    #     OperationLockIncrement(lock0),
    #     OperationLockIncrement(lock0),
    #     OperationLockWait(lock1, 3),
    # ]
    # state.operations_per_core[1][0:0] = [
    #     OperationLockIncrement(lock1),
    #     OperationLockWait(lock0, 1),
    #     OperationLockIncrement(lock1),
    #     OperationLockWait(lock0, 3),
    #     OperationLockIncrement(lock1),
    # ]

    print("Allocated buffers:")
    for name, buffer in state.buffers.items():
        print(f"  {name}: {buffer.shape}")

    print("Generating code")
    state.freeze_meta()
    generate_code(state)

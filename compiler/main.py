import os
import subprocess
from typing import Optional, Dict, List

import numpy as np
import onnx
from networkx import DiGraph
from onnx import ModelProto
from zigzag.classes.cost_model.cost_model import CostModelEvaluation
from zigzag.classes.mapping.combined_mapping import Mapping

from compiler.codegen import DataType, array_to_bytes
from compiler.ima_simulate import random_ima_input, random_ima_weight, ima_matmul
from compiler.operation import Operation, MemoryKind, Pointer, Lock, Profile, Cycles, OperationCopy, OperationMatmul, \
    Buffer, OperationPad, CyclesInfo, ProfileInfo, OperationRecordCycles
from compiler.plot_profile import plot_profile
from stream.classes.hardware.architecture.accelerator import Accelerator
from stream.classes.workload.computation_node import ComputationNode


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


class Allocator:
    def __init__(self):
        self.next_offset = 0

    def alloc(self, size):
        offset = self.next_offset
        self.next_offset += size
        return offset

    def __len__(self):
        return self.next_offset


class State:
    def __init__(self, onnx_model: ModelProto, core_count: int, sim: bool):
        self.onnx_model = onnx_model
        self.sim = sim

        self.buffers: Dict[str, Buffer] = {}
        self.operations_per_core: List[List[Operation]] = [[] for _ in range(core_count)]

        self.next_lock = 0
        self.profile_infos: List[ProfileInfo] = []
        self.cycle_infos: List[CyclesInfo] = []
        self.meta_frozen = False

        # define inputs and constants
        for input in onnx_model.graph.input:
            shape = tuple(d.dim_value for d in list(input.type.tensor_type.shape.dim))
            buffer = self.define_buffer(input.name, shape, DataType.Int8, const=True)

            buffer.input = True
            buffer.sim_value = random_ima_input(shape)
        for const in onnx_model.graph.initializer:
            shape = tuple(const.dims)
            buffer = self.define_buffer(const.name, shape, DataType.Int4, const=True)

            buffer.constant = True
            buffer.sim_value = random_ima_weight(shape)

        # built-in meta vars
        # self.cycles_start_init = self.new_cycles("start init")
        # self.cycles_end_init = self.new_cycles("end init")

    def get_buffer(self, name):
        return self.buffers[name]

    def define_buffer(self, name, shape: tuple, dtype: DataType, const: bool) -> Buffer:
        if name in self.buffers:
            raise KeyError(f"Buffer {name} already defined")
        upper = name_to_c_upper(name)

        pointer_l1 = Pointer(f"L1_BUFFER_{upper}", MemoryKind.L1)
        pointer_l2 = Pointer(f"L2_BUFFER_{upper}", MemoryKind.L2)
        if const:
            pointer_l3 = Pointer(f"(L3_CONST_START + L3_CONST_OFFSET_{upper})", MemoryKind.L3)
        else:
            pointer_l3 = Pointer(f"(L3_DYN_START + L3_DYN_OFFSET_{upper})", MemoryKind.L3)

        buffer = Buffer(
            shape=shape, dtype=dtype, const=const,
            pointer_l1=pointer_l1, pointer_l2=pointer_l2, pointer_l3=pointer_l3
        )
        self.buffers[name] = buffer
        return buffer

    def push_operation(self, core: int, operation: Operation):
        self.operations_per_core[core].append(operation)

    def push_cycles(self, core: int, kind: str, name: str):
        info = CyclesInfo(core=f"core_{core}", kind=kind, name=name)
        cycles = self.new_cycles(info)
        self.push_operation(core, OperationRecordCycles(cycles))

    # TODO reduce code duplication here
    def new_lock(self) -> Lock:
        assert not self.meta_frozen
        index = self.next_lock
        self.next_lock += 1
        return Lock(index)

    def new_profile(self, info: ProfileInfo) -> Profile:
        assert isinstance(info, ProfileInfo)
        assert not self.meta_frozen

        index = len(self.profile_infos)
        self.profile_infos.append(info)
        return Profile(index, info)

    def new_cycles(self, info: CyclesInfo) -> Cycles:
        assert isinstance(info, CyclesInfo)
        assert not self.meta_frozen

        index = len(self.cycle_infos)
        self.cycle_infos.append(info)
        return Cycles(index, info)

    def freeze_meta(self):
        self.meta_frozen = True


def generate_includes(f):
    f.writeln('#include "pmsis.h"')
    f.writeln('#include <bsp/bsp.h>')
    f.writeln('#include "run_layer.h"')
    f.writeln('#include "util.h"')


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

    output = state.define_buffer(output_name, tuple(output_shape), dtype=DataType.Int8, const=False)

    loop_ranges = cn.loop_ranges
    temporal = zcme.temporal_mapping
    spatial = zcme.spatial_mapping

    dim_size = cn.loop_dim_size

    core = cn.get_core_allocation()

    if cn.equation == 'O[b][k]+=A[b][c]*B[c][k]':
        # copy inputs
        state.push_cycles(core, "start", "input")
        state.push_operation(core, OperationCopy(input.pointer_l2, input.pointer_l3, input.size_bytes))
        state.push_operation(core, OperationCopy(input.pointer_l1, input.pointer_l2, input.size_bytes))
        state.push_operation(core, OperationCopy(weight.pointer_l2, weight.pointer_l3, weight.size_bytes))
        state.push_operation(core, OperationCopy(weight.pointer_l1, weight.pointer_l2, weight.size_bytes))
        state.push_cycles(core, "end", "input")

        # real operation
        state.push_cycles(core, "start", "calc")
        state.push_operation(core, OperationMatmul(
            b=dim_size["B"], k=dim_size["K"], c=dim_size["C"],
            weight=weight.pointer_l1, input=input.pointer_l1, output=output.pointer_l1,
            profile=state.new_profile(ProfileInfo(core=f"core_{core}_mm", name="matmul")),
        ))
        state.push_cycles(core, "end", "calc")

        # copy output
        state.push_cycles(core, "start", "output")
        state.push_operation(core, OperationCopy(output.pointer_l2, output.pointer_l1, output.size_bytes))
        state.push_operation(core, OperationCopy(output.pointer_l3, output.pointer_l2, output.size_bytes))
        state.push_operation(core, OperationPad())
        state.push_cycles(core, "end", "output")

        # simulate
        if state.sim:
            output.sim_value = ima_matmul(input.sim_value, weight.sim_value)
    else:
        raise ValueError(f"Unrecognised equation {cn.equation}")


def generate_meta(f, state: State):
    f.writeln("static struct pi_device *ram = NULL;")
    f.writeln(f"static volatile u32 LOCKS[{state.next_lock}] = {{}};")
    f.writeln(f"static u32 CYCLES[{len(state.cycle_infos)}] = {{}};")
    f.writeln(f"static struct Profile PROFILES[{len(state.profile_infos)}] = {{}};")


def generate_func_init(f, state: State):
    f.writeln("i32 generated_init_fabric(struct pi_device *ram_param, u32 l3_const_start, u32 l3_const_file_size) {")

    with f:
        # TODO re-enable once we can get this synchronized
        # OperationRecordCycles(state.cycles_start_init).generate_code(f, state)

        f.writeln("ram = ram_param;")
        f.writeln()

        f.writeln("if (l3_const_file_size != L3_CONST_SIZE) {")
        with f:
            f.writeln("printf(\"ERROR: L3 const size mismatch\\n\");")
            f.writeln("return -1;")
        f.writeln("}")
        f.writeln()

        f.writeln("L3_CONST_START = l3_const_start;")
        f.writeln("if (pi_ram_alloc(ram_param, &L3_DYN_START, L3_DYN_SIZE)) {")
        with f:
            f.writeln("printf(\"ERROR: Failed to allocate L3_DYN\\n\");")
            f.writeln("return -2;")
        f.writeln("}")
        f.writeln()

        # TODO re-enable once we can get this synchronized
        # OperationRecordCycles(state.cycles_end_init).generate_code(f, state)

        f.writeln("return 0;")
    f.writeln("}")
    f.writeln()

    f.writeln("void generated_init_cluster() {}")


def generate_func_final(f, state: State):
    f.writeln("void generated_final_cluster() {}")
    f.writeln()

    f.writeln("void generated_final_fabric() {")
    with f:
        # verify outputs
        for name, buffer in state.buffers.items():
            if not buffer.const and buffer.pointer_l3_expected is not None:
                f.writeln(
                    f"verify_output(ram, {buffer.pointer_l3_expected}, {buffer.pointer_l3}, {buffer.size_bytes}, \"{name}\");")

        f.writeln()

        # print named cycle counters
        for index, info in enumerate(state.cycle_infos):
            cycles = Cycles(index, info)
            f.writeln(f"printf(\"== profile == %d == {info.core} == {info.kind} == {info.name}\\n\", {cycles});")

        # print profiles
        for index, info in enumerate(state.profile_infos):
            profile = Profile(index, info)
            f.writeln(f"print_profile(&{profile}, \"{info.core}\", \"{info.name}\");")

    f.writeln("}")


def generate_func_core(f, state: State, core):
    f.writeln(f"void generated_core_{core}() {{")
    with f:
        for operation in state.operations_per_core[core]:
            operation.generate_code(f, state)
    f.writeln("}")


def generate_data(f, state: State, d_bin):
    alloc_const = Allocator()
    alloc_dyn = Allocator()

    for name, buffer in state.buffers.items():
        upper = name_to_c_upper(name)

        f.writeln(f"PI_L1 i8 {buffer.pointer_l1}[{buffer.size_bytes}];")
        f.writeln(f"PI_L2 i8 {buffer.pointer_l2}[{buffer.size_bytes}];")

        if buffer.sim_value is not None:
            offset = alloc_const.alloc(buffer.size_bytes)
            offset_name = f"L3_CONST_OFFSET_{upper}"
            f.writeln(f"#define {offset_name} {offset}")

            pointer = Pointer(f"(L3_CONST_START + {offset_name})", MemoryKind.L3)
            if buffer.const:
                assert buffer.pointer_l3 == pointer
            else:
                buffer.pointer_l3_expected = pointer

            value_bytes = array_to_bytes(buffer.sim_value.flatten(), buffer.dtype)
            assert len(value_bytes) == buffer.size_bytes
            d_bin.write(value_bytes)

        if not buffer.const:
            offset = alloc_dyn.alloc(buffer.size_bytes)
            offset_name = f"L3_DYN_OFFSET_{upper}"
            # TODO split into offset and start again
            f.writeln(f"#define {offset_name} {offset}")

    f.writeln()
    f.writeln("static u32 L3_CONST_START = 0;")
    f.writeln("static u32 L3_DYN_START = 0;")
    f.writeln(f"#define L3_CONST_SIZE {len(alloc_const)}")
    f.writeln(f"#define L3_DYN_SIZE {len(alloc_dyn)}")


def generate_code(state: State, project_path):
    path_source = os.path.join(project_path, "generated.c")
    path_data_bin = os.path.join(project_path, "generated_data.bin")

    with open(path_source, "w") as file_source, open(path_data_bin, "wb") as file_data_bin:
        f = Output(file_source)

        generate_includes(f)
        f.writeln()
        generate_meta(f, state)
        f.writeln()
        generate_data(f, state, file_data_bin)
        f.writeln()

        generate_func_init(f, state)
        f.writeln()
        generate_func_final(f, state)
        f.writeln()

        for core in range(len(state.operations_per_core)):
            generate_func_core(f, state, core)
            f.writeln()


def compile_and_run(onnx_path, scme, node_hw_performances, pulp_sdk_path, project_path, run: bool):
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
    # for core in range(len(state.operations_per_core)):
    #     if len(state.operations_per_core[core]) > 0:
    #         cycles_start = state.new_cycles(f"start core_{core}")
    #         cycles_end = state.new_cycles(f"end core_{core}")
    #         state.operations_per_core[core].insert(0, OperationRecordCycles(cycles_start))
    #         state.operations_per_core[core].append(OperationRecordCycles(cycles_end))

    print("Allocated buffers:")
    for name, buffer in state.buffers.items():
        print(f"  {name}: {buffer.shape}")

    print("Generating code")
    state.freeze_meta()

    wsl_home = subprocess.check_output(["wsl", "wslpath", "-w", "~"]).decode("utf-8").strip()
    generate_code(state, project_path.replace("~", wsl_home))

    if run:
        print("Running code")
        commands = [
            f"cd {pulp_sdk_path}",
            "source configs/pulp-open.sh",
            "export PATH=/opt/riscv/bin:$PATH",
            "export PULP_RISCV_GCC_TOOLCHAIN=/opt/riscv/",

            f"cd {project_path}",
            # TODO add clean if number of cores changed
            "make all run"
        ]
        result = subprocess.run(["wsl", *"; ".join(commands).split(" ")], stdout=subprocess.PIPE)
        stdout = result.stdout.decode("utf-8")

        print(stdout)
        with open("outputs/stdout.txt", "w") as f:
            f.write(stdout)

        result.check_returncode()

        plot_profile(stdout)

import os
from typing import Optional, Dict, List

import numpy as np
from networkx import DiGraph
from onnx import ModelProto

from compiler.allocator import LinearAllocator
from compiler.data_type import array_to_bytes, DataType
from compiler.ima_simulate import random_ima_input, random_ima_weight
from compiler.operation import Cycles, Profile, Pointer, MemoryKind, Buffer, Operation, ProfileInfo, CyclesInfo, \
    OperationRecordCycles, OperationCopy, Tensor, OperationCopy2D, OperationCopy3D, Lock
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


def generate_includes(f):
    f.writeln('#include "pmsis.h"')
    f.writeln('#include <bsp/bsp.h>')
    f.writeln('#include "run_layer.h"')
    f.writeln('#include "util.h"')
    f.writeln('#include "generated.h"')


def name_to_c_upper(name: str):
    return name.replace(".", "_").replace("/", "_").replace("::", "_").replace(":", "_").upper()


EQUATION_MATMUL = 'O[b][k]+=A[b][c]*B[c][k]'
EQUATION_CONV = 'O[b][g][k][oy][ox]+=W[k][c][fy][fx]*I[b][g][c][iy][ix]'


class State:
    def __init__(self, onnx_model: ModelProto, workload: DiGraph, core_count: int, simulate: bool):
        self.onnx_model = onnx_model
        self.workload = workload
        self.simulate = simulate

        self.buffers: Dict[str, Buffer] = {}
        self.core_count = core_count

        self.operations_per_core: List[List[Operation]] = [[] for _ in range(core_count)]
        self.operations_fabric: List[Operation] = []

        self.tmp_size_per_core: List[int] = [0 for _ in range(core_count)]
        self.cn_locks = dict()

        self.next_lock = 0
        self.profile_infos: List[ProfileInfo] = []
        self.cycle_infos: List[CyclesInfo] = []
        self.meta_frozen = False

        # define inputs and constants
        for input in onnx_model.graph.input:
            shape = tuple(d.dim_value for d in list(input.type.tensor_type.shape.dim))

            used_as_conv_input = False
            for cn in self.workload:
                if cn.equation == EQUATION_CONV:
                    if cn.input_names[0] == input.name:
                        used_as_conv_input = True
                        break

            if used_as_conv_input:
                assert len(shape) == 4, f"Conv input must be 4D, got shape {shape}"
                b, c, h, w = shape
                inner_shape = b, h, w, c
                padding = ((0, 0), (1, 1), (1, 1), (0, 0))
            else:
                inner_shape = shape
                padding = None

            buffer = self.define_buffer(
                name=input.name, dtype=DataType.Int8,
                inner_shape=inner_shape, padding=padding,
                const=True
            )

            buffer.input = True
            buffer.inner_simulated = random_ima_input(inner_shape)

        for const in onnx_model.graph.initializer:
            onnx_shape = tuple(const.dims)

            # special case weight ranks
            if len(onnx_shape) == 1:
                # skip biases for now
                continue
            if len(onnx_shape) == 2:
                k, c = onnx_shape
                shape = c, k
            elif len(onnx_shape) == 4:
                k, c, h, w = onnx_shape
                shape = h, w, c, k
            else:
                raise KeyError(f"Unexpected weight rank {len(onnx_shape)}")

            buffer = self.define_buffer(name=const.name, dtype=DataType.Int4, inner_shape=shape, padding=None,
                                        const=True)
            buffer.const = True
            buffer.inner_simulated = random_ima_weight(shape)

    def get_buffer(self, name):
        return self.buffers[name]

    # TODO switch to values instead of buffers
    # def define_value(self, name: str, dtype: DataType, shape: tuple[int], transposed: bool, allow_existing: bool, expected_data: np.array):
    #     pass

    def define_buffer(
            self,
            name,
            dtype: DataType,
            inner_shape: tuple,
            padding: Optional[tuple],
            const: bool,
            allow_existing: bool = False
    ) -> Buffer:
        if name in self.buffers:
            if allow_existing:
                # TODO assert that properties match?
                return self.buffers[name]

            raise KeyError(f"Buffer {name} already defined")

        upper = name_to_c_upper(name)

        print(
            f"Defining buffer {name} {upper} dtype={dtype}, inner_shape={inner_shape}, padding={padding}, const={const}")

        if const:
            pointer_l3 = Pointer(f"(L3_CONST_START + L3_CONST_OFFSET_{upper})", MemoryKind.L3)
        else:
            pointer_l3 = Pointer(f"(L3_DYN_START + L3_DYN_OFFSET_{upper})", MemoryKind.L3)

        if padding is None:
            padding = tuple([(0, 0)] * len(inner_shape))

        buffer = Buffer(
            dtype=dtype,
            inner_shape=inner_shape,
            padding=padding,
            pointer_l3=pointer_l3,
        )
        self.buffers[name] = buffer
        return buffer

    def push_operation(self, core: Optional[int], operation: Operation):
        if core is None:
            self.operations_fabric.append(operation)
        else:
            self.operations_per_core[core].append(operation)

    def push_cycles(self, core: Optional[int], kind: str, name: str):
        if core is None:
            core_name = "fabric"
        else:
            core_name = f"core_{core}"

        info = CyclesInfo(core=core_name, kind=kind, name=name)
        cycles = self.new_cycles(info)
        self.push_operation(core, OperationRecordCycles(cycles))

    def push_copy(self, core: int, dest, src, size_bytes):
        op = OperationCopy(dest=dest, src=src, size_bytes=size_bytes)
        self.push_operation(core, op)

    # tensor is only the upper tensor, the lower tensor has the same shape but has simple strides and no offset
    def push_copy_tensor(self, core: Optional[int], upper: Pointer, lower: Pointer, tensor: Tensor, down: bool):
        tensor = tensor.simplify_for_copy()
        upper_offset = upper.offset(tensor.offset_bytes)

        # TODO we could reorder the dims so that the last dim is contiguous
        #   to allow some more tensor types to be copied

        if tensor.rank == 1:
            assert tensor.strides_elem[-1] == 1

            if down:
                dest, src = lower, upper_offset
            else:
                dest, src = upper_offset, lower

            self.push_operation(core, OperationCopy(dest, src, tensor.size_bytes))
        elif tensor.rank == 2:
            assert tensor.strides_elem[-1] == 1

            self.push_operation(core, OperationCopy2D(
                upper_offset,
                lower,
                down,
                tensor.size_bytes,
                tensor.stride_bytes(0),
                tensor.shape_bytes(1),
            ))
        elif tensor.rank == 3:
            assert tensor.strides_elem[2] == 1

            self.push_operation(core, OperationCopy3D(
                upper_offset,
                lower,
                down,
                size_0=tensor.shape_bytes(0),
                size_1=tensor.shape_bytes(1),
                size_2=tensor.shape_bytes(2),
                stride_0=tensor.stride_bytes(0),
                stride_1=tensor.stride_bytes(1),
            ))

        else:
            raise NotImplementedError(f"Copy for tensor rank {tensor.rank}")

    # TODO reduce code duplication here
    def new_lock(self) -> Lock:
        assert not self.meta_frozen
        index = self.next_lock
        self.next_lock += 1
        return Lock(index)

    def get_cn_lock(self, cn: ComputationNode) -> Lock:
        if cn not in self.cn_locks:
            self.cn_locks[cn] = self.new_lock()
        return self.cn_locks[cn]

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


def generate_meta(f, state: State):
    # TODO remove this ram store, it's only used on the fabric controller which gets it anyway
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
        # clear ram to set padding and to make debugging easier
        f.writeln("pi_ram_clear(ram_param, L3_DYN_START, L3_DYN_SIZE);")
        f.writeln()

        # TODO re-enable once we can get this synchronized
        # OperationRecordCycles(state.cycles_end_init).generate_code(f, state)

        f.writeln("return 0;")
    f.writeln("}")
    f.writeln()

    f.writeln("i32 generated_init_cluster(int cluster_id, struct pi_device *cluster_device) {")
    with f:
        for i in range(state.core_count):
            core_l1_size = state.tmp_size_per_core[i]
            f.writeln(f"if (cluster_id == {i}) {{")
            with f:
                f.writeln(f"L1_START_C{i} = pi_l1_malloc(cluster_device, {core_l1_size});")
                f.writeln(f"if (L1_START_C{i} == NULL) {{")
                with f:
                    f.writeln("return -1;")
                f.writeln("}")
            f.writeln("}")

        f.writeln("return 0;")
    f.writeln("}")


def generate_func_final(f, state: State):
    f.writeln("void generated_final_cluster(int cluster_id, struct pi_device *cluster_device) {")
    with f:
        for i in range(state.core_count):
            core_l1_size = state.tmp_size_per_core[i]
            f.writeln(f"if (cluster_id == {i}) {{")
            with f:
                f.writeln(f"pi_l1_free(cluster_device, L1_START_C{i}, {core_l1_size});")
            f.writeln("}")
    f.writeln("}")
    f.writeln()

    f.writeln("void generated_final_fabric() {")
    with f:
        # verify outputs
        for name, buffer in state.buffers.items():
            if not buffer.const and buffer.pointer_l3_expected is not None:
                f.writeln(
                    f"verify_output(ram, {buffer.pointer_l3_expected}, {buffer.pointer_l3}, {buffer.padded_tensor.size_bytes}, \"{name}\");")

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


def generate_func_fabric(f, state: State):
    f.writeln(f"void generated_fabric(struct pi_device *ram_param) {{")
    with f:
        for op in state.operations_fabric:
            op.generate_code(None, f)
    f.writeln("}")


def generate_func_core(f, state: State, core):
    f.writeln(f"void generated_core_{core}() {{")
    with f:
        for op in state.operations_per_core[core]:
            op.generate_code(core, f)
    f.writeln("}")


def generate_data(f, state: State, d_bin):
    alloc_l3_const = LinearAllocator()
    alloc_l3_dyn = LinearAllocator()

    for name, buffer in state.buffers.items():
        upper = name_to_c_upper(name)

        if buffer.inner_simulated is not None:
            assert buffer.inner_simulated.shape == buffer.inner_tensor.shape

            offset = alloc_l3_const.alloc(buffer.padded_tensor.size_bytes)
            offset_name = f"L3_CONST_OFFSET_{upper}"
            f.writeln(f"#define {offset_name} {offset}")

            pointer = Pointer(f"(L3_CONST_START + {offset_name})", MemoryKind.L3)
            if buffer.const:
                assert buffer.pointer_l3 == pointer
            else:
                buffer.pointer_l3_expected = pointer

            padded_value = np.zeros_like(buffer.inner_simulated, shape=buffer.padded_tensor.shape)
            padded_value[*buffer.slices] = buffer.inner_simulated

            value_bytes = array_to_bytes(padded_value.flatten(), buffer.dtype)
            assert len(value_bytes) == buffer.padded_tensor.size_bytes
            d_bin.write(value_bytes)

        if not buffer.const:
            offset = alloc_l3_dyn.alloc(buffer.padded_tensor.size_bytes)
            offset_name = f"L3_DYN_OFFSET_{upper}"
            # TODO split into offset and start again
            f.writeln(f"#define {offset_name} {offset}")

        print(f"Buffer {upper}")
        print(f"  inner_shape: {buffer.inner_shape}")
        print(f"  padding: {buffer.padding}")
        # if buffer.inner_simulated is not None:
        #     print(f"  values: {buffer.inner_simulated.tolist()}")
        #     print(f"  bytes: [{', '.join(f'{x:#02x}' for x in value_bytes)}]")

    f.writeln()

    for i in range(state.core_count):
        f.writeln(f"static PI_L1 u8 *L1_START_C{i} = NULL;")
    f.writeln()

    for i in range(state.core_count):
        f.writeln(f"static PI_L2 u8 L2_START_C{i}[{state.tmp_size_per_core[i]}];")
    f.writeln()

    f.writeln("static u32 L3_CONST_START = 0;")
    f.writeln("static u32 L3_DYN_START = 0;")
    f.writeln(f"#define L3_CONST_SIZE {len(alloc_l3_const)}")
    f.writeln(f"#define L3_DYN_SIZE {len(alloc_l3_dyn)}")


def generate_core_dispatch(f, cores: int):
    f.writeln("void generated_core(int cluster_id, int core_id) {")
    with f:
        f.writeln("if (core_id != 0) return;")
        f.writeln()

        for core_id in range(cores):
            f.writeln(f"if (cluster_id == {core_id}) return generated_core_{core_id}();")
    f.writeln("}")


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

        generate_func_fabric(f, state)
        f.writeln()

        for core in range(len(state.operations_per_core)):
            generate_func_core(f, state, core)
            f.writeln()

        generate_core_dispatch(f, len(state.operations_per_core))

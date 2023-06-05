import os
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
from networkx import DiGraph
from onnx import ModelProto

from compiler.core_allocation import TensorGroups, CoreAllocations, Group
from compiler.data_type import DataType, array_to_bytes
from compiler.ima_simulate import random_ima_input, random_ima_weight
from compiler.operation import Cycles, Profile, Pointer, MemoryKind, Operation, ProfileInfo, CyclesInfo, \
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

# the order is the dense tensor stride order
# TODO add the other stuff
AXES_CANDIDATES_ORDERED = [
    # conv weights
    ("FY", "FX", "C", "K"),
    # conv IO
    ("B", "G", "IY", "IX", "C"),
    ("B", "G", "OY", "OX", "K"),
    # matmul weight
    ("C", "K"),
]


@dataclass
class TensorPlacement:
    offset_core: int
    tensor: Tensor

    # version of tensor that may contain zero padding and is hopefully simpler to copy
    padded_tensor: Tensor
    padded_loop_ranges: Dict[str, Tuple[int, int]]

    def offset(self, pointer: Pointer) -> Pointer:
        return pointer.offset(self.offset_core).offset(self.tensor.offset_bytes)


def loop_ranges_for_axes(
        axes: Tuple[str, ...],
        loop_ranges: Dict[str, Tuple[int, int]],
) -> List[Tuple[int, int]]:
    assert loop_ranges.keys() == set(axes)

    ranges = []
    for d in axes:
        ranges.append(loop_ranges[d])
    return ranges


def maybe_normalize_conv_axes(
        axes: Dict[str, Tuple[int, int]],
        to: Dict[str, Tuple[int, int]]
) -> Dict[str, Tuple[int, int]]:
    output = ["B", "G", "K", "OY", "OX"]
    input = ["B", "G", "C", "IY", "IX"]

    if axes.keys() == set(output) and to.keys() == set(input):
        # convert output to input
        return {input[output.index(d)]: r for d, r in axes.items()}
    if axes.keys() == set(input) and to.keys() == set(output):
        # convert input to output
        return {output[input.index(d)]: r for d, r in axes.items()}

    return axes


def init_onnx_constants(onnx_model) -> Dict[str, Any]:
    constants = {}

    # define inputs and constants
    for input in onnx_model.graph.input:
        onnx_shape = tuple(d.dim_value for d in list(input.type.tensor_type.shape.dim))

        if len(onnx_shape) == 4:
            # conv input, transpose and add group dim
            b, c, h, w = onnx_shape
            shape = b, 1, h, w, c
        else:
            # matmul input?
            shape = onnx_shape

        constants[input.name] = random_ima_input(shape)

    for const in onnx_model.graph.initializer:
        onnx_shape = tuple(const.dims)

        # special case weight ranks
        if len(onnx_shape) == 1:
            # skip biases for now
            continue
        if len(onnx_shape) == 2:
            # matmul weight
            k, c = onnx_shape
            shape = c, k
        elif len(onnx_shape) == 4:
            # conv weight
            k, c, h, w = onnx_shape
            shape = h, w, c, k
        else:
            raise KeyError(f"Unexpected weight rank {len(onnx_shape)}")

        constants[const.name] = random_ima_weight(shape)

    return constants


# TODO remove all of the buffer stuff
class State:
    def __init__(
            self,
            core_count: int,
            onnx_model: ModelProto, workload: DiGraph,
            groups_per_core: List[TensorGroups], allocations: CoreAllocations,
            simulate: bool,
    ):
        self.onnx_model = onnx_model
        self.workload = workload
        self.simulate = simulate
        self.groups_per_core = groups_per_core
        self.allocations = allocations

        self.core_count = core_count

        self.operations_per_core: List[List[Operation]] = [[] for _ in range(core_count)]
        self.operations_fabric: List[Operation] = []

        self.tmp_size_per_core: List[int] = [0 for _ in range(core_count)]
        self.cn_locks = dict()

        self.next_lock = 0
        self.profile_infos: List[ProfileInfo] = []
        self.cycle_infos: List[CyclesInfo] = []
        self.meta_frozen = False

        self.l3_base = Pointer("L3_BASE", MemoryKind.L3)
        self.l2_base_core = [Pointer(f"L2_BASE_C{i}", MemoryKind.L2) for i in range(core_count)]
        self.l1_base_core = [Pointer(f"L1_BASE_C{i}", MemoryKind.L1) for i in range(core_count)]

        self.simulated_constants = init_onnx_constants(onnx_model)
        self.simulated_values = {}

    def placement_for_tensor(self, core: int, step: int, tensor) -> TensorPlacement:
        group = self.groups_per_core[core].get_group(tensor)
        loop_ranges = {d: r for d, r in zip(tensor.loop_dimensions, tensor.loop_ranges)}
        return self.placement_for_group_range(core, step, group, loop_ranges)

    def placement_for_group_range(
            self,
            core: int, step: int, group: Group,
            loop_ranges: Dict[str, Tuple[int, int]],
    ) -> TensorPlacement:
        token = self.allocations.get_token_for_group(core, group.index, step)

        if group.elem_size_bits == 4:
            dtype = DataType.Int4
        elif group.elem_size_bits == 8:
            dtype = DataType.Int8
        else:
            raise ValueError(f"Unknown group elem size {group.elem_size_bits}")

        loop_ranges = maybe_normalize_conv_axes(loop_ranges, group.loop_ranges)

        for axes in AXES_CANDIDATES_ORDERED:
            if set(axes) == loop_ranges.keys():
                break
        else:
            raise ValueError(f"Unknown loop dimensions {loop_ranges}")

        assert loop_ranges.keys() == group.loop_ranges.keys(), \
            f"Loop dim mismatch between tensor and group: {loop_ranges} vs {group.loop_ranges}"

        group_ranges = loop_ranges_for_axes(axes, group.loop_ranges)
        piece_ranges = loop_ranges_for_axes(axes, loop_ranges)

        padded_ranges = piece_ranges

        if len(axes) == 5:
            range_b, range_g, range_iy, range_ix, range_c = piece_ranges

            padded_ranges = (
                range_b,
                range_g,
                (range_iy[0] - 1, range_iy[1] + 1),
                (range_ix[0] - 1, range_ix[1] + 1),
                range_c
            )

            for (pad_start, pad_end), (group_start, group_end) in zip(padded_ranges, group_ranges):
                if pad_start < group_start or pad_end > group_end:
                    # fail
                    padded_ranges = piece_ranges
                    break

        padded_loop_ranges = dict(zip(axes, padded_ranges))

        group_shape = [
            group_end - group_start
            for group_start, group_end in group_ranges
        ]
        group_tensor = Tensor.simple(dtype, tuple(group_shape))

        piece_slices = [
            slice(piece_start - group_start, piece_end - group_start)
            for (group_start, _), (piece_start, piece_end) in zip(group_ranges, piece_ranges)
        ]
        piece_tensor = group_tensor[*piece_slices]

        padded_slices = [
            slice(piece_start - group_start, piece_end - group_start)
            for (group_start, _), (piece_start, piece_end) in zip(group_ranges, padded_ranges)
        ]
        padded_tensor = group_tensor[*padded_slices]

        return TensorPlacement(
            offset_core=token.offset,
            tensor=piece_tensor,
            padded_tensor=padded_tensor,
            padded_loop_ranges=padded_loop_ranges,
        )

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

    def should_simulate(self, value_name: str) -> bool:
        return self.simulate and value_name not in self.simulated_values

    def get_simulation_value(self, name: str):
        assert not (name in self.simulated_constants and name in self.simulated_values)

        if name in self.simulated_constants:
            return self.simulated_constants[name]

        if name in self.simulated_values:
            return self.simulated_values[name]

        return None

    def set_simulation_value(self, name: str, value):
        assert name not in self.simulated_constants
        assert name not in self.simulated_values
        self.simulated_values[name] = value

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
    f.writeln("i32 generated_init_fabric(struct pi_device *ram_param, u32 l3_file_start, u32 l3_file_size) {")

    # TODO initialize ram with inputs

    with f:
        # TODO re-enable once we can get this synchronized
        # OperationRecordCycles(state.cycles_start_init).generate_code(f, state)

        f.writeln("ram = ram_param;")
        f.writeln()

        f.writeln("if (l3_file_size != L3_SIZE*2) {")
        with f:
            f.writeln("printf(\"ERROR: L3 file size mismatch\\n\");")
            f.writeln("return -1;")
        f.writeln("}")
        f.writeln()

        f.writeln("L3_BASE_INIT = l3_file_start;")
        f.writeln("L3_BASE_FINAL = l3_file_start + L3_SIZE;")
        f.writeln()
        f.writeln("if (pi_ram_alloc(ram_param, &L3_BASE, L3_SIZE)) {")
        with f:
            f.writeln("printf(\"ERROR: Failed to allocate L3_DYN\\n\");")
            f.writeln("return -2;")
        f.writeln("}")
        f.writeln()
        # copy the initial contents over
        f.writeln("pi_ram_copy_self(ram_param, L3_BASE, L3_BASE_INIT, L3_SIZE);")
        f.writeln()

        # TODO re-enable once we can get this synchronized
        # OperationRecordCycles(state.cycles_end_init).generate_code(f, state)

        f.writeln("return 0;")
    f.writeln("}")
    f.writeln()

    f.writeln("i32 generated_init_cluster(int cluster_id, struct pi_device *cluster_device) {")
    with f:
        for i in range(state.core_count):
            core_l1_size = state.allocations.core_allocators[i].allocated_size_used
            f.writeln(f"if (cluster_id == {i}) {{")
            with f:
                f.writeln(f"L1_BASE_C{i} = pi_l1_malloc(cluster_device, {core_l1_size});")
                f.writeln(f"if (L1_BASE_C{i} == NULL) {{")
                with f:
                    f.writeln("return -1;")
                f.writeln("}")
            f.writeln("}")

        f.writeln("return 0;")
    f.writeln("}")


def generate_func_final(f, state: State, outputs_to_check):
    f.writeln("void generated_final_cluster(int cluster_id, struct pi_device *cluster_device) {")
    with f:
        for i in range(state.core_count):
            core_l1_size = state.tmp_size_per_core[i]
            f.writeln(f"if (cluster_id == {i}) {{")
            with f:
                f.writeln(f"pi_l1_free(cluster_device, L1_BASE_C{i}, {core_l1_size});")
            f.writeln("}")
    f.writeln("}")
    f.writeln()

    f.writeln("void generated_final_fabric() {")
    with f:
        for (name, offset, size) in outputs_to_check:
            print(f"Verifying {name}: offset {offset} size {size}")
            f.writeln(
                f"verify_output(ram, L3_BASE_FINAL + {offset}, L3_BASE + {offset}, {size}, \"{name}\");")

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


def simulate_write_value(target, placement: TensorPlacement, value) -> (int, int):
    assert placement.tensor.rank == len(value.shape)
    assert all(t >= v for t, v in zip(placement.tensor.shape, value.shape))
    assert placement.tensor.has_simple_strides

    padded_shape = placement.tensor.shape
    padded_value = np.zeros(padded_shape, dtype=np.uint8)

    slices = []
    for t, v in zip(placement.tensor.shape, value.shape):
        assert t >= v
        delta = t - v
        assert delta % 2 == 0
        padding = delta // 2
        slices.append(slice(padding, padding + v))

    padded_value[*slices] = value

    offset = placement.offset_core + placement.tensor.offset_bytes
    size = placement.tensor.size_bytes
    target[offset:offset + size] = list(array_to_bytes(padded_value.flatten(), placement.tensor.dtype))


def determine_l3_data(state: State):
    offchip_core = state.core_count
    l3_allocator = state.allocations.core_allocators[offchip_core]
    l3_size = l3_allocator.allocated_size_used

    l3_groups = state.groups_per_core[offchip_core]

    l3_init = np.zeros(l3_size, dtype=np.uint8)

    for group in l3_groups.groups:
        if group.value_name in state.simulated_constants:
            value = state.simulated_constants[group.value_name]
            placement = state.placement_for_group_range(offchip_core, -np.inf, group, group.loop_ranges)
            print(f"Initially placing constant with shape {value.shape} in {placement}")

            simulate_write_value(l3_init, placement, value)

    l3_final = np.zeros(l3_size, dtype=np.uint8)

    output_names = [n.name for n in state.onnx_model.graph.output]
    outputs_to_check = []

    for group in l3_groups.groups:
        if group.value_name in output_names and group.value_name in state.simulated_values:
            value = state.simulated_values[group.value_name]
            placement = state.placement_for_group_range(offchip_core, np.inf, group, group.loop_ranges)

            print(f"Finally placing value with shape {value.shape} in {placement}")

            simulate_write_value(l3_final, placement, value)
            outputs_to_check.append((group.value_name, placement.offset_core, placement.tensor.size_bytes))

    return l3_size, l3_init, l3_final, outputs_to_check


def generate_data(f, state: State, d_bin):
    # TODO is this correct? is it necessary to store L1 in L1 and L2 in L2?
    for i in range(state.core_count):
        f.writeln(f"static PI_L1 u8 *L1_BASE_C{i} = NULL;")
    f.writeln()

    for i in range(state.core_count):
        core_l2_size = state.allocations.core_allocators[i].allocated_size_used
        f.writeln(f"static PI_L2 u8 L2_BASE_C{i}[{core_l2_size}];")
    f.writeln()

    l3_size, data_init, data_final, outputs_to_check = determine_l3_data(state)
    assert len(data_init) == l3_size
    assert len(data_final) == l3_size

    d_bin.write(data_init)
    d_bin.write(data_final)

    f.writeln("static u32 L3_BASE = 0;")
    f.writeln("static u32 L3_BASE_INIT = 0;")
    f.writeln("static u32 L3_BASE_FINAL = 0;")
    f.writeln(f"#define L3_SIZE {l3_size}")

    return outputs_to_check


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
        outputs_to_check = generate_data(f, state, file_data_bin)
        f.writeln()

        generate_func_init(f, state)
        f.writeln()
        generate_func_final(f, state, outputs_to_check)
        f.writeln()

        generate_func_fabric(f, state)
        f.writeln()

        for core in range(len(state.operations_per_core)):
            generate_func_core(f, state, core)
            f.writeln()

        generate_core_dispatch(f, len(state.operations_per_core))

import os
import subprocess
from typing import Optional, Dict, List

import numpy as np
import onnx
from matplotlib import pyplot as plt
from networkx import DiGraph
from onnx import ModelProto
from zigzag.classes.cost_model.cost_model import CostModelEvaluation

from compiler.allocator import LinearAllocator
from compiler.data_type import DataType, array_to_bytes
from compiler.ima_simulate import random_ima_input, random_ima_weight, ima_matmul, ima_conv
from compiler.operation import Operation, MemoryKind, Pointer, Lock, Profile, Cycles, OperationCopy, OperationMatmul, \
    Buffer, OperationPad, CyclesInfo, ProfileInfo, OperationRecordCycles, OperationLockIncrement, OperationLockWait, \
    OperationCopy2D, Tensor, OperationComment, OperationConvPadded, OperationCopy3D
from compiler.plot_profile import plot_profile
from stream.classes.cost_model.cost_model import StreamCostModelEvaluation
from stream.classes.cost_model.record import Step, StepAddTensorToCore, StepRemoveTensorFromCore, StepRunNode
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


def generate_includes(f):
    f.writeln('#include "pmsis.h"')
    f.writeln('#include <bsp/bsp.h>')
    f.writeln('#include "run_layer.h"')
    f.writeln('#include "util.h"')
    f.writeln('#include "generated.h"')


def name_to_c_upper(name: str):
    return name.replace(".", "_").replace("/", "_").replace("::", "_").replace(":", "_").upper()


def visit_matmul(core: int, workload, cn: ComputationNode, orig_cn: ComputationNode, state: State):
    # get inputs
    # TODO use bias
    input_name, weight_name, _ = cn.input_names
    output_name, = cn.output_names

    input = state.get_buffer(input_name)
    weight = state.get_buffer(weight_name)

    # figure out ranges
    assert orig_cn.loop_ranges.keys() == {"B", "K", "C"}, orig_cn.loop_ranges.keys()
    assert cn.loop_ranges.keys() == {"B", "K", "C"}, cn.loop_ranges.keys()
    (b_start, b_end) = cn.loop_ranges["B"]
    (k_start, k_end) = cn.loop_ranges["K"]
    (c_start, c_end) = cn.loop_ranges["C"]
    (b_zero, b_full) = orig_cn.loop_ranges["B"]
    (k_zero, k_full) = orig_cn.loop_ranges["K"]
    (c_zero, c_full) = orig_cn.loop_ranges["C"]
    assert b_zero == 0 and k_zero == 0 and c_zero == 0

    state.push_operation(core, OperationComment(
        f"matmul b={b_start}..{b_end}, k={k_start}..{k_end}, c={c_start}..{c_end}"))

    # get full output buffer
    output = state.define_buffer(
        name=output_name, dtype=DataType.Int8,
        inner_shape=(b_full, k_full), padding=None,
        const=False,
        allow_existing=True
    )

    full_input = input.inner_tensor
    full_weight = weight.inner_tensor
    full_output = output.inner_tensor

    piece_input = full_input[b_start:b_end, c_start:c_end]
    piece_weight = full_weight[c_start:c_end, k_start:k_end]  # weight is stored transposed
    piece_output = full_output[b_start:b_end, k_start:k_end]

    # allocate temporary buffers
    # TODO we can just reuse the same buffer, or at least we can reuse in/weight for output
    tmp_alloc = LinearAllocator()
    tmp_input = tmp_alloc.alloc(piece_input.size_bytes)
    tmp_weight = tmp_alloc.alloc(piece_weight.size_bytes)
    tmp_output = tmp_alloc.alloc(piece_output.size_bytes)

    # allocate space for temporary buffers in L1 and L2
    state.tmp_size_per_core[core] = max(len(tmp_alloc), state.tmp_size_per_core[core])

    # wait for dependencies (this automatically handles the split input case)
    state.push_cycles(core, "start", "wait")
    for (prev, _) in workload.in_edges(cn):
        state.push_operation(core, OperationLockWait(state.get_cn_lock(prev), 1))
    state.push_cycles(core, "end", "wait")

    start_l1 = Pointer(f"L1_START_C{core}", MemoryKind.L1)
    start_l2 = Pointer(f"L2_START_C{core}", MemoryKind.L2)

    # fabric operations
    fabric_in_start = state.new_lock()
    fabric_in_done = state.new_lock()
    fabric_out_start = state.new_lock()
    fabric_out_done = state.new_lock()

    state.push_operation(None, OperationLockWait(fabric_in_start, 1))
    state.push_cycles(None, "start", "down")
    state.push_copy_tensor(None, input.pointer_l3, start_l2.offset(tmp_input), piece_input, True)
    state.push_copy_tensor(None, weight.pointer_l3, start_l2.offset(tmp_weight), piece_weight, True)
    state.push_operation(None, OperationLockIncrement(fabric_in_done))
    state.push_cycles(None, "end", "down")
    state.push_operation(None, OperationPad())

    state.push_operation(None, OperationLockWait(fabric_out_start, 1))
    state.push_cycles(None, "start", "up")
    state.push_copy_tensor(None, output.pointer_l3, start_l2.offset(tmp_output), piece_output, False)
    state.push_operation(None, OperationLockIncrement(fabric_out_done))
    state.push_cycles(None, "end", "up")
    state.push_operation(None, OperationPad())

    # copy inputs
    state.push_cycles(core, "start", "input")
    state.push_operation(core, OperationLockIncrement(fabric_in_start))
    state.push_operation(core, OperationLockWait(fabric_in_done, 1))
    state.push_copy(core, start_l1.offset(tmp_input), start_l2.offset(tmp_input), piece_input.size_bytes)
    state.push_copy(core, start_l1.offset(tmp_weight), start_l2.offset(tmp_weight), piece_weight.size_bytes)
    state.push_cycles(core, "end", "input")

    # real operation
    state.push_cycles(core, "start", "calc")
    state.push_operation(core, OperationMatmul(
        b=(b_end - b_start), k=k_end - k_start, c=c_end - c_start,
        weight=start_l1.offset(tmp_weight), input=start_l1.offset(tmp_input), output=start_l1.offset(tmp_output),
        profile=state.new_profile(ProfileInfo(core=f"ima_core_{core}", name="matmul")),
    ))
    state.push_cycles(core, "end", "calc")

    # copy output
    state.push_cycles(core, "start", "output")
    state.push_copy(core, start_l2.offset(tmp_output), start_l1.offset(tmp_output), piece_output.size_bytes)
    state.push_operation(core, OperationLockIncrement(fabric_out_start))
    state.push_operation(core, OperationLockWait(fabric_out_done, 1))
    state.push_operation(core, OperationLockIncrement(state.get_cn_lock(cn)))
    state.push_cycles(core, "end", "output")

    state.push_operation(core, OperationPad())

    # simulate if required and not already simulated
    if state.simulate and output.inner_simulated is None:
        # (c, k) -> (k, c)
        weight_trans = weight.inner_simulated.transpose([1, 0])
        output.inner_simulated = ima_matmul(input.inner_simulated, weight_trans)


# TODO unify with matmul codegen
def visit_conv(core: int, workload, cn: ComputationNode, orig_cn: ComputationNode, state: State):
    # get inputs
    # TODO include bias
    input_name, weight_name, _ = cn.input_names
    output_name, = cn.output_names

    input = state.get_buffer(input_name)
    weight = state.get_buffer(weight_name)

    # figure out ranges
    for node in [orig_cn, cn]:
        expected_keys = {'B', 'K', 'G', 'OX', 'OY', 'C', 'FX', 'FY', 'IX', 'IY'}
        assert node.loop_ranges.keys() == expected_keys, orig_cn.loop_ranges.keys()
        assert cn.loop_ranges['G'] == (0, 1)
        assert cn.loop_ranges['FX'] == (0, 3)
        assert cn.loop_ranges['FY'] == (0, 3)

    (b_start, b_end) = cn.loop_ranges["B"]
    (k_start, k_end) = cn.loop_ranges["K"]
    (c_start, c_end) = cn.loop_ranges["C"]
    (oy_start, oy_end) = cn.loop_ranges["OY"]
    (ox_start, ox_end) = cn.loop_ranges["OX"]
    (iy_start, iy_end) = cn.loop_ranges["IY"]
    (ix_start, ix_end) = cn.loop_ranges["IX"]

    (b_zero, b_full) = orig_cn.loop_ranges["B"]
    (k_zero, k_full) = orig_cn.loop_ranges["K"]
    (c_zero, c_full) = orig_cn.loop_ranges["C"]
    (oy_zero, oy_full) = orig_cn.loop_ranges["OY"]
    (ox_zero, ox_full) = orig_cn.loop_ranges["OX"]
    (iy_zero, iy_full) = orig_cn.loop_ranges["IY"]
    (ix_zero, ix_full) = orig_cn.loop_ranges["IX"]

    assert all(v == 0 for v in [b_zero, k_zero, c_zero, ox_zero, oy_zero])
    assert iy_zero == -1 and ix_zero == -1
    assert iy_start == oy_start - 1
    assert ix_start == ox_start - 1
    assert iy_full == oy_full + 1
    assert ix_full == ox_full + 1
    assert iy_start == oy_start - 1
    assert iy_end == oy_end + 1
    assert ix_start == ox_start - 1
    assert ix_end == ox_end + 1

    comment = f"conv b={b_start}..{b_end}, k={k_start}..{k_end}, c={c_start}..{c_end} ox={ox_start}..{ox_end} oy={oy_start}..{oy_end}"
    state.push_operation(core, OperationComment(comment))

    # get full output buffer
    output = state.define_buffer(
        name=output_name, dtype=DataType.Int8,
        inner_shape=(b_full, oy_full, ox_full, k_full),
        padding=((0, 0), (1, 1), (1, 1), (0, 0)),
        const=False,
        allow_existing=True
    )

    full_input = input.padded_tensor
    full_weight = weight.padded_tensor
    full_output = output.inner_tensor

    # everything is transposed, careful
    piece_input = full_input[b_start:b_end, 1 + iy_start:1 + iy_end, 1 + ix_start:1 + ix_end, c_start:c_end]
    piece_weight = full_weight[:, :, c_start:c_end, k_start:k_end]
    piece_output = full_output[b_start:b_end, oy_start:oy_end, ox_start:ox_end, k_start:k_end]

    # allocate temporary buffers
    tmp_alloc = LinearAllocator()
    tmp_input = tmp_alloc.alloc(piece_input.size_bytes)
    tmp_weight = tmp_alloc.alloc(piece_weight.size_bytes)
    tmp_output = tmp_alloc.alloc(piece_output.size_bytes)

    # allocate space for temporary buffers in L1 and L2
    state.tmp_size_per_core[core] = max(len(tmp_alloc), state.tmp_size_per_core[core])

    # wait for dependencies (this automatically handles the split input case)
    state.push_cycles(core, "start", "wait")
    for (prev, _) in workload.in_edges(cn):
        state.push_operation(core, OperationLockWait(state.get_cn_lock(prev), 1))
    state.push_cycles(core, "end", "wait")

    start_l1 = Pointer(f"L1_START_C{core}", MemoryKind.L1)
    start_l2 = Pointer(f"L2_START_C{core}", MemoryKind.L2)

    # fabric operations
    fabric_in_start = state.new_lock()
    fabric_in_done = state.new_lock()
    fabric_out_start = state.new_lock()
    fabric_out_done = state.new_lock()

    state.push_operation(None, OperationLockWait(fabric_in_start, 1))
    state.push_cycles(None, "start", "down")
    state.push_copy_tensor(None, input.pointer_l3, start_l2.offset(tmp_input), piece_input, True)
    state.push_copy_tensor(None, weight.pointer_l3, start_l2.offset(tmp_weight), piece_weight, True)
    state.push_operation(None, OperationLockIncrement(fabric_in_done))
    state.push_cycles(None, "end", "down")
    state.push_operation(None, OperationPad())

    state.push_operation(None, OperationLockWait(fabric_out_start, 1))
    state.push_cycles(None, "start", "up")
    state.push_copy_tensor(None, output.pointer_l3, start_l2.offset(tmp_output), piece_output, False)
    state.push_operation(None, OperationLockIncrement(fabric_out_done))
    state.push_cycles(None, "end", "up")
    state.push_operation(None, OperationPad())

    # copy inputs
    state.push_cycles(core, "start", "input")
    state.push_operation(core, OperationLockIncrement(fabric_in_start))
    state.push_operation(core, OperationLockWait(fabric_in_done, 1))
    state.push_copy(core, start_l1.offset(tmp_input), start_l2.offset(tmp_input), piece_input.size_bytes)
    state.push_copy(core, start_l1.offset(tmp_weight), start_l2.offset(tmp_weight), piece_weight.size_bytes)
    state.push_cycles(core, "end", "input")

    # real operation
    state.push_cycles(core, "start", "calc")
    state.push_operation(core, OperationConvPadded(
        b=(b_end - b_start), k=(k_end - k_start), c=(c_end - c_start),
        oh=(oy_end - oy_start), ow=(ox_end - ox_start),
        fh=3, fw=3,
        weight=start_l1.offset(tmp_weight), input=start_l1.offset(tmp_input), output=start_l1.offset(tmp_output),
        profile=state.new_profile(ProfileInfo(core=f"ima_core_{core}", name="conv")),
    ))
    state.push_cycles(core, "end", "calc")

    # copy output
    state.push_cycles(core, "start", "output")
    state.push_copy(core, start_l2.offset(tmp_output), start_l1.offset(tmp_output), piece_output.size_bytes)
    state.push_operation(core, OperationLockIncrement(fabric_out_start))
    state.push_operation(core, OperationLockWait(fabric_out_done, 1))
    state.push_operation(core, OperationLockIncrement(state.get_cn_lock(cn)))
    state.push_cycles(core, "end", "output")

    state.push_operation(core, OperationPad())

    if state.simulate and output.inner_simulated is None:
        # (h w c k) -> (k c h w)
        weight_trans = weight.inner_simulated.transpose([3, 2, 0, 1])
        # (b h w c) -> (b c h w)
        input_trans = input.inner_simulated.transpose([0, 3, 1, 2])

        output_trans = ima_conv(input_trans, weight_trans)

        # (b k h w) -> (b h w k)
        output.inner_simulated = output_trans.transpose([0, 2, 3, 1])


def visit_node(state: State, workload, cn: ComputationNode, zcme: CostModelEvaluation):
    core = cn.get_core_allocation()
    orig_cn = cn.original_node if cn.original_node is not None else cn

    if cn.equation == EQUATION_MATMUL:
        visit_matmul(core, workload, cn, orig_cn, state)
    elif cn.equation == EQUATION_CONV:
        visit_conv(core, workload, cn, orig_cn, state)
    else:
        raise ValueError(f"Unrecognised equation {cn.equation}")


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


# TODO ensure fabric operations are in a good order (topologically sorted?)
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


class TensorMerger:
    def __init__(self):
        self.groups = []
        self.tensor_to_group = {}

    def get_group(self, tensor) -> int:
        hash = tensor.equality_hash()
        if hash in self.tensor_to_group:
            return self.tensor_to_group[hash]

        group = len(self.groups)
        self.groups.append({hash})
        self.tensor_to_group[hash] = group
        return group

    def merge_groups(self, x_group: int, y_group: int):
        if x_group == y_group:
            return

        self.groups[x_group] |= self.groups[y_group]
        self.groups[y_group] = None

        for k in self.tensor_to_group:
            if self.tensor_to_group[k] == y_group:
                self.tensor_to_group[k] = x_group

        return x_group

    def merge_matching_tensors(self, inputs):
        for i, x in enumerate(inputs):
            for y in inputs[i + 1:]:
                if x.origin == y.origin and x.layer_operand == y.layer_operand:
                    x_group = self.get_group(x)
                    y_group = self.get_group(y)
                    self.merge_groups(x_group, y_group)

    def final_groups(self):
        return [g for g in self.groups if g is not None]


def tensor_core_allocations(steps: List[Step]):
    core_tensor_lifetime = {}
    hash_to_tensor = {}
    max_lifetime = 0

    merger = TensorMerger()

    # TODO handle the case where a tensor is evicted and then later loaded back onto the same core
    for step in steps:
        print(step)
        max_lifetime = max(max_lifetime, step.time_end)

        if isinstance(step, StepAddTensorToCore):
            key = (step.core, step.tensor.equality_hash())
            hash_to_tensor.setdefault(step.tensor.equality_hash(), step.tensor)

            assert key not in core_tensor_lifetime
            core_tensor_lifetime[key] = [step.time_start, None, None]

        elif isinstance(step, StepRemoveTensorFromCore):
            key = (step.core, step.tensor.equality_hash())
            hash_to_tensor.setdefault(step.tensor.equality_hash(), step.tensor)

            if key in core_tensor_lifetime:
                assert core_tensor_lifetime[key][2] is None
                core_tensor_lifetime[key][2] = step.time_start
            else:
                print(f"  Warning: {step} has no matching add step")

        elif isinstance(step, StepRunNode):
            for x in step.inputs:
                print(f"  {x}")

            for x in step.inputs:
                key = (step.core, x.equality_hash())
                core_tensor_lifetime[key][1] = step.time_end

            merger.merge_matching_tensors(step.inputs)
        else:
            assert False, f"Unknown step type {step}"

        print()

    print("grouped tensors: ")
    for group_hashes in merger.final_groups():
        group = [hash_to_tensor[h] for h in group_hashes]
        print(f"  : {group}")

    fig, axes = plt.subplots(
        nrows=len(core_tensor_lifetime),
        sharex="all", squeeze=False, figsize=(32, 32)
    )
    axes = axes.squeeze(1)

    for i, ((core, tensor_hash), [start, last_used, end]) in enumerate(core_tensor_lifetime.items()):
        tensor = hash_to_tensor[tensor_hash]
        print(f"Slice {core} {tensor} {start}..{end} last={last_used}")

        ax = axes[i]
        ax.set_ylabel(f"{core.id}, {tensor.id}", rotation='horizontal', ha='right')

        if end is None:
            end = max_lifetime

        if start is not None and end is not None:
            if last_used is None:
                ax.axvspan(start, end, facecolor="b", alpha=1.0)
            else:
                ax.axvspan(start, last_used, facecolor="g", alpha=1.0)
                ax.axvspan(last_used, end, facecolor="r", alpha=1.0)
        else:
            print(f"Warning: {core} {tensor} has invalid lifetime {start}..{end}")

    # fig.tight_layout()
    plt.savefig("outputs/tensor_core_life.png")
    plt.show(block=False)


def compile_and_run(
        onnx_path, scme: StreamCostModelEvaluation, node_hw_performances,
        pulp_sdk_path, project_path,
        simulate: bool, run: bool, plot: bool
):
    print("Collecting workload")

    tensor_core_allocations(scme.recording.steps)
    return

    assert onnx_path.endswith(".onnx")
    onnx_model = onnx.load(onnx_path, load_external_data=False)
    # noinspection PyUnresolvedReferences
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    workload: DiGraph = scme.workload
    accelerator: Accelerator = scme.accelerator
    cluster_cores = len(accelerator.cores) - 1

    np.random.seed(0)
    state = State(onnx_model, workload, cluster_cores, simulate=simulate)

    nodes = sorted(workload, key=lambda node: node.start)

    for cn in nodes:
        cn: ComputationNode
        print(cn)

        core = cn.core_allocation

        key_node = next(other for other in node_hw_performances.keys() if other.id[0] == cn.id[0])
        zcme = next(m for c, m in node_hw_performances[key_node].items() if c.id == core)

        print(f"Visiting {cn}")
        print(f"  incoming edge nodes: {list(a for a, _ in workload.in_edges(cn))}")
        print(f"  outgoing edge nodes: {list(b for _, b in workload.out_edges(cn))}")
        print(f"  inputs: {cn.input_names}")
        print(f"  outputs: {cn.output_names}")
        print(f"  loop_ranges: {cn.loop_ranges}")
        print(f"  temporal: {zcme.temporal_mapping}")
        print(f"  spatial: {zcme.spatial_mapping}")

        visit_node(state, workload, cn, zcme)

    print("Allocated buffers:")
    for name, buffer in state.buffers.items():
        print(f"  {name}: {buffer.padded_tensor} {buffer.inner_tensor}")

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
            f"./safe_make all run CORES={8} CLUSTERS={cluster_cores}",
        ]
        result = subprocess.run(["wsl", *"; ".join(commands).split(" ")], stdout=subprocess.PIPE)
        stdout = result.stdout.decode("utf-8")

        print(stdout)
        with open("outputs/stdout.txt", "w") as f:
            f.write(stdout)

        result.check_returncode()

        if plot:
            plot_profile(stdout, "outputs/profile.png", block=False)

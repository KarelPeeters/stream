import random
import subprocess
from dataclasses import dataclass
from math import prod
from typing import List, Dict, Tuple, Any, Set

import numpy as np
import onnx
from matplotlib import pyplot as plt
from networkx import DiGraph
from zigzag.classes.cost_model.cost_model import CostModelEvaluation

from compiler.allocator import LinearAllocator, TimeAllocator, Token
from compiler.codegen import generate_code, State, EQUATION_MATMUL, EQUATION_CONV
from compiler.data_type import DataType
from compiler.ima_simulate import ima_matmul, ima_conv
from compiler.operation import MemoryKind, Pointer, OperationMatmul, \
    OperationPad, ProfileInfo, OperationLockIncrement, OperationLockWait, \
    OperationComment, OperationConvPadded
from compiler.plot_profile import plot_profile
from stream.classes.cost_model.cost_model import StreamCostModelEvaluation
from stream.classes.cost_model.record import Step, StepAddTensorToCore, StepRemoveTensorFromCore, StepRunNode
from stream.classes.hardware.architecture.accelerator import Accelerator
from stream.classes.workload.computation_node import ComputationNode


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


@dataclass
class Group:
    index: int
    keys: Set[Any]
    elem_size_bits: int
    loop_dimensions: Tuple[str]
    loop_ranges: Tuple[Tuple[int, int]]


@dataclass
class TensorGroups:
    key_to_group: Dict[Any, int]
    key_to_tensor: Dict[Any, Any]
    groups: List[Group]

    def get_group(self, tensor) -> Group:
        key = tensor.equality_key()
        group = self.key_to_group[key]
        return self.groups[group]


class TensorGrouper:
    def __init__(self):
        self.groups = []
        self.key_to_group = {}
        self.key_to_tensor = {}

    def get_group(self, tensor, allow_new: bool) -> int:
        key = tensor.equality_key()
        self.key_to_tensor.setdefault(key, tensor)

        if key in self.key_to_group:
            return self.key_to_group[key]
        else:
            assert allow_new, f"New group not allowed, tensor {tensor}"

        group = len(self.groups)
        self.groups.append({key})
        self.key_to_group[key] = group
        return group

    def merge_groups(self, x_group: int, y_group: int):
        if x_group == y_group:
            return

        self.groups[x_group] |= self.groups[y_group]
        self.groups[y_group] = None

        for k in self.key_to_group:
            if self.key_to_group[k] == y_group:
                self.key_to_group[k] = x_group

        return x_group

    def merge_matching_tensors(self, inputs):
        for i, x in enumerate(inputs):
            for y in inputs[i + 1:]:
                if x.origin == y.origin and x.layer_operand == y.layer_operand:
                    x_group = self.get_group(x, allow_new=False)
                    y_group = self.get_group(y, allow_new=False)
                    self.merge_groups(x_group, y_group)

    def finish(self) -> TensorGroups:
        groups = []
        group_map = []

        for group in self.groups:
            if group is None:
                group_map.append(None)
                continue

            index = len(groups)
            group_map.append(index)

            loop_dimensions = None
            loop_ranges = None
            elem_size_bits = None

            for key in group:
                tensor = self.key_to_tensor[key]

                if loop_dimensions is None:
                    loop_dimensions = tensor.loop_dimensions
                    loop_ranges = list(tensor.loop_ranges)
                    elem_size_bits = tensor.size / prod(end - start for start, end in tensor.loop_ranges)
                else:
                    assert loop_dimensions == tensor.loop_dimensions
                    assert elem_size_bits == tensor.size / prod(end - start for start, end in tensor.loop_ranges)

                    for i in range(len(loop_dimensions)):
                        old_min, old_max = loop_ranges[i]
                        new_min, new_max = tensor.loop_ranges[i]
                        loop_ranges[i] = (min(old_min, new_min), max(old_max, new_max))

            assert loop_dimensions is not None

            groups.append(Group(index, group, elem_size_bits, loop_dimensions, tuple(loop_ranges)))

        # map group indices
        key_to_group = {
            k: group_map[v] for k, v in self.key_to_group.items()
        }

        return TensorGroups(
            key_to_group=key_to_group,
            key_to_tensor=self.key_to_tensor,
            groups=groups
        )


def collect_tensor_groups(steps: List[Step]):
    core_tensor_lifetime = {}
    max_lifetime = 0

    merger = TensorGrouper()

    # TODO handle the case where a tensor is evicted and then later loaded back onto the same core
    for step in steps:
        print(step)
        max_lifetime = max(max_lifetime, step.time_end)

        if isinstance(step, StepAddTensorToCore):
            key = (step.core, step.tensor.equality_key())

            assert key not in core_tensor_lifetime
            core_tensor_lifetime[key] = [step.time_start, None, None]

            # ensure a group is created for this tensor
            merger.get_group(step.tensor, allow_new=True)

        elif isinstance(step, StepRemoveTensorFromCore):
            key = (step.core, step.tensor.equality_key())

            if key in core_tensor_lifetime:
                assert core_tensor_lifetime[key][2] is None
                core_tensor_lifetime[key][2] = step.time_start
            else:
                print(f"  Warning: {step} has no matching add step")

            # double check that the group for this tensor already exists
            merger.get_group(step.tensor, allow_new=False)

        elif isinstance(step, StepRunNode):
            for x in step.inputs:
                print(f"  {x}")

            for x in step.inputs:
                key = (step.core, x.equality_key())
                core_tensor_lifetime[key][1] = step.time_end

            merger.merge_matching_tensors(step.inputs)
        else:
            assert False, f"Unknown step type {step}"

        print()

    merged_groups = merger.finish()

    print("grouped tensors: ")
    for group in merged_groups.groups:
        print(f"  : {group}")

    fig, axes = plt.subplots(
        nrows=len(core_tensor_lifetime),
        sharex="all", squeeze=False, figsize=(32, 32)
    )
    axes = axes.squeeze(1)

    for i, ((core, tensor), [start, last_used, end]) in enumerate(core_tensor_lifetime.items()):
        print(f"Slice {core} {tensor} {start}..{end} last={last_used}")

        ax = axes[i]
        ax.set_ylabel(f"{core.id}, {tensor}", rotation='horizontal', ha='right')

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

    return merged_groups


@dataclass
class AllocatedGroup:
    token: Token
    parts_left: int


@dataclass
class CoreAllocations:
    core_allocators: Dict[int, TimeAllocator]
    step_group_to_token: Dict[int, Dict[int, Token]]


def allocate_per_core(groups: TensorGroups, steps: List[Step]) -> CoreAllocations:
    allocators = {}
    curr_core_group_allocated = {}
    step_group_to_token = {}

    def core_alloc(core: int):
        if core not in allocators:
            allocators[core] = TimeAllocator(start_time=0)
        return allocators[core]

    for step_index, step in enumerate(steps):
        if isinstance(step, StepAddTensorToCore):
            # allocate entire group at once
            group = groups.get_group(step.tensor)
            key = (step.core.id, group.index)

            if key in curr_core_group_allocated:
                alloc = curr_core_group_allocated[key]
                alloc.parts_left += 1

            else:
                group_size_bits = group.elem_size_bits * prod(end - start for start, end in group.loop_ranges)
                group_size_bytes = (group_size_bits + 7) // 8

                token = core_alloc(step.core.id).alloc(group_size_bytes, time=step.time_start)
                alloc = AllocatedGroup(token, len(group.keys))

                curr_core_group_allocated[key] = alloc

            step_group_to_token.setdefault(step_index, {})[group.index] = alloc.token

        elif isinstance(step, StepRemoveTensorFromCore):
            # subtract from group, only free if entirely clear
            # TODO theoretically we could free part of the token already, but that's not supported by the allocator yet

            group = groups.get_group(step.tensor)
            key = (step.core.id, group.index)
            alloc = curr_core_group_allocated[key]

            alloc.parts_left -= 1
            if alloc.parts_left == 0:
                core_alloc(step.core.id).free(alloc.token, time=step.time_end)
                del curr_core_group_allocated[key]

            step_group_to_token.setdefault(step_index, {})[group.index] = alloc.token

        elif isinstance(step, StepRunNode):

            for tensor in step.inputs:
                group = groups.get_group(tensor)
                key = (step.core.id, group.index)
                alloc = curr_core_group_allocated[key]
                step_group_to_token.setdefault(step_index, {})[group.index] = alloc.token

        else:
            assert False, f"Unknown step type {step}"

    return CoreAllocations(core_allocators=allocators, step_group_to_token=step_group_to_token)


def compile_and_run(
        onnx_path, scme: StreamCostModelEvaluation, node_hw_performances,
        pulp_sdk_path, project_path,
        l1_size: int, l2_size: int,
        simulate: bool, run: bool, plot: bool
):
    random.seed(0xdeadbeef)
    np.random.seed(0xdeadbeef)

    print("Collecting workload")
    assert onnx_path.endswith(".onnx")
    onnx_model = onnx.load(onnx_path, load_external_data=False)
    # noinspection PyUnresolvedReferences
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    workload: DiGraph = scme.workload
    accelerator: Accelerator = scme.accelerator
    cluster_cores = len(accelerator.cores) - 1

    steps = scme.recording.steps

    print("Collecting tensor groups")
    groups = collect_tensor_groups(steps)
    print("Collecting allocations per core")
    allocations = allocate_per_core(groups, steps)

    final_time = max(step.time_end for step in steps)
    for core, allocator in allocations.core_allocators.items():
        print(f"Allocating for core {core}")
        history = allocator.run_allocation(l1_size, final_time)
        history.plot_history(f"outputs/alloc_core_{core}.png", False)

    # TODO onnx is probably not necessary any more
    state = State(
        core_count=cluster_cores,
        onnx_model=onnx_model, workload=workload,
        groups=groups, allocations=allocations,
        simulate=simulate
    )

    return

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

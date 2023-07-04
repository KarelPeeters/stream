import os
import random
import subprocess
from typing import Optional

import numpy as np
import onnx
from networkx import DiGraph

from compiler.allocator import LinearAllocator
from compiler.codegen import generate_code, State, EQUATION_MATMUL, EQUATION_CONV
from compiler.core_allocation import collect_tensor_groups, allocate_per_core
from compiler.data_type import DataType
from compiler.ima_simulate import ima_matmul, ima_conv
from compiler.operation import MemoryKind, Pointer, OperationMatmul, \
    OperationPad, OperationLockIncrement, OperationLockWait, \
    OperationComment, OperationConvPadded, OperationMemClear, ProfileInfo
from compiler.plot_profile import parse_profile_info, plot_profile, CollectedProfile
from stream.classes.cost_model.cost_model import StreamCostModelEvaluation
from stream.classes.cost_model.record import Step, StepRemoveTensorFromCore, StepRunNode, StepAddTensorToCore, \
    StepTransferData
from stream.classes.hardware.architecture.accelerator import Accelerator
from stream.classes.workload.computation_node import ComputationNode


def visit_matmul(core: int, workload, cn: ComputationNode, orig_cn: ComputationNode, state: State):
    raise NotImplementedError("TODO: implement matmul again")

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
def visit_conv(state: State, core: int, step_index: int, step: StepRunNode, orig_cn: ComputationNode):
    cn = step.node

    print("Conv")
    print(f"  inputs: {step.inputs}")
    print(f"  ranges: {cn.loop_ranges}")

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

    assert c_start == c_zero and c_end == c_full, "Partial output accumulation not supported yet"

    # get input groups
    group_per_operand = {}
    for operand, input in zip(step.inputs_operand, step.inputs):
        group = state.groups_per_core[core].get_group(input)
        if operand in group_per_operand:
            assert group_per_operand[operand] == group
        else:
            group_per_operand[operand] = group

    assert group_per_operand.keys() == {"I1", "I2"}

    input_group = group_per_operand["I1"]
    input_place = state.placement_for_group_range(
        core, step_index, input_group,
        {"B": (b_start, b_end), "G": (0, 1), "C": (c_start, c_end), "IY": (iy_start, iy_end), "IX": (ix_start, ix_end)}
    )

    weight_group = group_per_operand["I2"]
    weight_place = state.placement_for_group_range(
        core, step_index, weight_group,
        {"K": (k_start, k_end), "C": (c_start, c_end), "FY": (0, 3), "FX": (0, 3)}
    )

    output_place = state.placement_for_tensor(core, step_index, step.output)

    print(f"Running conv")
    print(f"   input: {input_place}")
    print(f"  weight: {weight_place}")
    print(f"  output: {output_place}")

    assert input_place.tensor.simplify_for_copy().has_simple_strides
    assert weight_place.tensor.simplify_for_copy().has_simple_strides

    out_stride_b, _, out_stride_h, out_stride_w, out_stride_k = output_place.tensor.strides_elem
    assert out_stride_k == 1

    l1_base = state.l1_base_core[core]

    fills_output = output_place.tensor == output_place.padded_tensor
    group_cleared = output_place.group.index in state.groups_cleared_per_core[core]

    # clear the entire group at once if necessary
    # TODO there even more cases where we can skip this, eg. if all convs together cover the entire output
    if not fills_output and not group_cleared:
        state.groups_cleared_per_core[core].add(output_place.group.index)

        group_place = state.placement_for_group_range(core, step_index, output_place.group,
                                                      output_place.group.loop_ranges)

        state.push_cycles(core, "start", "clear")
        state.push_operation(core, OperationMemClear(
            l1_base.offset(output_place.offset_core),
            group_place.tensor,
        ))
        state.push_cycles(core, "end", "clear")

    state.push_cycles(core, "start", "calc")
    state.push_operation(core, OperationConvPadded(
        b=(b_end - b_start), k=(k_end - k_start), c=(c_end - c_start),
        oh=(oy_end - oy_start), ow=(ox_end - ox_start),
        fh=3, fw=3,
        out_stride_b=out_stride_b, out_stride_h=out_stride_h, out_stride_w=out_stride_w,
        input=input_place.offset(l1_base),
        weight=weight_place.offset(l1_base),
        output=output_place.offset(l1_base),
        profile=state.new_profile(ProfileInfo(core=f"ima_core_{core}", name="conv")),
    ))
    state.push_cycles(core, "end", "calc")
    state.push_operation(core, OperationPad())

    # simulate
    if state.should_simulate(node.output_names[0]):
        input_const_trans = state.get_simulation_value(node.input_names[0])
        weight_const_trans = state.get_simulation_value(node.input_names[1])

        # b, g, h, w, c -> b, c, h, w
        input_const = input_const_trans.transpose([0, 1, 4, 2, 3]).squeeze(1)
        # h, w, c, k -> k, c, h, w
        weight_const = weight_const_trans.transpose([3, 2, 0, 1])

        output_const = ima_conv(input_const, weight_const)

        # b, c, h, w -> b, 1, h, w, c
        output_const_trans = np.expand_dims(output_const.transpose([0, 2, 3, 1]), 1)

        state.set_simulation_value(node.output_names[0], output_const_trans)


def visit_step_transfer(state: State, step_index: int, step: StepTransferData):
    offchip_core = state.core_count

    sender = step.sender.id
    receiver = step.receiver.id

    placement_sender = state.placement_for_tensor(sender, step_index, step.tensor)
    placement_receiver = state.placement_for_tensor(receiver, step_index, step.tensor)

    down = sender == offchip_core
    up = receiver == offchip_core

    if not (down ^ up):
        lock_up_ready = state.new_lock()
        lock_down_ready = state.new_lock()

        sender_l1 = placement_sender.offset(state.l1_base_core[sender])
        receiver_l1 = placement_receiver.offset(state.l1_base_core[receiver])

        print(f"Warning: skipped transfer {step}")
        comment = OperationComment(f"Warning: probably incorrect transfer {step.tensor} {sender}->{receiver}")

        state.push_operation(sender, comment)
        state.push_cycles(sender, "start", "send")
        state.push_copy_tensor(sender, state.l2_scratch_space, sender_l1, placement_sender.tensor, False)
        state.push_operation(sender, OperationLockIncrement(lock_up_ready))
        state.push_operation(sender, OperationLockWait(lock_down_ready, 1))
        state.push_cycles(sender, "end", "send")
        state.push_operation(sender, OperationPad())

        # TODO no fabric code?
        state.push_operation(None, comment)
        state.push_operation(None, OperationPad())

        state.push_operation(receiver, comment)
        state.push_cycles(receiver, "start", "receive")
        state.push_operation(receiver, OperationLockWait(lock_up_ready, 1))
        state.push_copy_tensor(receiver, state.l2_scratch_space, receiver_l1, placement_sender.tensor, True)
        state.push_operation(receiver, OperationLockIncrement(lock_down_ready))
        state.push_cycles(receiver, "end", "receive")
        state.push_operation(receiver, OperationPad())
        return

    assert up ^ down
    if down:
        core = receiver
        placement_core = placement_receiver
        placement_offchip = placement_sender
        dir_str = "down"
    else:
        core = sender
        placement_core = placement_sender
        placement_offchip = placement_receiver
        dir_str = "up"

    # TODO support real tensors on the core too
    tensor_core = placement_core.tensor.simplify_for_copy()
    if tensor_core.rank != 1:
        # try copying the padded tensor, and hope that one has simpler strides
        tensor_core = placement_core.padded_tensor.simplify_for_copy()

        # also use the padded version of the offchip tensor
        group_offchip = state.groups_per_core[offchip_core].get_group(step.tensor)
        placement_offchip = state.placement_for_group_range(
            offchip_core,
            step_index,
            group_offchip,
            placement_core.padded_loop_ranges
        )

    assert tensor_core.rank == 1, \
        f"Only simple core tensors supported for now, got {placement_core.tensor} or {placement_core.padded_tensor}"

    # L3 tensor offset happens in copy_to_tensor
    pointer_l3 = state.l3_base.offset(placement_offchip.offset_core)
    pointer_l2 = state.l2_base_core[core].offset(placement_core.offset_core).offset(tensor_core.offset_bytes)
    pointer_l1 = state.l1_base_core[core].offset(placement_core.offset_core).offset(tensor_core.offset_bytes)

    # locks
    fabric_start = state.new_lock()
    fabric_done = state.new_lock()

    # fabric operations
    state.push_operation(None, OperationLockWait(fabric_start, 1))
    state.push_cycles(None, "start", dir_str)
    state.push_copy_tensor(None, pointer_l3, pointer_l2, placement_offchip.tensor, down)
    state.push_operation(None, OperationLockIncrement(fabric_done))
    state.push_cycles(None, "end", dir_str)
    state.push_operation(None, OperationPad())

    # core operations
    state.push_cycles(core, "start", dir_str)

    if down:
        state.push_operation(core, OperationLockIncrement(fabric_start))
        state.push_operation(core, OperationLockWait(fabric_done, 1))
        state.push_copy(core, pointer_l1, pointer_l2, tensor_core.size_bytes)
    else:
        state.push_copy(core, pointer_l2, pointer_l1, tensor_core.size_bytes)
        state.push_operation(core, OperationLockIncrement(fabric_start))
        state.push_operation(core, OperationLockWait(fabric_done, 1))

    state.push_cycles(core, "end", dir_str)
    state.push_operation(core, OperationPad())


def visit_step_run_node(state: State, step_index: int, step: StepRunNode):
    core = step.core.id
    cn = step.node

    orig_cn = cn.original_node if cn.original_node is not None else cn

    if cn.equation == EQUATION_MATMUL:
        # TODO fix visit_matmul
        # visit_matmul(core, cn, orig_cn, state)
        pass
    elif cn.equation == EQUATION_CONV:
        visit_conv(state, core, step_index, step, orig_cn)
    else:
        raise ValueError(f"Unrecognised equation {cn.equation}")


# TODO decide on the strides of all full tensors beforehand instead of this hacky trick
# TODO write some visitor pattern thing for this
def visit_step(state: State, step_index: int, step: Step):
    if isinstance(step, StepAddTensorToCore) or isinstance(step, StepRemoveTensorFromCore):
        # no code to generate, these just affect memory allocation
        pass
    elif isinstance(step, StepTransferData):
        # TODO support direct core<->core through L2?
        print(
            f"step {step.time_start}..{step.time_end}: transfer data {step.tensor} from {step.sender} to {step.receiver}"
        )

        visit_step_transfer(state, step_index, step)

    elif isinstance(step, StepRunNode):
        print(f"step {step.time_start}..{step.time_end}: run {step.node}")
        visit_step_run_node(state, step_index, step)

        # key_node = next(other for other in node_hw_performances.keys() if other.id[0] == cn.id[0])
        # zcme = next(m for c, m in node_hw_performances[key_node].items() if c.id == core)

        # print(f"Visiting {cn}")
        # print(f"  incoming edge nodes: {list(a for a, _ in workload.in_edges(cn))}")
        # print(f"  outgoing edge nodes: {list(b for _, b in workload.out_edges(cn))}")
        # print(f"  inputs: {cn.input_names}")
        # print(f"  outputs: {cn.output_names}")
        # print(f"  loop_ranges: {cn.loop_ranges}")
        # print(f"  temporal: {zcme.temporal_mapping}")
        # print(f"  spatial: {zcme.spatial_mapping}")

    else:
        assert False, f"Unknown step type {step}"


def compile_and_run(
        onnx_path, scme: StreamCostModelEvaluation, node_hw_performances,
        pulp_sdk_path, project_path,
        l1_size: int, l2_size: int,
        simulate: bool, run: bool, plot: bool,
        output_path: str,
) -> Optional[CollectedProfile]:
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
    groups_per_core = collect_tensor_groups(cluster_cores, steps, output_path)
    print("Collecting allocations per core")
    allocations = allocate_per_core(groups_per_core, steps)

    final_time = max(step.time_end for step in steps)
    for core, allocator in enumerate(allocations.core_allocators):
        size = None if core == cluster_cores else l1_size
        print(f"Allocating for core {core} with size {size}")
        history = allocator.run_allocation(size, final_time)
        os.makedirs(f"{output_path}/alloc", exist_ok=True)
        history.plot_history(f"{output_path}/alloc/alloc_core_{core}_block.png",
                             f"{output_path}/alloc/alloc_core_{core}_line.png")

    # TODO onnx is probably not necessary any more
    state = State(
        core_count=cluster_cores,
        onnx_model=onnx_model, workload=workload,
        groups_per_core=groups_per_core, allocations=allocations,
        simulate=simulate,
    )

    print("Generating code for steps")
    for (step_index, step) in enumerate(steps):
        visit_step(state, step_index, step)

    # TODO remove
    # print("Allocated buffers:")
    # for name, buffer in state.buffers.items():
    #     print(f"  {name}: {buffer.padded_tensor} {buffer.inner_tensor}")

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
        with open(f"{output_path}/stdout.txt", "w") as f:
            f.write(stdout)

        result.check_returncode()
        profile = parse_profile_info(stdout)

        if plot:
            plot_profile(profile, f"{output_path}/profile.png", block=False)

        return profile

    return None

from dataclasses import dataclass
from math import prod
from typing import Set, Any, Tuple, Dict, List, Optional

from matplotlib import pyplot as plt

from compiler.allocator import Token, TimeAllocator
from stream.classes.cost_model.record import Step, StepAddTensorToCore, StepRemoveTensorFromCore, StepRunNode, \
    StepTransferData


@dataclass
class Group:
    index: int
    elem_size_bits: int
    tensor_keys: Set[Any]
    loop_ranges: Dict[str, Tuple[int, int]]


@dataclass
class TensorGroups:
    key_to_group: Dict[Any, int]
    key_to_tensor: Dict[Any, Any]
    groups: List[Group]

    def get_group(self, tensor) -> Group:
        key = tensor.equality_key()
        group = self.key_to_group[key]
        return self.groups[group]


@dataclass
class PartialGroup:
    tensor_keys: Set[Any]
    loop_ranges: Dict[str, Tuple[int, int]]
    elem_size_bits: Optional[int]

    def merge_loop_ranges(self, loop_ranges: Dict[str, Tuple[int, int]], must_match: bool):
        if must_match:
            assert self.loop_ranges.keys() == loop_ranges.keys()

        for d in self.loop_ranges.keys():
            if d in loop_ranges:
                self.loop_ranges[d] = (
                    min(self.loop_ranges[d][0], loop_ranges[d][0]),
                    max(self.loop_ranges[d][1], loop_ranges[d][1])
                )


class TensorGrouper:
    def __init__(self):
        self.groups: List[Optional[PartialGroup]] = []
        self.key_to_group = {}
        self.key_to_tensor = {}

    def get_group(self, tensor, allow_new: bool) -> int:
        key = tensor.equality_key()
        self.key_to_tensor.setdefault(key, tensor)

        if key in self.key_to_group:
            return self.key_to_group[key]
        else:
            assert allow_new, f"New group not allowed, tensor {tensor}"

        elem_size_bits = tensor.size / prod(end - start for start, end in tensor.loop_ranges)
        assert int(elem_size_bits) == elem_size_bits
        elem_size_bits = int(elem_size_bits)

        group = len(self.groups)
        self.groups.append(PartialGroup(
            tensor_keys={key},
            loop_ranges={d: r for d, r in zip(tensor.loop_dimensions, tensor.loop_ranges)},
            elem_size_bits=elem_size_bits,
        ))
        self.key_to_group[key] = group
        return group

    def merge_groups(self, x_group: int, y_group: int):
        if x_group == y_group:
            return x_group

        x_partial = self.groups[x_group]
        y_partial = self.groups[y_group]

        x_partial.tensor_keys |= y_partial.tensor_keys
        x_partial.merge_loop_ranges(y_partial.loop_ranges, must_match=True)

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

        for partial in self.groups:
            if partial is None:
                group_map.append(None)
                continue

            index = len(groups)
            group_map.append(index)

            groups.append(Group(
                index=index,
                elem_size_bits=partial.elem_size_bits,
                tensor_keys=partial.tensor_keys,
                loop_ranges=partial.loop_ranges
            ))

        # map group indices
        key_to_group = {
            k: group_map[v] for k, v in self.key_to_group.items()
        }

        return TensorGroups(
            key_to_group=key_to_group,
            key_to_tensor=self.key_to_tensor,
            groups=groups
        )


def collect_tensor_groups(cores: int, steps: List[Step]) -> List[TensorGroups]:
    core_tensor_lifetime = {}
    max_lifetime = 0

    merger_per_core = [TensorGrouper() for _ in range(cores + 1)]

    # TODO handle the case where a tensor is evicted and then later loaded back onto the same core
    for step in steps:
        print(step)
        max_lifetime = max(max_lifetime, step.time_end)

        if isinstance(step, StepAddTensorToCore):
            tensor_key = (step.core, step.tensor.equality_key())

            assert tensor_key not in core_tensor_lifetime
            core_tensor_lifetime[tensor_key] = [step.time_start, None, None]

            # ensure a group is created for this tensor
            merger_per_core[step.core.id].get_group(step.tensor, allow_new=True)

        elif isinstance(step, StepRemoveTensorFromCore):
            # never remove tensors from the offchip core
            if step.core.id == cores:
                continue

            tensor_key = (step.core, step.tensor.equality_key())

            if tensor_key in core_tensor_lifetime:
                assert core_tensor_lifetime[tensor_key][2] is None
                core_tensor_lifetime[tensor_key][2] = step.time_start
            else:
                print(f"  Warning: {step} has no matching add step")

            # double check that the group for this tensor already exists
            merger_per_core[step.core.id].get_group(step.tensor, allow_new=False)

        elif isinstance(step, StepRunNode):
            for x in step.inputs:
                print(f"  {x}")

            for x in step.inputs:
                tensor_key = (step.core, x.equality_key())
                core_tensor_lifetime[tensor_key][1] = step.time_end

            core_merger = merger_per_core[step.core.id]

            # we only need to merge dims in which there is potential padding
            merged_keys = [
                ("IY", "IY"),
                ("IX", "IX"),
                ("IY", "OY"),
                ("IX", "OX"),
             ]
            merged_ranges = {k1: step.node.loop_ranges[k0] for (k0, k1) in merged_keys if k0 in step.node.loop_ranges}

            print(f"Merging in loop ranges: {merged_ranges}")

            for x in step.inputs:
                core_merger.groups[core_merger.get_group(x, allow_new=False)] \
                    .merge_loop_ranges(merged_ranges, must_match=False)

            core_merger.merge_matching_tensors(step.inputs)

        elif isinstance(step, StepTransferData):
            # doesn't influence group allocation
            pass

        else:
            assert False, f"Unknown step type {step}"

        print()

    # merge all tensors that are conceptually the same on the offchip core
    merger_per_core[-1].merge_matching_tensors(list(merger_per_core[-1].key_to_tensor.values()))

    merged_groups_per_core = [m.finish() for m in merger_per_core]

    print("grouped tensors: ")
    for core, groups in enumerate(merged_groups_per_core):
        print(f"  core {core}:")
        for group in groups.groups:
            print(f"    {group}")

    # TODO move this plotting to a seprate function
    colormap = plt.get_cmap("Set1")
    max_group_count = max(len(m.groups) for m in merged_groups_per_core)
    group_color = [colormap.colors[i % len(colormap.colors)] for i in range(max_group_count)]

    for core_id in range(cores + 1):
        # collect items on this core, sorted by group
        core_groups = merged_groups_per_core[core_id]
        core_tensors = [
            (tensor_key, lifetime) for (other_core, tensor_key), lifetime in core_tensor_lifetime.items()
            if other_core.id == core_id
        ]
        core_tensors = sorted(core_tensors, key=lambda x: core_groups.get_group(core_groups.key_to_tensor[x[0]]).index)

        fig, axes = plt.subplots(
            nrows=len(core_tensors),
            sharex="all", squeeze=False, figsize=(32, 32)
        )
        axes = axes.squeeze(1)

        for i, (tensor_key, (start, last_used, end)) in enumerate(core_tensors):
            tensor = core_groups.key_to_tensor[tensor_key]
            print(f"Slice {core_id} {tensor} {start}..{end} last={last_used}")

            ax = axes[i]
            ax.set_ylabel(f"{core_id}, {tensor}", rotation='horizontal', ha='right')

            if end is None:
                end = max_lifetime

            if start is not None and end is not None:
                color = group_color[core_groups.get_group(tensor).index]
                ax.axvspan(start, end, facecolor=color, alpha=1.0)
            else:
                print(f"Warning: {core_id} {tensor} has invalid lifetime {start}..{end}")

        # fig.tight_layout()
        plt.savefig(f"outputs/tensor_core_life_{core_id}.png")
        plt.close()

    return merged_groups_per_core


@dataclass
class AllocatedGroup:
    token: Token
    parts_left: int


@dataclass
class CoreAllocations:
    core_allocators: List[TimeAllocator]
    core_group_step_to_token: Dict[Tuple[int, int], List[Tuple[int, Token]]]

    def get_token_for_group(self, core: int, group: int, step: int) -> Token:
        step_list = self.core_group_step_to_token[(core, group)]

        prev_token = None

        for curr_step, token in step_list:
            # find the last token with curr_step <= step
            if curr_step > step:
                assert prev_token is not None
                return prev_token
            prev_token = token

        assert prev_token is not None
        return prev_token


def allocate_per_core(groups_per_core: List[TensorGroups], steps: List[Step]) -> CoreAllocations:
    curr_core_group_allocated = {}
    core_group_step_to_token = {}

    allocators = [TimeAllocator(start_time=0) for _ in range(len(groups_per_core))]

    for step_index, step in enumerate(steps):
        if isinstance(step, StepAddTensorToCore):
            # allocate entire group at once
            core = step.core.id
            group = groups_per_core[core].get_group(step.tensor)
            key = (core, group.index)

            if key in curr_core_group_allocated:
                alloc = curr_core_group_allocated[key]
                alloc.parts_left += 1

            else:
                group_size_bits = group.elem_size_bits * prod(end - start for start, end in group.loop_ranges.values())
                group_size_bytes = (group_size_bits + 7) // 8

                token = allocators[core].alloc(group_size_bytes, time=step.time_start)
                alloc = AllocatedGroup(token, len(group.tensor_keys))

                curr_core_group_allocated[key] = alloc

            core_group_step_to_token.setdefault(key, []).append((step_index, alloc.token))

        elif isinstance(step, StepRemoveTensorFromCore):
            # subtract from group, only free if entirely clear
            # TODO theoretically we could free part of the token already, but that's not supported by the allocator yet

            core = step.core.id
            group = groups_per_core[core].get_group(step.tensor)
            key = (core, group.index)
            alloc = curr_core_group_allocated[key]

            alloc.parts_left -= 1
            if alloc.parts_left == 0:
                allocators[core].free(alloc.token, time=step.time_end)
                del curr_core_group_allocated[key]

            core_group_step_to_token.setdefault(key, []).append((step_index, alloc.token))

        elif isinstance(step, StepRunNode):
            core = step.core.id

            for tensor in step.inputs:
                # asser that operands have been allocated
                group = groups_per_core[core].get_group(tensor)
                key = (step.core.id, group.index)
                _ = curr_core_group_allocated[key]

        elif isinstance(step, StepTransferData):
            # doesn't influence allocation
            pass

        else:
            assert False, f"Unknown step type {step}"

    return CoreAllocations(core_allocators=allocators, core_group_step_to_token=core_group_step_to_token)

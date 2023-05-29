from dataclasses import dataclass
from math import prod
from typing import Set, Any, Tuple, Dict, List

from matplotlib import pyplot as plt

from compiler.allocator import Token, TimeAllocator
from stream.classes.cost_model.record import Step, StepAddTensorToCore, StepRemoveTensorFromCore, StepRunNode


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

from abc import ABC
from dataclasses import dataclass

from stream.classes.workload.computation_node import ComputationNode
from stream.classes.workload.tensor import Tensor


class Step(ABC):
    pass


@dataclass
class StepAddTensorToCore(Step):
    core: int
    tensor: Tensor


@dataclass
class StepRemoveTensorFromCore(Step):
    core: int
    tensor: Tensor
    write_offchip: bool


@dataclass
class StepRunNode(Step):
    node: ComputationNode


class RecordedSchedule:
    def __init__(self):
        self.steps = []

    def push(self, step: Step):
        self.steps.append(step)


class RemoveTensorFromCore(Step):
    pass

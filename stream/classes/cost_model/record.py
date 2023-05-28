from abc import ABC
from dataclasses import dataclass
from typing import Tuple

from zigzag.classes.hardware.architecture.core import Core

from stream.classes.workload.computation_node import ComputationNode
from stream.classes.workload.tensor import Tensor


@dataclass(kw_only=True)
class Step(ABC):
    time_start: int
    time_end: int


@dataclass(kw_only=True)
class StepAddTensorToCore(Step):
    core: Core
    tensor: Tensor


@dataclass(kw_only=True)
class StepRemoveTensorFromCore(Step):
    time_start: int
    time_end: int
    core: Core
    tensor: Tensor
    write_offchip: bool


@dataclass(kw_only=True)
class StepRunNode(Step):
    time_start: int
    time_end: int
    core: Core
    node: ComputationNode
    inputs: Tuple[Tensor]


class RecordedSchedule:
    def __init__(self):
        self.steps = []

    def push(self, step: Step):
        self.steps.append(step)

import bisect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List

from zigzag.classes.hardware.architecture.core import Core

from stream.classes.hardware.architecture.communication_link import CommunicationLink
from stream.classes.workload.computation_node import ComputationNode
from stream.classes.workload.tensor import Tensor


@dataclass(kw_only=True)
class Step(ABC):
    time_start: int
    time_end: int

    @abstractmethod
    def priority(self) -> int:
        return 0


@dataclass(kw_only=True)
class StepAddTensorToCore(Step):
    core: Core
    tensor: Tensor

    def priority(self) -> int:
        return 10


@dataclass(kw_only=True)
class StepRemoveTensorFromCore(Step):
    core: Core
    tensor: Tensor

    def priority(self) -> int:
        return -10


@dataclass(kw_only=True)
class StepTransferData(Step):
    tensor: Tensor
    sender: Core
    receiver: Core
    links: List[CommunicationLink]

    def priority(self) -> int:
        return 1


@dataclass(kw_only=True)
class StepRunNode(Step):
    core: Core
    node: ComputationNode
    inputs: Tuple[Tensor]
    # TODO remove these and look at the CN directly again?
    inputs_operand: Tuple[str]
    output: Tensor

    def priority(self) -> int:
        return 0


class RecordedSchedule:
    def __init__(self):
        self.steps = []

    def push(self, step: Step):
        def key(s):
            return s.time_start, -s.priority()

        # insert first based on time then on priority
        index = bisect.bisect_right(self.steps, key(step), key=key)
        self.steps.insert(index, step)

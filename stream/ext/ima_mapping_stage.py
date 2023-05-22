import json
import logging
import os
from typing import Generator, Tuple, Any

from zigzag.classes.cost_model.cost_model import CostModelEvaluation
from zigzag.classes.stages import MainStage, MinimalLatencyStage, SpatialMappingGeneratorStage, LomaStage, \
    CostModelStage, Stage

from stream.classes.stages import IntraCoreMappingStage
from stream.classes.workload.computation_node import ComputationNode

logger = logging.getLogger(__name__)


# class ImaSetUserProvidedMappingStage(Stage):
#     def run(self):
#         for node in self.kwargs["workload"]:
#             node: ComputationNode
#             node.user_spatial_mapping = "derp"
#             print(f"setting node {node}")
#
#         print([node.user_spatial_mapping for node in self.kwargs["workload"]])
#
#         sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
#         for cme, extra_info in sub_stage.run():
#             yield cme, extra_info


class ImaIntraCoreMappingState(IntraCoreMappingStage):
    def get_intra_core_mapping_flow(self, node, too_large_operands, core_id):

        logger.info(f"Launching intra-core mapping optimization for {node}...")

        if too_large_operands:
            accelerator = self.add_offchip_to_core(core_id, too_large_operands, node.id[0])
        else:
            accelerator = self.accelerator

        main_stage = MainStage([  # Initializes the MainStage as entry point
            MinimalLatencyStage,
            SpatialMappingGeneratorStage,  # Generates multiple spatial mappings (SM)
            MinimalLatencyStage,  # Reduces all CMEs, returning minimal latency one
            LomaStage,  # Generates multiple temporal mappings (TM)
            ImaCostModelStage  # Evaluates generated SM and TM through cost model
        ],
            layer=node,
            accelerator=accelerator,  # required by a number of stages
            loma_lpf_limit=self.loma_lpf_limit,  # required by LomaStage
            loma_show_progress_bar=self.loma_show_progress_bar
        )
        return main_stage


with open(os.path.join(os.path.dirname(__file__), "ima_profile.json"), "r") as f:
    IMA_PROFILE_DATA = json.load(f)


class ImaCostModelEvaluation(CostModelEvaluation):
    def run(self):
        plot_per_area = IMA_PROFILE_DATA["plot_per_area"]
        job_offset = IMA_PROFILE_DATA["job_offset"]
        job_b_factor = IMA_PROFILE_DATA["job_per_b"]

        # TODO ensure that this is actually an IMA core
        core = self.accelerator.get_core(self.core_id)

        # TODO is is extraction of b, k, c correct?
        k = None
        c = None

        for key, value in self.spatial_mapping.spatial_loop_dim_size:
            if key == "K":
                k = value
            elif key == "C":
                c = value

        assert k is not None and c is not None
        assert round(k) == k and round(c) == c
        k = round(k)
        c = round(c)

        b = self.temporal_mapping.total_cycle

        factor = job_b_factor[f"{k}x{c}"]
        plot_cycles = plot_per_area * k * c
        job_cycles = job_offset + factor * b
        total_cycles = plot_cycles + job_cycles

        print(f"IMA cost evaluation for {b}x{k}x{c} => {plot_cycles}+{job_cycles}={total_cycles}")

        # TODO is this okay?
        self.latency_total1 = total_cycles
        self.latency_total2 = total_cycles

        self.energy_total = 0


class ImaCostModelStage(CostModelStage):
    def run(self) -> Generator[Tuple[ImaCostModelEvaluation, Any], None, None]:
        """
        Run the cost model stage by calling the internal zigzag cost model with the correct inputs.
        """
        self.cme = ImaCostModelEvaluation(
            accelerator=self.accelerator,
            layer=self.layer,
            spatial_mapping=self.spatial_mapping,
            temporal_mapping=self.temporal_mapping,
            # the below parameter is optional
            access_same_data_considered_as_no_access=self.access_same_data_considered_as_no_access,
        )
        yield self.cme, None

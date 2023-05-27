import json
import logging
import os
from typing import Generator, Tuple, Any

from zigzag.classes.cost_model.cost_model import CostModelEvaluation
from zigzag.classes.stages import MainStage, MinimalLatencyStage, SpatialMappingGeneratorStage, LomaStage, \
    CostModelStage

from stream.classes.stages import IntraCoreMappingStage

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

        # TODO is any of this reasonable? we're basically ignoring
        #   the mapping engines completely

        spatial_dims = {}
        for dim, size in self.spatial_mapping.spatial_loop_dim_size:
            assert int(size) == size
            spatial_dims[dim] = int(size)

        temporal_dims = {}
        for operand, values in self.mapping.temporal_mapping.mapping_dic_origin.items():
            total_sizes = {}

            # TODO does the flattening make sense? we're losing the memory hierarchy info here
            values_flat = [v for sub_values in values for v in sub_values]
            for dim, size in values_flat:
                assert int(size) == size
                size = int(size)

                if dim not in total_sizes:
                    total_sizes[dim] = 1
                total_sizes[dim] *= size
            for dim, size in total_sizes.items():
                if dim in temporal_dims:
                    assert temporal_dims[dim] == size
                else:
                    temporal_dims[dim] = size

        assert len(spatial_dims.keys() & temporal_dims.keys()) == 0
        all_dims = spatial_dims | temporal_dims

        if all_dims.keys() == {"B", "K", "C"}:
            # matmul
            ima_b = all_dims["B"]
            ima_k = all_dims["K"]
            ima_c = all_dims["C"]
        elif all_dims.keys() == {"B", "K", "C", "OY", "OX", "FY", "FX"} or all_dims.keys() == {"K", "C", "OY", "OX",
                                                                                               "FY", "FX"}:
            # conv
            B = all_dims.get("B", 1)
            K = all_dims["K"]
            C = all_dims["C"]
            OY = all_dims["OY"]
            OX = all_dims["OX"]
            FY = all_dims["FY"]
            FX = all_dims["FX"]

            ima_b = B * OY * OX
            ima_k = K
            ima_c = C * FY * FX
        else:
            raise KeyError(f"Unexpected combination of dims: {all_dims}")

        factor = job_b_factor[f"{ima_k}x{ima_c}"]
        plot_cycles = plot_per_area * ima_k * ima_c
        job_cycles = job_offset + factor * ima_b
        total_cycles = plot_cycles + job_cycles

        # print(f"IMA cost evaluation for {b}x{k}x{c} => {plot_cycles}+{job_cycles}={total_cycles}")
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

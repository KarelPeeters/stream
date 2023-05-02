import pickle
import re
from typing import Generator

from zigzag.classes.stages import *

from compiler.main import compile_and_run
from stream.api_edited import save_graph
from stream.classes.stages import *


class DebugStage(Stage):
    def run(self) -> Generator:
        kwargs = self.kwargs.copy()
        save_graph(self.kwargs["workload"], f"outputs/graph.svg", all_ranges=True)

        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        for i, (cme, extra_info) in enumerate(sub_stage.run()):
            yield cme, extra_info


def get_hardware_performance_stream(hardware, workload, mapping, CN_define_mode, hint_loops, node_hw_performances_path):
    # Initialize the logger
    import logging as _logging
    _logging_level = _logging.INFO
    # _logging_format = '%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
    _logging_format = '%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
    _logging.basicConfig(level=_logging_level,
                         format=_logging_format)

    mainstage = MainStage([  # Initializes the MainStage as entry point
        AcceleratorParserStage,  # Parses the accelerator
        StreamONNXModelParserStage,  # Parses the ONNX Model into the workload
        # UserDefinedModelParserStage,  # Parses the user-defined Model into the workload
        GenerateCNWorkloadHybridStage,
        DebugStage,
        IntraCoreMappingStage,
        InterCoreMappingStage,
    ],

        accelerator=hardware,  # required by AcceleratorParserStage
        workload_path=workload,  # required by ModelParserStage
        mapping_path=mapping,  # required by ModelParserStage
        loma_lpf_limit=6,  # required by LomaStage
        nb_ga_individuals=4,  # number of individuals in each genetic algorithm generation
        nb_ga_generations=0,  # number of genetic algorithm generations
        node_hw_performances_path=node_hw_performances_path,
        # saved node_hw_performances to skip re-computation
        plot_hof=True,  # Save schedule and memory usage plot of each individual in the Genetic Algorithm hall of fame
        plot_file_name=True,
        plot_full_schedule=True,
        plot_data_transfer=True,
        cn_define_mode=CN_define_mode,
        hint_loops=hint_loops,
        scheduler_candidate_selection='memory'
    )

    # Launch the MainStage
    answers = mainstage.run()
    return answers


def main():
    # accelerator = 'inputs.examples.hardware.TPU_like_quad_core'
    # workload = 'inputs.examples.workload.resnet18_few_layers'
    # mapping = 'inputs.examples.mapping.tpu_like_quad_core'

    CN_define_mode = 1
    # hint_loops = [('B', 'all')]
    # hint_loops = [('OY', 'all'), ('OX', 'all')]
    # hint_loops = [('OY', 'all')]
    # hint_loops = [('OY', 4)]
    hint_loops = []

    accelerator = 'inputs.testing.hardware.dual_testing_core_offchip'
    # workload = 'inputs.testing.workload.testing_workload_for_2_cores'
    # workload = 'inputs.testing.workload.simple_example_workload'
    # workload = r"C:\Documents\Programming\Python\MLPlayGround\branching_conv.onnx"
    # workload = r"C:\Documents\Programming\Python\MLPlayGround\linear_conv.onnx"
    workload = "inputs/onnx/linear.onnx"
    mapping = 'inputs.testing.mapping.testing_mapping'

    hw_name = accelerator.split(".")[-1]
    wl_name = re.split(r"/|\.", workload)[-1]
    experiment_id = f"{hw_name}-{wl_name}-CNmode_{CN_define_mode}-hintloop_{str(hint_loops)}"
    node_hw_cost_pkl_name = f'saved_CN_HW_cost-{experiment_id}'

    node_hw_performances_path = f"outputs/{node_hw_cost_pkl_name}.pickle"

    scme, _ = get_hardware_performance_stream(accelerator, workload, mapping, CN_define_mode, hint_loops,
                                              node_hw_performances_path)

    from stream.visualization.schedule import plot_timeline_brokenaxes
    from stream.visualization.memory_usage import plot_memory_usage
    from stream.visualization.plot_scme import bar_plot_stream_cost_model_evaluations_breakdown

    plot_stream = False
    plot_profile = False

    if plot_stream:
        draw_dependencies = True
        plot_data_transfer = True
        section_start_percent = (0,)
        percent_shown = (100,)
        timeline_fig_path = "outputs/schedule_plot.png"
        memory_fig_path = "outputs/memory_plot.png"
        energy_fig_path = "outputs/energy_plot.png"
        plot_timeline_brokenaxes(scme[0].workload, scme[0].accelerator, draw_dependencies, section_start_percent,
                                 percent_shown, plot_data_transfer, fig_path=timeline_fig_path)
        plot_memory_usage(scme[0].accelerator.memory_manager, fig_path=memory_fig_path)
        bar_plot_stream_cost_model_evaluations_breakdown([scme], fig_path=energy_fig_path)

    with open(node_hw_performances_path, "rb") as f:
        node_hw_performances = pickle.load(file=f)

    pulp_sdk_path = r"~/new-attempt/pulp-sdk"
    project_path = r"~/new-attempt/pulp-sdk/applications/custom"
    compile_and_run(workload, scme[0], node_hw_performances, pulp_sdk_path, project_path, run=True, plot=plot_profile)


if __name__ == "__main__":
    main()

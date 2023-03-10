import re
from typing import Generator

import networkx
from matplotlib import pyplot as plt
from zigzag.classes.stages import *

from stream.classes.stages import *
from stream.classes.workload.computation_node import ComputationNode


class DebugStage(Stage):
    def run(self) -> Generator:
        kwargs = self.kwargs.copy()
        save_graph(self.kwargs["workload"], f"outputs/graph.svg", all_ranges=True)

        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        for i, (cme, extra_info) in enumerate(sub_stage.run()):
            yield cme, extra_info


def get_hardware_performance_stream(hardware, workload, mapping, CN_define_mode, hint_loops):

    # Initialize the logger
    import logging as _logging
    _logging_level = _logging.INFO
    # _logging_format = '%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
    _logging_format = '%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
    _logging.basicConfig(level=_logging_level,
                         format=_logging_format)

    mainstage = MainStage([  # Initializes the MainStage as entry point
        AcceleratorParserStage,  # Parses the accelerator
        # StreamONNXModelParserStage,  # Parses the ONNX Model into the workload
        UserDefinedModelParserStage,  # Parses the user-defined Model into the workload
        GenerateCNWorkloadHybridStage,
        IntraCoreMappingStage,
        InterCoreMappingStage,
    ],

        accelerator=hardware,  # required by AcceleratorParserStage
        workload_path=workload,  # required by ModelParserStage
        mapping_path=mapping,  # required by ModelParserStage
        loma_lpf_limit=6,  # required by LomaStage
        nb_ga_individuals=4,  # number of individuals in each genetic algorithm generation
        nb_ga_generations=1,  # number of genetic algorithm generations
        node_hw_performances_path=f"outputs/{node_hw_cost_pkl_name}.pickle",  # saved node_hw_performances to skip re-computation
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


def save_graph(graph, path: str, all_ranges: bool):
    for (node, data) in graph.nodes(data=True):
        node: ComputationNode
        rows = ""

        rows += f'<tr><td colspan="3"><b>{node}</b></td></tr>'
        rows += f'<tr><td colspan="3">{type(node).__name__}</td></tr>'
        rows += f"<tr><td>time</td><td>{node.start}</td><td>{node.end}</td></tr>"
        for axis, (start, end) in node.loop_ranges.items():
            if all_ranges:
                plot_range = True
            else:
                const = True
                for (other, other_data) in graph.nodes(data=True):
                    if axis in other.loop_ranges and other.loop_ranges[axis] != (start, end):
                        const = False
                        break
                plot_range = not const

            if plot_range:
                rows += f"<tr><td>{axis}</td><td>{start}</td><td>{end}</td></tr>"

        data["label"] = f'<<table border="0">{rows}</table>>'

    graph_dot = networkx.drawing.nx_pydot.to_pydot(graph)

    print(f"Writing to {path}")
    with open(path, "wb") as f:
        f.write(graph_dot.create_svg())


if __name__ == "__main__":
    # accelerator = 'inputs.examples.hardware.TPU_like_quad_core'
    # workload = 'inputs.examples.workload.resnet18_few_layers'
    # mapping = 'inputs.examples.mapping.tpu_like_quad_core'

    accelerator = 'inputs.testing.hardware.quad_testing_core_offchip'
    workload = 'inputs.testing.workload.testing_workload_for_4_cores'
    mapping = 'inputs.testing.mapping.testing_mapping'

    CN_define_mode = 1  # manually define outer CN size for all cores and all layers
    hint_loops = [('OY', 'all')]
    # hint_loops = []

    hw_name = accelerator.split(".")[-1]
    wl_name = re.split(r"/|\.", workload)[-1]
    experiment_id = f"{hw_name}-{wl_name}-CNmode_{CN_define_mode}-hintloop_{str(hint_loops)}"
    node_hw_cost_pkl_name = f'saved_CN_HW_cost-{experiment_id}'

    scme, _ = get_hardware_performance_stream(accelerator, workload, mapping, CN_define_mode, hint_loops)

    from stream.visualization.schedule import plot_timeline_brokenaxes
    from stream.visualization.memory_usage import plot_memory_usage
    from stream.visualization.plot_scme import bar_plot_stream_cost_model_evaluations_breakdown
    plot_full_schedule = True
    draw_dependencies = True
    plot_data_transfer = True
    section_start_percent = (0,)
    percent_shown = (100,)
    timeline_fig_path = "outputs/schedule_plot.png"
    memory_fig_path = "outputs/memory_plot.png"
    energy_fig_path = "outputs/energy_plot.png"
    plot_timeline_brokenaxes(scme[0].workload, scme[0].accelerator, draw_dependencies, section_start_percent, percent_shown, plot_data_transfer, fig_path=timeline_fig_path)
    plot_memory_usage(scme[0].accelerator.memory_manager, fig_path=memory_fig_path)
    # bar_plot_stream_cost_model_evaluations_breakdown([scme], fig_path=energy_fig_path)

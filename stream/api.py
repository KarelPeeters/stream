import os
import pickle
import random
import re
from typing import Generator

import onnx
import torch
from matplotlib import pyplot as plt
from onnx import shape_inference
from zigzag.classes.stages import *

from compiler.main import compile_and_run
from stream.api_edited import save_graph
from stream.classes.cost_model.cost_model import StreamCostModelEvaluation
from stream.classes.hardware.architecture.communication_link import CommunicationLink
from stream.classes.stages import *
from stream.ext.ima_mapping_stage import ImaIntraCoreMappingState
from stream.inputs.testing.hardware.custom.ima import ima_with_offchip
from stream.test_network import TestNetwork


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
        # AcceleratorParserStage,  # Parses the accelerator
        StreamONNXModelParserStage,  # Parses the ONNX Model into the workload
        # UserDefinedModelParserStage,  # Parses the user-defined Model into the workload
        GenerateCNWorkloadHybridStage,
        DebugStage,
        ImaIntraCoreMappingState,
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


def sort_dict(d: dict, f: callable = lambda x: x):
    return dict(sorted(d.items(), key=lambda item: f(item[0])))


def time_str(start, end):
    return f"{start:6} .. {end:6} = {end - start:6}"


def print_workload_per_core(scme: StreamCostModelEvaluation):
    print("Workload:")

    # get_name_for_schedule_plot

    # collect info
    nodes_per_core = {}
    for node in scme.workload.nodes:
        nodes_per_core.setdefault(node.core_allocation, []).append(node)

    # actually print things
    for core, nodes in sort_dict(nodes_per_core).items():
        nodes = sorted(nodes, key=lambda n: n.start)

        print(f"  core {core}:")
        for node in nodes:
            print(f"    {time_str(node.start, node.end)} {node}")

    seen_links = set()
    for (key, links) in sort_dict(scme.accelerator.pair_links, f=lambda c: (c[0].id, c[1].id)).items():
        for link in links:
            link: CommunicationLink
            if link in seen_links:
                continue
            seen_links.add(link)

            # name = link.get_name_for_schedule_plot()
            # print(f"  link {name} {key} {link}:")

            if link.bidirectional:
                arrow = "<->"
            else:
                arrow = "-->"

            periods = [
                *(("active", s, e, o, g) for (s, e, o, g) in link.active_periods),
                *(("blocked", s, e, None, g) for (s, e, g) in link.blocked_periods),
            ]
            periods = sorted(periods, key=lambda p: p[1])

            if len(periods) == 0:
                continue

            print(f"  link {key[0].id}{arrow}{key[1].id} bw={link.bandwidth}")
            for (kind, start, end, operand, origin) in periods:
                print(f"    {time_str(start, end)} {kind} {operand} {origin}")


def export_onnx(path):
    print("Exporting ONNX model")

    network = TestNetwork()

    input = network.example_input()
    _ = network(input)
    torch.onnx.export(network, input, path)

    model = onnx.load_model(path)
    model_shaped = shape_inference.infer_shapes(model)
    onnx.save_model(model_shaped, path)


def main():
    random.seed(0xdeadbeef)

    # accelerator = 'inputs.examples.hardware.TPU_like_quad_core'
    # workload = 'inputs.examples.workload.resnet18_few_layers'
    # mapping = 'inputs.examples.mapping.tpu_like_quad_core'

    CN_define_mode = 1
    # hint_loops = [('B', 'all')]
    # hint_loops = [('OY', 'all'), ('OX', 'all')]
    # hint_loops = [('OY', 'all')]
    hint_loops = [('OY', 4)]
    # hint_loops = [('B', 4)]
    # hint_loops = []

    # TODO divide L2 size by the number of cores?
    # TODO higher level: start properly using L2, maybe just as another global cache external to the cores?
    l1_size = 0x00100000
    l2_size = 0x60000000

    # accelerator = 'inputs.testing.hardware.dual_testing_core_offchip'
    # TODO use weight_size=4 at some point
    # TODO does weight_size even matter any more?
    accelerator = ima_with_offchip(
        core_count=2,
        width=256, height=256,
        weight_size=4,
        l1_bits=l1_size * 8, l2_bits=l2_size * 8
    )
    # workload = 'inputs.testing.workload.testing_workload_for_2_cores'
    # workload = 'inputs.testing.workload.simple_example_workload'
    # workload = r"C:\Documents\Programming\Python\MLPlayGround\branching_conv.onnx"
    # workload = r"C:\Documents\Programming\Python\MLPlayGround\linear_conv.onnx"
    workload = "inputs/onnx/linear.onnx"
    mapping = 'inputs.testing.mapping.testing_mapping'

    export_onnx(workload)

    hw_name = accelerator.name.split(".")[-1]
    wl_name = re.split(r"/|\.", workload)[-1]
    experiment_id = f"{hw_name}-{wl_name}-CNmode_{CN_define_mode}-hintloop_{str(hint_loops)}"
    node_hw_cost_pkl_name = f'saved_CN_HW_cost-{experiment_id}'

    node_hw_performances_path = f"outputs/{node_hw_cost_pkl_name}.pickle"

    if os.path.exists(node_hw_performances_path):
        os.remove(node_hw_performances_path)

    print("Running stream")
    scme, _ = get_hardware_performance_stream(accelerator, workload, mapping, CN_define_mode, hint_loops,
                                              node_hw_performances_path)

    from stream.visualization.schedule import plot_timeline_brokenaxes

    # TODO clean up plotting bools and blocking
    # TODO why does simulate false cause out-of-bounds RAM errors?
    generate = True
    simulate = True
    run = True
    plot_stream = True
    plot_profile = True

    print_workload_per_core(scme[0])

    if plot_stream:
        draw_dependencies = True
        plot_data_transfer = True
        section_start_percent = (0,)
        percent_shown = (100,)
        timeline_fig_path = "outputs/schedule_plot.png"
        memory_fig_path = "outputs/memory_plot.png"
        energy_fig_path = "outputs/energy_plot.png"
        with plt.rc_context():
            plot_timeline_brokenaxes(scme[0], draw_dependencies, section_start_percent,
                                     percent_shown, plot_data_transfer, fig_path=timeline_fig_path)
        # plot_memory_usage(scme[0].accelerator.memory_manager, fig_path=memory_fig_path)
        # bar_plot_stream_cost_model_evaluations_breakdown([scme], fig_path=energy_fig_path)

    if generate:
        with open(node_hw_performances_path, "rb") as f:
            node_hw_performances = pickle.load(file=f)

        pulp_sdk_path = r"~/new-attempt/pulp-sdk"
        project_path = r"~/new-attempt/pulp-sdk/applications/custom"
        compile_and_run(
            workload, scme[0], node_hw_performances,
            pulp_sdk_path, project_path,
            l1_size=l1_size, l2_size=l2_size,
            simulate=simulate, run=run, plot=plot_profile
        )


if __name__ == "__main__":
    main()

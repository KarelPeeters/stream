import os
import pickle
import random
import re
from dataclasses import dataclass
from typing import Generator

from matplotlib import pyplot as plt
from torch import nn
from zigzag.classes.stages import *

from compiler.main import compile_and_run
from stream.api_util import export_onnx, print_workload_per_core, save_graph
from stream.classes.io.onnx.model import ONNXModelParser
from stream.classes.stages import *
from stream.ext.ima_mapping_stage import ImaIntraCoreMappingState
from stream.inputs.testing.hardware.custom.ima import ima_with_offchip
from stream.test_network import TestNetwork
from stream.visualization.memory_usage import plot_memory_usage
from stream.visualization.plot_scme import bar_plot_stream_cost_model_evaluations_breakdown


class DebugStage(Stage):
    def run(self) -> Generator:
        kwargs = self.kwargs.copy()
        output_path = kwargs["output_path"]

        save_graph(self.kwargs["workload"], f"{output_path}/graph.svg", all_ranges=True)

        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        for i, (cme, extra_info) in enumerate(sub_stage.run()):
            yield cme, extra_info


def get_hardware_performance_stream(hardware, workload, mapping, CN_define_mode, hint_loops, node_hw_performances_path, output_path):
    # Initialize the logger
    import logging as _logging
    _logging_level = _logging.INFO
    # _logging_format = '%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
    _logging_format = '%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
    _logging.basicConfig(level=_logging_level,
                         format=_logging_format)

    mainstage = MainStage([  # Initializes the MainStage as entry point
        # AcceleratorParserStage,  # Parses the accelerator

        # StreamONNXModelParserStage,  # Parses the ONNX Model into the workload
        # UserDefinedModelParserStage,  # Parses the user-defined Model into the workload

        GenerateCNWorkloadHybridStage,

        DebugStage,
        ImaIntraCoreMappingState,
        InterCoreMappingStage,
    ],

        accelerator=hardware,  # required by AcceleratorParserStage
        # workload_path=workload,  # required by ModelParserStage
        workload=workload,  # required by ModelParserStage
        mapping_path=mapping,  # required by ModelParserStage
        loma_lpf_limit=6,  # required by LomaStage
        nb_ga_individuals=4,  # number of individuals in each genetic algorithm generation
        nb_ga_generations=16,  # number of genetic algorithm generations
        node_hw_performances_path=node_hw_performances_path,
        # saved node_hw_performances to skip re-computation
        plot_hof=True,  # Save schedule and memory usage plot of each individual in the Genetic Algorithm hall of fame
        plot_file_name=True,
        plot_full_schedule=True,
        plot_data_transfer=True,
        cn_define_mode=CN_define_mode,
        hint_loops=hint_loops,
        scheduler_candidate_selection='memory',
        output_path=output_path,
    )

    # Launch the MainStage
    answers = mainstage.run()
    return answers


@dataclass
class Setup:
    # TODO divide L2 size by the number of cores?
    # TODO higher level: start properly using L2, maybe just as another global cache external to the cores?
    l1_size: int
    l2_size: int

    ima_width: int
    ima_height: int
    cores: int

    network: nn.Module

    hint_loops: list[tuple[str, int]]


@dataclass
class SetupResult:
    predicted_latency: float
    actual_latency: float


def run_setup(setup: Setup, output_path: str):
    os.makedirs(output_path, exist_ok=True)
    random.seed(0xdeadbeef)

    CN_define_mode = 1

    # TODO use weight_size=4 at some point
    # TODO does weight_size even matter any more?
    accelerator = ima_with_offchip(
        core_count=setup.cores,
        width=setup.ima_width, height=setup.ima_height,
        weight_size=4,
        l1_bits=setup.l1_size * 8, l2_bits=setup.l2_size * 8
    )
    onnx_path = f"{output_path}/inputs/network.onnx"

    mapping = 'inputs.testing.mapping.testing_mapping'

    network = setup.network
    export_onnx(network, onnx_path)

    onnx_model_parser = ONNXModelParser(onnx_path, mapping, accelerator)
    onnx_model_parser.mapping = {'default': {'core_allocation': list(range(setup.cores))}}
    onnx_model_parser.run()

    _onnx_model = onnx_model_parser.get_onnx_model()
    workload = onnx_model_parser.get_workload()

    hw_name = accelerator.name.split(".")[-1]
    wl_name = re.split(r"[/.]", onnx_path)[-1]
    experiment_id = f"{hw_name}-{wl_name}-CNmode_{CN_define_mode}-hintloop_{str(setup.hint_loops)}"
    node_hw_cost_pkl_name = f'saved_CN_HW_cost-{experiment_id}'

    node_hw_performances_path = f"{output_path}/{node_hw_cost_pkl_name}.pickle"

    if os.path.exists(node_hw_performances_path):
        os.remove(node_hw_performances_path)

    print("Running stream")
    scme, _ = get_hardware_performance_stream(
        accelerator,
        workload,
        mapping,
        CN_define_mode,
        setup.hint_loops,
        node_hw_performances_path,
        output_path
    )

    from stream.visualization.schedule import plot_timeline_brokenaxes

    # TODO clean up plotting bools and blocking
    # TODO why does simulate false cause out-of-bounds RAM errors?
    generate = True
    simulate = False
    run = True
    plot_stream = True
    plot_profile = True

    print_workload_per_core(scme[0])

    if plot_stream:
        draw_dependencies = True
        plot_data_transfer = True
        section_start_percent = (0,)
        percent_shown = (100,)
        timeline_fig_path = f"{output_path}/schedule_plot.png"
        memory_fig_path = f"{output_path}/memory_plot.png"
        energy_fig_path = f"{output_path}/energy_plot.png"
        with plt.rc_context():
            plot_timeline_brokenaxes(scme[0], draw_dependencies, section_start_percent,
                                     percent_shown, plot_data_transfer, fig_path=timeline_fig_path)
        plot_memory_usage(scme[0], fig_path=memory_fig_path)
        bar_plot_stream_cost_model_evaluations_breakdown([scme], fig_path=energy_fig_path)

    if generate:
        with open(node_hw_performances_path, "rb") as f:
            node_hw_performances = pickle.load(file=f)

        pulp_sdk_path = r"~/new-attempt/pulp-sdk"
        project_path = r"~/new-attempt/pulp-sdk/applications/custom"
        profile = compile_and_run(
            onnx_path, scme[0], node_hw_performances,
            pulp_sdk_path, project_path,
            l1_size=setup.l1_size, l2_size=setup.l2_size,
            simulate=simulate, run=run, plot=plot_profile,
            output_path=f"{output_path}/",
        )

        return SetupResult(
            predicted_latency=scme[0].latency,
            actual_latency=profile.latency,
        )


def main():
    # resnet18_section = ConvNetwork(depth=8, n=1, c=32, s=64)
    resnet18_section = TestNetwork()

    setup = Setup(
        l1_size=0x00100000,
        l2_size=0x60000000,
        ima_width=256,
        ima_height=256,
        cores=1,
        network=resnet18_section,
        hint_loops=[('OY', 4)],
    )

    setup.cores = 4
    run_setup(setup, "outputs/resnet18_single")

    # pred_latency = []
    # actual_latency = []
    # max_cores = 2
    # core_values = list(range(1, max_cores + 1))

    # for cores in core_values:
    #     print(f"Running resnet18 split cores={cores}")
    #     setup.cores = cores
    #     setup.hint_loops = [('OY', 4)]
    #     result = run_setup(setup, f"outputs/resnet18_split_{cores}")
    #
    #     pred_latency.append(result.predicted_latency)
    #     actual_latency.append(result.actual_latency)

    # plt.figure()
    # plt.plot(core_values, pred_latency, label="Predicted")
    # plt.plot(core_values, actual_latency, label="Actual")
    # plt.xlabel("Number of cores")
    # plt.ylabel("Latency (cycles)")
    # plt.legend()
    # plt.show()
    # plt.savefig("outputs/resnet18_split.png")


if __name__ == "__main__":
    main()

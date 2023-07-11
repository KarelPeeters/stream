import json
import os
import pickle
import random
import re
from dataclasses import dataclass
from typing import Generator, List

from matplotlib import pyplot as plt
from torch import nn
from zigzag.classes.stages import *

from compiler.main import compile_and_run, CollectedInfo
from stream.api_util import export_onnx, print_workload_per_core, save_graph
from stream.classes.io.onnx.model import ONNXModelParser
from stream.classes.stages import *
from stream.ext.ima_mapping_stage import ImaIntraCoreMappingState
from stream.inputs.testing.hardware.custom.ima import ima_with_offchip
from stream.test_network import TestNetwork
from stream.visualization.memory_usage import plot_memory_usage
from stream.visualization.plot_scme import bar_plot_stream_cost_model_evaluations_breakdown
from stream.visualization.schedule import visualize_timeline_plotly


class DebugStage(Stage):
    def run(self) -> Generator:
        kwargs = self.kwargs.copy()
        output_path = kwargs["output_path"]

        save_graph(self.kwargs["workload"], f"{output_path}/graph.svg", all_ranges=True)

        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        for i, (cme, extra_info) in enumerate(sub_stage.run()):
            yield cme, extra_info


def get_hardware_performance_stream(hardware, workload, mapping, CN_define_mode, hint_loops, node_hw_performances_path,
                                    output_path):
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
        nb_ga_individuals=64,  # number of individuals in each genetic algorithm generation
        nb_ga_generations=64,  # number of genetic algorithm generations
        node_hw_performances_path=node_hw_performances_path,
        # saved node_hw_performances to skip re-computation
        plot_hof=True,  # Save schedule and memory usage plot of each individual in the Genetic Algorithm hall of fame
        plot_file_name=True,
        plot_full_schedule=True,
        plot_data_transfer=True,
        cn_define_mode=CN_define_mode,
        hint_loops=hint_loops,
        scheduler_candidate_selection='latency',
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
    predicted_peak_mem: List[int]
    info: CollectedInfo


def run_setup(setup: Setup, output_path: str):
    result = run_setup_inner(setup, output_path)

    plt.close("all")

    if result is not None:
        with open(f"{output_path}/results.txt", "w") as f:
            d = {
                "pred_latency": result.predicted_latency,
                "actual_latency": result.info.profile.latency,
            }
            json.dump(d, f)

    print(result)
    with open(f"{output_path}/log.txt", "w") as f:
        print(result, file=f)

    return result


def run_setup_inner(setup: Setup, output_path: str):
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

    # mapping = 'inputs.testing.mapping.testing_mapping'

    network = setup.network
    export_onnx(network, onnx_path)

    onnx_model_parser = ONNXModelParser(onnx_path, None, accelerator)
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

    force_stagger_cores = False
    os.environ["FORCE_STAGGER_CORES"] = str(force_stagger_cores)

    print("Running stream")
    scme, _ = get_hardware_performance_stream(
        accelerator,
        workload,
        None,
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

    predicted_peak_mem = None
    if plot_stream:
        draw_dependencies = True
        plot_data_transfer = True
        draw_communication = True
        section_start_percent = (0,)
        percent_shown = (100,)
        timeline_fig_path = f"{output_path}/schedule_plot.png"
        timeline_fig_path_plotly = f"{output_path}/schedule_plot.html"
        memory_fig_path = f"{output_path}/memory_plot.png"
        energy_fig_path = f"{output_path}/energy_plot.png"
        with plt.rc_context():
            plot_timeline_brokenaxes(scme[0], draw_dependencies, section_start_percent,
                                     percent_shown, plot_data_transfer, fig_path=timeline_fig_path)
        predicted_peak_mem = plot_memory_usage(scme[0], fig_path=memory_fig_path)
        bar_plot_stream_cost_model_evaluations_breakdown([scme], fig_path=energy_fig_path)
        visualize_timeline_plotly(scme[0], draw_dependencies, draw_communication, fig_path=timeline_fig_path_plotly)

    if generate:
        with open(node_hw_performances_path, "rb") as f:
            node_hw_performances = pickle.load(file=f)

        pulp_sdk_path = r"~/new-attempt/pulp-sdk"
        project_path = r"~/new-attempt/pulp-sdk/applications/custom"
        info = compile_and_run(
            onnx_path, scme[0], node_hw_performances,
            pulp_sdk_path, project_path,
            l1_size=setup.l1_size, l2_size=setup.l2_size,
            simulate=simulate, run=run, plot=plot_profile,
            output_path=f"{output_path}/",
        )

        return SetupResult(
            predicted_latency=scme[0].latency,
            predicted_peak_mem=predicted_peak_mem,
            info=info,
        )


def basic_setup(cores: int, hint_loops, network):
    return Setup(
        l1_size=0x00100000,
        l2_size=0x60000000,
        ima_width=256,
        ima_height=256 * 4,
        cores=cores,
        network=network,
        hint_loops=hint_loops,
    )


def main():
    network = TestNetwork()
    setup = basic_setup(
        cores=2,
        hint_loops=[],
        network=network
    )
    run_setup(setup, "outputs/api_test")


if __name__ == "__main__":
    main()

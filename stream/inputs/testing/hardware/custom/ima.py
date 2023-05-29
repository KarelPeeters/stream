from networkx import DiGraph
from zigzag.classes.hardware.architecture.accelerator import Accelerator
from zigzag.classes.hardware.architecture.core import Core
from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance
from zigzag.classes.hardware.architecture.operational_array import MultiplierArray
from zigzag.classes.hardware.architecture.operational_unit import Multiplier

from stream.classes.hardware.architecture.accelerator import Accelerator
from stream.classes.hardware.architecture.communication_link import CommunicationLink


def basic_memory_instance(
        name: str, size: int,
        r_bw: int, w_bw: int,
        r_port: int, w_port: int, rw_port: int,
) -> MemoryInstance:
    return MemoryInstance(
        name=name, size=size,
        r_bw=r_bw, w_bw=w_bw,
        r_port=r_port, w_port=w_port, rw_port=rw_port,
        # latency isn't actually used anywhere yet
        latency=1,
        # granularities also aren't used anywhere yet
        min_r_granularity=None, min_w_granularity=None,
        # we don't are about these (yet)
        area=0, r_cost=0, w_cost=0,
        # used for auto cost extraction, we don't use this
        mem_type="sram", auto_cost_extraction=False
    )


# TODO ensure that everything refers to the digital clock

def get_memory_hierarchy(multiplier_array, width: int, height: int, weight_bits: int, l1_size_bits: int, l2_size_bits: int):
    # accelerator registers
    reg_input = MemoryInstance(
        name="reg_input", size=8, r_bw=8, w_bw=8, r_cost=0, w_cost=0, area=0,
        r_port=1, w_port=1, rw_port=0, latency=1
    )

    reg_weight = MemoryInstance(
        name="reg_weight", size=weight_bits, r_bw=weight_bits, w_bw=weight_bits, r_cost=0, w_cost=0, area=0,
        r_port=1, w_port=1, rw_port=0, latency=1
    )

    reg_output = MemoryInstance(
        name="reg_output", size=8, r_bw=8, w_bw=8, r_cost=0, w_cost=0, area=0,
        r_port=2, w_port=2, rw_port=0, latency=1
    )

    # # trick to make weight loading extremely slow
    # #   * bandwidth: 4 weights per cycle, like the real accelerator
    # #   * size just large enough to fit lower level
    # factor = 1 / 8
    # weight_bottleneck = basic_memory_instance(
    #     "weight_bottleneck", size=width * height * weight_size,
    #     w_bw=factor * weight_size, r_bw=factor * weight_size, w_port=1, r_port=1, rw_port=0,
    # )

    # weight_inf = basic_memory_instance(
    #     "weight_inf", size=width * height * weight_size,
    #     w_bw=1, r_bw=
    # )

    # 8*64 bytes per cycle
    real_l12_bw = 64 * 8 * 8

    # actual memories
    l1 = basic_memory_instance(
        name="L1", size=0x00100000, r_bw=real_l12_bw, w_bw=real_l12_bw, r_port=2, w_port=2, rw_port=0,
    )
    l2 = basic_memory_instance(
        name="L2", size=0x60000000, r_bw=real_l12_bw, w_bw=real_l12_bw, r_port=2, w_port=2, rw_port=0,
    )

    memory_hierarchy_graph = MemoryHierarchy(operational_array=multiplier_array)

    # memory_hierarchy_graph.add_memory(
    #     memory_instance=reg_input,
    #     operands=('I1',),
    #     port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),
    #     served_dimensions={(1, 0)}
    # )
    # memory_hierarchy_graph.add_memory(
    #     memory_instance=reg_weight,
    #     operands=('I2',),
    #     port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),
    #     served_dimensions={(0, 0)}
    # )
    # memory_hierarchy_graph.add_memory(
    #     memory_instance=reg_output,
    #     operands=('O',),
    #     operands=('O',),
    #     port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': 'w_port_2', 'th': 'r_port_2'},),
    #     served_dimensions={(0, 1)}
    # )

    # memory_hierarchy_graph.add_memory(
    #     memory_instance=weight_bottleneck,
    #     operands=('I2',),
    #     port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),
    #     served_dimensions='all',
    # )

    memory_hierarchy_graph.add_memory(
        memory_instance=l1,
        operands=('I1', 'I2', 'O'),
        port_alloc=(
            {'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},
            {'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},
            {'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': 'w_port_2', 'th': 'r_port_2'},
        ),
        served_dimensions='all'
    )

    memory_hierarchy_graph.add_memory(
        memory_instance=l2,
        operands=('I1', 'I2', 'O'),
        port_alloc=(
            {'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},
            {'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},
            {'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': 'w_port_2', 'th': 'r_port_2'},
        ),
        served_dimensions='all'
    )

    # visualize_memory_hierarchy_graph(memory_hierarchy_graph)

    return memory_hierarchy_graph


def get_operational_array(width: int, height: int):
    """ Multiplier array variables """
    multiplier_input_precision = [8, 8]
    multiplier_energy = 0.04
    multiplier_area = 1
    dimensions = {'D1': width, 'D2': height}  # {'D1': ('K', 16), 'D2': ('C', 16)}
    # dimensions = {'D1': 16, 'D2': 16, 'D3': 4, 'D4': 4}

    multiplier = Multiplier(multiplier_input_precision, multiplier_energy, multiplier_area)
    multiplier_array = MultiplierArray(multiplier, dimensions)

    return multiplier_array


def get_dataflows(width: int, height: int):
    # TODO figure out the syntax for combined unrolling here
    # return None
    # return [{'D1': ('K', 16), 'D2': ('C', 16), 'D3': ('OX', 4), 'D4': ('FX', 3)}]
    # return []
    return [{'D1': ('K', height), 'D2': ('C', width)}]


def get_ima_core(id, width: int, height: int, weight_bits: int, l1_bits: int, l2_bits: int):
    operational_array = get_operational_array(width, height)
    memory_hierarchy = get_memory_hierarchy(operational_array, width, height, weight_bits, l1_bits, l2_bits)
    dataflows = get_dataflows(width, height)
    core = Core(id, operational_array, memory_hierarchy, dataflows)
    return core


def get_dummy_multiplier_array():
    """ Multiplier array variables """
    multiplier_input_precision = [8, 8]
    multiplier_energy = float('inf')
    multiplier_area = 0
    dimensions = {'D1': 1, 'D2': 1}
    multiplier = Multiplier(multiplier_input_precision, multiplier_energy, multiplier_area)
    multiplier_array = MultiplierArray(multiplier, dimensions)

    return multiplier_array


def get_offchip_core(id):
    multiplier_array = get_dummy_multiplier_array()

    # actual size 0x00800000, but just use "infinity" here for now
    # TODO model that strided 2D reads are a lot slower
    l3 = basic_memory_instance(
        name="l3", size=10000000000,
        r_bw=8, w_bw=8, r_port=0, w_port=0, rw_port=1,
    )

    memory_hierarchy_graph = MemoryHierarchy(operational_array=multiplier_array)
    memory_hierarchy_graph.add_memory(
        memory_instance=l3,
        operands=('I1', 'I2', 'O'),
        port_alloc=(
            {'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': None, 'th': None},
            {'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': None, 'th': None},
            {'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': 'rw_port_1', 'th': 'rw_port_1'},
        ),
        served_dimensions='all'
    )

    core = Core(id, multiplier_array, memory_hierarchy_graph)
    return core


def get_cores_graph(cores, offchip_core, unit_energy_cost: float):
    edges = []

    # TODO also limit to in-core bandwidth
    offchip_read_bandwidth = offchip_core.mem_r_bw_dict['O'][0]
    offchip_write_bandwidth = offchip_core.mem_w_bw_dict['O'][0]

    # if the offchip core has only one port
    if len(offchip_core.mem_hierarchy_dict['O'][0].port_list) == 1:
        to_offchip_link = CommunicationLink(offchip_core, "Any", offchip_write_bandwidth, unit_energy_cost,
                                            bidirectional=True)
        from_offchip_link = to_offchip_link

    # if the offchip core has more than one port
    else:
        to_offchip_link = CommunicationLink("Any", offchip_core, offchip_write_bandwidth, unit_energy_cost)
        from_offchip_link = CommunicationLink(offchip_core, "Any", offchip_read_bandwidth, unit_energy_cost)
    if not isinstance(offchip_core, Core):
        raise ValueError("The given offchip_core is not a Core object.")
    for core in cores:
        edges.append((core, offchip_core, {'cl': to_offchip_link}))
        edges.append((offchip_core, core, {'cl': from_offchip_link}))

    # Build the graph using the constructed list of edges
    H = DiGraph(edges)

    return H


def ima_with_offchip(core_count: int, width: int, height: int, weight_size: int, l1_bits: int, l2_bits: int):
    # TODO divide l2 by the number of cores?
    cores = [get_ima_core(i, width, height, weight_size, l1_bits, l2_bits) for i in range(core_count)]

    offchip_core_id = core_count

    offchip_core = get_offchip_core(id=offchip_core_id)
    cores_graph = get_cores_graph(cores, offchip_core, 0)

    accelerator = Accelerator("Testing-2-core-with-offchip", cores_graph, offchip_core_id=offchip_core_id)
    return accelerator


if __name__ == "__main__":
    print(ima_with_offchip(2, 1024, 1024, 8, 1024*1024*8, 1024*1024*8))

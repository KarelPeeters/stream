import os.path

import networkx
import onnx
import torch
from onnx import shape_inference

from stream.classes.cost_model.cost_model import StreamCostModelEvaluation
from stream.classes.hardware.architecture.communication_link import CommunicationLink
from stream.classes.workload.computation_node import ComputationNode


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


def export_onnx(network, path):
    print("Exporting ONNX model")

    input = network.example_input()
    _ = network(input)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.onnx.export(network, input, path)

    model = onnx.load_model(path)
    model_shaped = shape_inference.infer_shapes(model)
    onnx.save_model(model_shaped, path)


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

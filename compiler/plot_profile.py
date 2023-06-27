import re
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt

pattern_profile = re.compile(r"^== profile == (\d+) == ([^=]+) == ([^=]+) == ([^=]+)$")
pattern_core = re.compile(r"^.*core_(\d+)$")


@dataclass
class CollectedProfile:
    core_slices: dict
    core_last_time: dict
    names: dict

    latency: int


def parse_profile_info(stdout: str) -> CollectedProfile:
    core_slices = {}
    key_last_time = {}
    names = dict()  # deterministic set

    for line in stdout.splitlines():
        if m := pattern_profile.match(line):
            time, core, kind, name = m.groups()
            time = int(time)
            key = (core, name)
            names[name] = None

            if kind == "start":
                if key in key_last_time:
                    print(f"warning: name already started {core, name}")
                key_last_time[key] = time

            if kind == "end":
                if key in key_last_time:
                    core_slices.setdefault(core, []).append((name, key_last_time[key], time))
                    del key_last_time[key]
                else:
                    print(f"warning: name not started {name}")

    for key in key_last_time:
        print(f"warning: {key} didn't end")

    # sort cores
    core_slices = dict(sorted(core_slices.items()))

    core_last_time = dict()
    for core, slices in core_slices.items():
        if m := pattern_core.match(core):
            core_index = int(m.group(1))
            last_time = max(end for (name, start, end) in slices)
            core_last_time[core_index] = last_time
    latency = max(core_last_time.values())

    # print for debugging
    for core, slices in core_slices.items():
        print(f"{core}: {slices}")

    info = CollectedProfile(core_slices, core_last_time, names, latency)
    return info


def plot_profile(info: CollectedProfile, output_path: str, block: bool = True):
    if len(info.core_slices) == 0:
        return

    # plot slices
    fig, axes = plt.subplots(nrows=len(info.core_slices), sharex="all", squeeze=False)
    axes = axes.squeeze(1)

    cmap = matplotlib.colormaps["tab10"]
    name_colors = {name: cmap(i) for i, name in enumerate(info.names)}

    for i, (core, slices) in enumerate(info.core_slices.items()):
        used_names = {}

        ax = axes[i]
        # ax.set_ylabel(name, rotation=0, labelpad=50)
        ax.set_ylabel(core)
        for (name, start, end) in slices:
            used_names[name] = ()
            color = name_colors[name]
            ax.axvspan(start, end, facecolor=color, alpha=1.0)

        ax.legend(handles=[matplotlib.patches.Patch(color=name_colors[name], label=name) for name in used_names])

    # fig.tight_layout()
    plt.savefig(output_path)
    plt.show(block=block)


def main():
    # path = r"\\wsl.localhost\Ubuntu\home\karel\new-attempt\pulp-sdk\applications\custom\output.txt"

    # path = "../stream/outputs/stdout.txt"
    # output_path = "../stream/outputs/profile.png"

    path = r"\\wsl.localhost\Ubuntu\home\karel\new-attempt\pulp-sdk\applications\custom\log.txt"
    output_path = "profile.png"

    with open(path, "r") as f:
        data = f.read()
    profile = parse_profile_info(data)
    plot_profile(profile, output_path)


if __name__ == '__main__':
    main()

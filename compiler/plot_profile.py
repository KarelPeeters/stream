import re

import matplotlib
import matplotlib.pyplot as plt

pattern_profile = re.compile(r"^== profile == (\d+) == ([^=]+) == ([^=]+) == ([^=]+)$")


def plot_profile(stdout: str, block: bool = True):
    core_slices = {}
    key_last_time = {}
    names = set()

    for line in stdout.splitlines():
        if m := pattern_profile.match(line):
            time, core, kind, name = m.groups()
            time = int(time)
            key = (core, name)
            names.add(name)

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

    for core, slices in core_slices.items():
        print(f"{core}: {slices}")

    if len(core_slices) == 0:
        return

    # plot slices
    fig, axes = plt.subplots(nrows=len(core_slices), sharex="all", squeeze=False)
    axes = axes.squeeze(1)

    cmap = matplotlib.colormaps["tab10"]
    name_colors = {name: cmap(i) for i, name in enumerate(names)}

    for i, (core, slices) in enumerate(core_slices.items()):
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
    plt.show(block=block)


def main():
    # path = r"\\wsl.localhost\Ubuntu\home\karel\new-attempt\pulp-sdk\applications\custom\output.txt"
    path = "../stream/outputs/stdout.txt"
    with open(path, "r") as f:
        data = f.read()
    plot_profile(data)


if __name__ == '__main__':
    main()

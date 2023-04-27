import re

import matplotlib.pyplot as plt

pattern_profile = re.compile(r"^== profile == (\d+): (.*)$")
pattern_start = re.compile(r"^start (.*)$")
pattern_end = re.compile(r"^end (.*)$")
pattern_layer = re.compile(r"^layer(\d+)$")


def plot_profile(stdout: str, block: bool=True):
    item_last_time = {}
    item_slices = {}
    curr_layer = None

    for line in stdout.splitlines():
        if m := pattern_profile.match(line):
            time = int(m.group(1))
            message = m.group(2)

            if m_start := pattern_start.match(message):
                item = m_start.group(1)
                # print(f"start {time} {item}")
                if item in item_last_time:
                    print(f"warning: item already started {item}")
                item_last_time[item] = time

                if m_layer := pattern_layer.match(item):
                    curr_layer = int(m_layer.group(1))

            if m_end := pattern_end.match(message):
                item = m_end.group(1)
                # print(f"end {time} {item}")

                if item in item_last_time:
                    if item not in item_slices:
                        item_slices[item] = []
                    item_slices[item].append((item_last_time[item], time, curr_layer))
                    del item_last_time[item]
                else:
                    print(f"warning: item not started {item}")

                if m_layer := pattern_layer.match(item):
                    if m_layer.group(1) != str(curr_layer):
                        print(f"warning: layer end mismatch, expected {curr_layer} got {item}")
                    curr_layer = None

    for item in item_last_time:
        print(f"warning: item {item} not ended")

    for item, slices in item_slices.items():
        print(f"{item}: {slices}")

    # plot slices
    fig, axes = plt.subplots(nrows=len(item_slices), sharex="all", squeeze=False)
    axes = axes.squeeze(1)

    layer_colors = ["r", "g", "b"]

    for i, (item, slices) in enumerate(item_slices.items()):
        ax = axes[i]
        # ax.set_ylabel(item, rotation=0, labelpad=50)
        ax.set_ylabel(item)
        for (start, end, layer) in slices:
            color = layer_colors[layer] if layer is not None else "k"
            ax.axvspan(start, end, facecolor=color, alpha=0.5)

    # fig.tight_layout()
    plt.show(block=block)


def main():
    path = r"\\wsl.localhost\Ubuntu\home\karel\new-attempt\pulp-sdk\applications\custom\output.txt"
    with open(path, "r") as f:
        data = f.read()
    plot_profile(data)


if __name__ == '__main__':
    main()

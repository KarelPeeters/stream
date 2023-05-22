import json
import re

import numpy as np
from matplotlib import pyplot as plt

# "502x32x184 ima total 47713 plot 41979 job 5562"
pattern = re.compile(r"^(\d+)x(\d+)x(\d+) ima total (\d+) plot (\d+) job (\d+)$")


def main():
    # path = r"\\wsl.localhost\Ubuntu\home\karel\new-attempt\pulp-sdk\applications\custom\log_ima_profile.txt"
    path = r"\\wsl.localhost\Ubuntu\home\karel\new-attempt\pulp-sdk\applications\custom\log.txt"

    data = []

    with open(path, "r") as f:
        for line in f:
            if m := pattern.match(line):
                data.append([int(m.group(i)) for i in range(1, 7)])

    data = np.array(data)
    b = data[:, 0]
    k = data[:, 1]
    c = data[:, 2]
    cycles_total = data[:, 3]
    cycles_plot = data[:, 4]
    cycles_job = data[:, 5]

    plot_cycles_per_area = np.nanmean(cycles_plot / (k * c))
    print("Plot cycles per area: ", plot_cycles_per_area)

    kc_values = np.linspace(0, 256 * 256)
    plt.plot(k * c, cycles_plot, '.')
    plt.plot(kc_values, 7.1 * kc_values)
    plt.show()

    print("Min job cycles: ", np.min(cycles_job))

    # for 16x16: 41 + 9*b

    # plt.figure()
    # plt.plot(k, cycles_job - 41, '.')
    # plt.show(block=False)
    # plt.figure()
    # plt.plot(c, cycles_job - 41, '.')
    # plt.show()

    # k_uniq = np.unique(k)
    # c_uniq = np.unique(c)
    #
    # map = np.zeros((len(k_uniq), len(c_uniq)))
    # for i in range(len(data)):
    #     map[np.argwhere(k_uniq == k[i]), np.argwhere(c_uniq == c[i])] = cycles_job[i]
    #
    # print(map.shape)
    # plt.imshow(map, cmap='hot', interpolation="nearest")
    # plt.show()

    b = np.unique(b)
    assert len(b) == 1
    b = b[0]

    b_factors = {}
    for i in range(len(data)):
        b_factors[f"{k[i]}x{c[i]}"] = round((cycles_job[i] - 41) / b)

    json_data = {
        "plot_per_area": plot_cycles_per_area,
        "job_offset": 41,
        "job_per_b": b_factors,
    }
    with open("ima_profile.json", "w") as f:
        json.dump(json_data, f)

    for factor in np.unique(((cycles_job - 41) / b + 0.5).astype(int)):
        mask = ((cycles_job - 41) / b + 0.5).astype(int) == factor,
        plt.scatter(k[mask], c[mask], s=128, marker="s", label=f"{factor}")
    plt.legend()
    plt.show()

    # labels = []
    # values = []
    # bytes_per_step = 4*16
    #
    # for curr_k in np.unique(k):
    #     for curr_c in np.unique(c):
    #         mask = (k == curr_k) & (c == curr_c)
    #
    #         if mask.sum() == 0:
    #             continue
    #
    #         factor = (cycles_job[mask] - 41) / b[mask]
    #         # plt.plot(b[mask], factor, '.', label=f"{curr_k}x{curr_c}")
    #
    #         labels.append(curr_k // bytes_per_step + curr_c // bytes_per_step)
    #         values.append(np.nanmean(factor))
    #
    # plt.plot(labels, values, '.')
    # plt.show()

    # print(f"{curr_k}x{curr_c}: {np.nanmean(factor)}")

    # for curr_c in [32, 64, 128, 256]:
    #     mask = c == curr_c
    #     plt.plot(k[mask], cycles_job[mask], '.', label=f"c={curr_c}")
    #
    #     k_space = np.arange(np.min(k[mask]), np.max(k[mask]))
    #
    #     plt.plot(k_space, pred_job_cycles(32, k_space, curr_c), label=f"pred c={curr_c}")

    # plt.legend()
    # plt.show()


def pred_job_cycles(b, k, c):
    ports = 16
    bytes_per_port = 4

    bytes_per_step = ports * bytes_per_port

    input_steps = c // bytes_per_step
    output_steps = k // bytes_per_step

    pred_job_cycles = b * (16 - 2 + input_steps + output_steps)
    return pred_job_cycles


if __name__ == '__main__':
    main()

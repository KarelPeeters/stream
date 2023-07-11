from matplotlib import pyplot as plt

from stream.api import basic_setup, run_setup
from stream.test_network import SplitConvNetwork


def main_full(cores: int):
    network = SplitConvNetwork(
        depth=8, width=cores,
        n=1, c=16, s=32
    )
    setup = basic_setup(
        cores=cores,
        hint_loops=[],
        network=network
    )

    run_setup(setup, f"outputs/split_{cores}")


def main_pipeline():
    # TODO disable IMA weight loading
    network = SplitConvNetwork(depth=4, width=1, n=1, c=16, s=64)
    setup = basic_setup(4, [("OY", 8)], network)
    run_setup(setup, "outputs/pipeline")


def main_trivial():
    network = SplitConvNetwork(depth=1, width=1, n=1, c=16, s=64)
    setup = basic_setup(1, [], network)
    run_setup(setup, "outputs/trivial")


def main_trivial2():
    network = SplitConvNetwork(depth=2, width=1, n=1, c=16, s=64)
    setup = basic_setup(1, [], network)
    run_setup(setup, "outputs/trivial2")


def main_single():
    network = SplitConvNetwork(
        depth=8, width=1,
        n=1, c=16, s=32
    )
    setup = basic_setup(
        cores=1,
        hint_loops=[],
        network=network
    )

    run_setup(setup, "outputs/single")


def latency_mismatch_for_cores(setup, name: str, max_cores: int):
    pred_latency = []
    actual_latency = []
    core_values = list(range(1, max_cores + 1))

    for cores in core_values:
        print(f"Running split cores={cores}")
        setup.cores = cores
        result = run_setup(setup, f"outputs/{name}_{cores}")

        pred_latency.append(result.predicted_latency)
        actual_latency.append(result.info.profile.latency)

    plt.figure()
    plt.plot(core_values, pred_latency, label="Predicted")
    plt.plot(core_values, actual_latency, label="Actual")
    plt.xlabel("Number of cores")
    plt.ylabel("Latency (cycles)")
    plt.legend()
    plt.show()
    plt.savefig(f"outputs/{name}_latency.png")


def main_latency_mismatch():
    network = SplitConvNetwork(8, 1, 1, 16, 64)
    setup = basic_setup(
        cores=2,
        hint_loops=[],
        network=network,
    )
    latency_mismatch_for_cores(setup, "mismatch", 8)


def main_cn_splitting():
    size = 64

    network = SplitConvNetwork(8, 1, 1, 16, size)

    splits = [1, 2, 4, 8, 16, 32]
    pred_latencies = []
    actual_latencies = []

    for split in splits:
        setup = basic_setup(
            cores=4,
            hint_loops=[("OY", split)],
            network=network,
        )

        result = run_setup(setup, f"outputs/cn_{split}")
        pred_latencies.append(result.predicted_latency)
        actual_latencies.append(result.actual_latency)

    plt.plot(splits, pred_latencies, label="predicted latency")
    plt.plot(splits, actual_latencies, label="actual latency")
    plt.savefig("outputs/cn_split_latency.png")
    plt.show()


def main():
    # main_single()

    # for i in range(1, 8+1):
    #     main_full(i)

    main_full(4)

    # main_pipeline()

    # main_trivial()
    # main_trivial2()

    # main_latency_mismatch()

    # main_cn_splitting()


if __name__ == "__main__":
    main()

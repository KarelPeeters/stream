import numpy as np

from compiler.codegen import array_to_str, DataType


def ima_sum(value, weight, adc_low: int, adc_high: int):
    ps = 0
    for (idx, w) in enumerate(weight):
        ps += ((weight[idx] * 1.0) / 7) * ((value[idx] * 1.0) / 127)
    if ps > adc_high:
        ps = adc_high
    if ps < -adc_low:
        ps = -adc_low
    ps_int = round(ps * (127 / (adc_high + adc_low)))
    if ps_int > 127:
        ps_int = 127
    if ps_int < -127:
        ps_int = -127
    return ps_int


def random_ima_weight(shape):
    return np.random.randint(-7, 7, shape)


def random_ima_input(shape):
    return np.random.randint(-128, 127, shape)


def main():
    n = 16
    height = 64
    width = 64
    adc_low = 16
    adc_high = 16

    np.random.seed(0)

    x = random_ima_input((n, height))
    w = random_ima_weight((height, width))

    y = np.zeros((n, width), dtype=np.int32)
    for i in range(n):
        for j in range(width):
            y[i, j] = ima_sum(x[i, :], w[:, j], adc_low, adc_high)

    with open("layer_weights.h", "w") as f:
        print(f"#define LAYER_N {n}", file=f)
        print(f"#define LAYER_HEIGHT {height}", file=f)
        print(f"#define LAYER_WIDTH {width}", file=f)

        print(f"#define DATA_I {array_to_str(x.flatten(), DataType.Int8)}", file=f)
        print(f"#define DATA_W {array_to_str(w.flatten(), DataType.Int4)}", file=f)
        print(f"#define DATA_O {array_to_str(y.flatten(), DataType.Int8)}", file=f)

    print(y)


if __name__ == '__main__':
    main()

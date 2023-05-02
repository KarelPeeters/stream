import numpy as np

from compiler.codegen import array_to_str, DataType


def ima_sum(input, weight, adc_low: int, adc_high: int):
    assert np.all((-128 <= input) & (input < 127))
    assert np.all((-7 <= weight) & (weight < 7))

    # matches operations in GVSOC simulator exactly
    value = np.float32(0.0)
    DAC_PRECISION = 8
    STOR_DWIDTH = 4
    ADC_PRECISION = 8

    for i in range(len(weight)):
        input_float = np.float32(input[i]) / np.float32((1 << (DAC_PRECISION - 1)) - 1)
        weight_float = np.float32(weight[i])
        value += input_float * weight_float / np.float32((1 << (STOR_DWIDTH - 1)) - 1)

    if value > adc_high:
        value = np.float32(adc_high)
    elif value < -adc_low:
        value = np.float32(adc_low)

    # (float) (value * (((1 << (ADC_PRECISION - 1)) - 1))) / ((this->job->adc_high + this->job->adc_low))
    value = value * np.float32((1 << (ADC_PRECISION - 1)) - 1) / np.float32(adc_high + adc_low)

    if value >= ((1 << (ADC_PRECISION - 1)) - 1):
        value = ((1 << (ADC_PRECISION - 1)) - 1)
    elif value <= -((1 << (ADC_PRECISION - 1)) - 1):
        value = -((1 << (ADC_PRECISION - 1)) - 1)

    if value >= 0:
        if value - int(value) >= 0.5:
            return int(value) + 1
        return int(value)
    else:
        if value - int(value) <= -0.5:
            return int(value) - 1
        return int(value)


def random_ima_weight(shape):
    return np.random.randint(-7, 7, shape)


def random_ima_input(shape):
    return np.random.randint(-128, 127, shape)


def ima_matmul(input, weight):
    (b, c1) = input.shape
    (k, c) = weight.shape
    assert c1 == c, f"Shape mismatch: input={input.shape}, weight={weight.shape}"

    output = np.zeros((b, k), dtype=np.int32)
    for bi in range(b):
        for ki in range(k):
            output[bi, ki] = ima_sum(input[bi, :], weight[ki, :], 16, 16)

    return output


def main():
    n = 512
    size = 32
    adc_low = 16
    adc_high = 16

    np.random.seed(0)

    x = random_ima_input((n, size))
    w1 = random_ima_weight((size, size))
    w2 = random_ima_weight((size, size))

    y1 = ima_matmul(x, w1)
    y2 = ima_matmul(y1, w2)

    with open("layer_weights.h", "w") as f:
        print(f"#define LAYER_N {n}", file=f)
        print(f"#define LAYER_SIZE {size}", file=f)

        print(f"#define DATA_I {array_to_str(x.flatten(), DataType.Int8)}", file=f)
        print(f"#define DATA_W1 {array_to_str(w1.flatten(), DataType.Int4)}", file=f)
        print(f"#define DATA_W2 {array_to_str(w2.flatten(), DataType.Int4)}", file=f)
        print(f"#define DATA_O {array_to_str(y2.flatten(), DataType.Int8)}", file=f)

    print(y1)


if __name__ == '__main__':
    main()

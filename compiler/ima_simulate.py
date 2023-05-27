import matplotlib.pyplot as plt
import numpy as np


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
    return np.clip(np.random.standard_normal(shape) * 50, -126, 126).astype(int)
    # return np.random.randint(-128, 127, shape)


def ima_matmul(input, weight):
    (b, c1) = input.shape
    (k, c) = weight.shape
    assert c1 == c, f"Shape mismatch: input={input.shape}, weight={weight.shape}"

    output = np.zeros((b, k), dtype=np.int32)
    for bi in range(b):
        for ki in range(k):
            output[bi, ki] = ima_sum(input[bi, :], weight[ki, :], 16, 16)

    return output


def ima_conv(input, weight):
    (b, c1, ih, iw) = input.shape
    (k, c, fh, fw) = weight.shape

    assert c1 == c, f"Shape mismatch: input={input.shape}, weight={weight.shape}"
    assert fh == 3 and fw == 3, f"Only 3x3 filters supported, got {fh}x{fw}"

    output = np.zeros((b, k, ih, iw), dtype=np.int32)

    input_padded = np.zeros((b, c1, ih + 2, iw + 2), dtype=np.int32)
    input_padded[:, :, 1:-1, 1:-1] = input

    for bi in range(b):
        for ki in range(k):
            for y in range(ih):
                for x in range(iw):
                    output[bi, ki, y, x] = ima_sum(
                        input_padded[bi, :, y:y + 3, x:x + 3].flatten(),
                        weight[ki, :, :, :].flatten(),
                        16, 16
                    )

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

    bins = np.linspace(-128, 128, 256)

    ax1 = plt.subplot(311)
    plt.hist(x.flatten(), bins=bins)
    ax2 = plt.subplot(312, sharex=ax1)
    plt.hist(y1.flatten(), bins=bins)
    _ = plt.subplot(313, sharex=ax2)
    plt.hist(y2.flatten(), bins=bins)

    plt.show()


if __name__ == '__main__':
    main()

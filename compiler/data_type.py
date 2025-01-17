import enum

import numpy as np


class DataType(enum.Enum):
    Int4 = enum.auto()
    Int8 = enum.auto()

    @property
    def size_bytes(self):
        if self == DataType.Int4:
            return 0.5
        elif self == DataType.Int8:
            return 1

        raise ValueError(f"unknown data type {self}")


def array_to_str(data, data_type: DataType):
    data_bytes = array_to_bytes(data, data_type)
    result = "{" + ", ".join([f"0x{v:02x}" for v in data_bytes]) + "}"
    return result


def array_to_bytes(data, data_type: DataType):
    data = np.array(data)
    assert len(data.shape) == 1

    if data_type == DataType.Int4:
        data_flat = ((data[1::2] << 4) & 0xf0) | (data[::2] & 0x0f)
    elif data_type == DataType.Int8:
        data_flat = data & 0xff
    else:
        raise ValueError(f"unknown data type {data_type}")

    data_byte = data_flat.astype(np.uint8)
    return bytes(data_byte)

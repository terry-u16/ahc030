import base64
import struct

import numpy as np


def pack(value: np.float64) -> bytes:
    return struct.pack("<d", value)


def pack_vec(vec: np.ndarray) -> str:
    stream = b""
    n = len(vec)

    for i in range(n):
        stream += pack(vec[i])

    s = base64.b64encode(stream).decode("utf-8")
    return s

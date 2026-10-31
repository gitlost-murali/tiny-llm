import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    # Supplied for Day 1; a manual implementation is an optional bonus exercise.
    return mx.softmax(x, axis=axis)


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    matmul = mx.matmul(x, w.swapaxes(-2, -1))
    if bias is not None:
        matmul = matmul + bias
    return matmul

def silu(x: mx.array) -> mx.array:
    pass

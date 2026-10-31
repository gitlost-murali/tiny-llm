import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    if scale is None:
        scale = 1/(mx.sqrt(mx.array(query.shape[-1])))
    scaled_QK = mx.matmul(query, key.swapaxes(-2, -1)) * scale

    if mask is not None:
        scaled_QK = mx.add(scaled_QK, mask)
    normalized_attention_weights = softmax(scaled_QK, axis=-1)
    final_output = mx.matmul(normalized_attention_weights, value)
    return final_output

class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.small_head_size = self.hidden_size//self.num_heads

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        N, L, _ = query.shape
        query = mx.matmul(query, self.wq.swapaxes(-1,-2)).reshape(N,L,self.num_heads, self.small_head_size).swapaxes(-2,-3)
        key = mx.matmul(key, self.wk.swapaxes(-1,-2)).reshape(N,L,self.num_heads, self.small_head_size).swapaxes(-2,-3)
        value = mx.matmul(value, self.wv.swapaxes(-1,-2)).reshape(N,L,self.num_heads, self.small_head_size).swapaxes(-2,-3)
        output = scaled_dot_product_attention_simple(query, key, value, scale = None, mask=mask)
        output = mx.matmul(output.swapaxes(-2,-3).reshape(N,L, self.num_heads*self.small_head_size), self.wo.swapaxes(-1,-2))
        return output



def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def paged_attention(
    query: mx.array,
    key_pages: mx.array,
    value_pages: mx.array,
    block_table: mx.array,
    context_lens: mx.array,
    page_size: int,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    """Attend to paged K/V storage without reconstructing a dense cache.

    Week 3 Day 4 owns the correctness-first decode and long-prefill paths for
    both float32 and BF16. Day 5 optimizes the same public boundary.
    """
    pass

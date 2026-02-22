import torch
from torch import Tensor
from jaxtyping import Float, Int, Bool
from cs336_basics.utils import linear, scaled_dot_product_attention, rope, rmsnorm, swiglu, embedding

def multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    # 注意这里d_in就是d_model
    sequence_length: int = in_features.shape[-2]
    q: Float[Tensor, " ... sequence_length d_k"]= linear(q_proj_weight, in_features)
    k: Float[Tensor, " ... sequence_length d_k"] = linear(k_proj_weight, in_features)
    v: Float[Tensor, " ... sequence_length d_v"] = linear(v_proj_weight, in_features)
    # d_k就是num_heads*head_dim 所以我们需要进行拆分
    # q.shape[:-1]就是把最后一维去掉，保留剩下的
    # *用于解包，这是Python语法；最后的那个-1表示自动计算最后一维
    q = q.view(*q.shape[:-1], num_heads, -1) # ... sequence_length num_heads head_dim
    k = k.view(*k.shape[:-1], num_heads, -1) # ... sequence_length num_heads head_dim
    v = v.view(*v.shape[:-1], num_heads, -1) # ... sequence_length num_heads head_dim
    # 转置
    q = q.transpose(-3, -2) # ... num_heads sequence_length head_dim
    k = k.transpose(-3, -2) # ... num_heads sequence_length head_dim
    v = v.transpose(-3, -2) # ... num_heads sequence_length head_dim_v
    # 转置以后 倒数第三维是num_heads 倒数第二维是sequence_length
    # 生成一个下三角矩阵
    mask = torch.tril(torch.ones(sequence_length, sequence_length)) # sequence_length sequence_length
    # 神人scaled_dot_product_attention
    res: Float[Tensor, " ... num_heads sequence_length head_dim_v"] = scaled_dot_product_attention(q, k, v, mask)
    # 合并多个头 也就是把num_heads和head_dim_v合并，变成d_v
    # 先转置
    res = res.transpose(-3, -2) # ... sequence_length num_heads head_dim_v
    # 再view 用-1自动合并
    # 千万注意一下 要加contiguous() 因为transpose以后内存不连续
    res = res.contiguous().view(*res.shape[:-2], -1) # ... sequence_length d_v
    return linear(o_proj_weight, res)

def multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    # 注意这里d_in就是d_model
    sequence_length: int = in_features.shape[-2]
    head_dim: int = q_proj_weight.shape[-2] // num_heads
    q: Float[Tensor, " ... sequence_length d_k"]= linear(q_proj_weight, in_features)
    k: Float[Tensor, " ... sequence_length d_k"] = linear(k_proj_weight, in_features)
    v: Float[Tensor, " ... sequence_length d_v"] = linear(v_proj_weight, in_features)
    # d_k就是num_heads*head_dim 所以我们需要进行拆分
    # q.shape[:-1]就是把最后一维去掉，保留剩下的
    # *用于解包，这是Python语法；最后的那个-1表示自动计算最后一维
    q = q.view(*q.shape[:-1], num_heads, -1) # ... sequence_length num_heads head_dim
    k = k.view(*k.shape[:-1], num_heads, -1) # ... sequence_length num_heads head_dim
    v = v.view(*v.shape[:-1], num_heads, -1) # ... sequence_length num_heads head_dim
    # 转置
    q = q.transpose(-3, -2) # ... num_heads sequence_length head_dim
    k = k.transpose(-3, -2) # ... num_heads sequence_length head_dim
    v = v.transpose(-3, -2) # ... num_heads sequence_length head_dim_v
    # 转置以后 倒数第三维是num_heads 倒数第二维是sequence_length
    # 很好玩的啦 可以转的啦（rope）
    if token_positions is None:
        token_positions = torch.arange(sequence_length)
    q = rope(head_dim, theta, q, token_positions)
    k = rope(head_dim, theta, k, token_positions)
    # 生成一个下三角矩阵
    mask = torch.tril(torch.ones(sequence_length, sequence_length)) # sequence_length sequence_length
    # 神人scaled_dot_product_attention
    res: Float[Tensor, " ... num_heads sequence_length head_dim_v"] = scaled_dot_product_attention(q, k, v, mask)
    # 合并多个头 也就是把num_heads和head_dim_v合并，变成d_v
    # 先转置
    res = res.transpose(-3, -2) # ... sequence_length num_heads head_dim_v
    # 再view 用-1自动合并
    # 千万注意一下 要加contiguous() 因为transpose以后内存不连续
    res = res.contiguous().view(*res.shape[:-2], -1) # ... sequence_length d_v
    return linear(o_proj_weight, res)

def transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    eps: float = 1e-5
    res: Float[Tensor, " batch sequence_length d_model"] = rmsnorm(eps, weights["ln1.weight"], in_features)
    res = multihead_self_attention_with_rope(d_model, num_heads, max_seq_len, theta, weights["attn.q_proj.weight"], weights["attn.k_proj.weight"], weights["attn.v_proj.weight"], weights["attn.output_proj.weight"], res)
    res += in_features
    # FFN就是Feed-Forward Network，前馈网络
    res2: Float[Tensor, " batch sequence_length d_model"] = rmsnorm(eps, weights["ln2.weight"], res)
    res2 = swiglu(weights["ffn.w1.weight"], weights["ffn.w2.weight"], weights["ffn.w3.weight"], res2)
    res += res2
    return res

def transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    eps: float = 1e-5
    embedded_tokens: Float[Tensor, " batch_size sequence_length d_model"] = embedding(weights["token_embeddings.weight"], in_indices)
    for i in range(num_layers):
        # 提取每一层的weight字典
        layer_weights: dict[str, Tensor] = {k.removeprefix(f"layers.{i}."): v for k, v in weights.items() if k.startswith(f"layers.{i}")}
        embedded_tokens = transformer_block(d_model, num_heads, d_ff, context_length, rope_theta, layer_weights, embedded_tokens)
    res: Float[Tensor, " batch_size sequence_length d_model"] = rmsnorm(eps, weights["ln_final.weight"], embedded_tokens)
    res: Float[Tensor, " batch_size sequence_length vocab_size"] = linear(weights["lm_head.weight"], res)
    return res

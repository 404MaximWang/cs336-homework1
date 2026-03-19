import torch
from torch import Tensor
from jaxtyping import Float, Int, Bool
import numpy.typing as npt
import numpy as np

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # 对所有key行行进行“减去最大值”操作
    x_optimized = x - torch.max(x, dim = dim, keepdim = True).values
    # 对优化后的x进行softmax
    return torch.exp(x_optimized) / torch.sum(torch.exp(x_optimized), dim = dim, keepdim = True)

def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"],
) -> Float[Tensor, ""]:
    # softmax_inputs = softmax(inputs, dim = -1)
    # 以下是比较好理解的版本
    # total_loss: float = 0
    # for i in range(inputs.shape[0]):
    #     total_loss += -torch.log(softmax_inputs[i, targets[i]])
    # return total_loss / inputs.shape[0]
    # 以下是向量化版本，非常优雅，但是需要一定理解成本。它直接把需要的概率拿出来计算后取平均
    # return -torch.log(softmax_inputs[torch.arange(inputs.shape[0]), targets]).mean()
    # 疑似出了一点问题 重新写过 我们需要防止一个情况：exp(-1000) = 0，然后log(0) = -inf。
    x_optimized = inputs - torch.max(inputs, dim = -1, keepdim = True).values
    log_softmax = x_optimized - torch.logsumexp(x_optimized, dim = -1, keepdim = True)
    return -log_softmax[torch.arange(inputs.shape[0], device=inputs.device), targets].mean()

def embedding(
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    return weights[token_ids]

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def linear(
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    # 这里我们明确一下矩阵长什么样
    # in_features的纵向长度是词数量；横向长度是输入维度，也就是d_in
    # weights的纵向长度是输出维度，也就是d_out；横向长度是输入维度，也就是d_in
    # 最后做乘法时，我们用in_features乘weights的转置
    return in_features @ weights.T

def rmsnorm(
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    # 这个函数是用来稳定数值的
    rms = torch.sqrt(torch.mean(in_features ** 2, dim = -1, keepdim = True) + eps)
    # 注意一下乘除法的广播规则是从右边对齐。这里d_model维度和d_model维度对齐，可以广播乘法。
    # weights中的数会乘上x/rms(x)对应的那一列
    return in_features / rms * weights

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    # 假设这里的QKV是二维的，那么QKV矩阵的一行就是一个词；也就是说，行数就是词数，列数就是维度
    # 但是QKV不一定是二维，所以接下来的scores计算中我们不能使用转置
    # 这里没有直接传d_k。我们用Q.shape[-1]来表示它
    # 还有queries = keys = values = sequence_length
    scores: Float[Tensor, " ... queries keys"] = Q @ K.transpose(-2, -1) / Q.shape[-1] ** 0.5
    # 这里设的-inf，在softmax处理后会变成0
    # mask中的0会被丢掉
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    # softmax
    weights: Float[Tensor, " ... queries keys"] = softmax(scores, dim = -1)
    # 从V张量中拿东西 注意这里的乘法只乘最后两个维度 前面的不管
    return weights @ V
    
def swiglu(
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    m1: Float[Tensor, " ... d_ff"] = linear(w1_weight, in_features)
    m2: Float[Tensor, " ... d_ff"] = linear(w3_weight, in_features)
    # 逐元素相乘
    m3: Float[Tensor, " ... d_ff"] = silu(m1) * m2
    return linear(w2_weight, m3)

def rope(
    d_k: int,
    theta: float,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    # 用神奇方法一次取出所有奇数维度和偶数维度
    x: Float[Tensor, " ... sequence_length d_k / 2"] = in_query_or_key[..., 0::2]
    y: Float[Tensor, " ... sequence_length d_k / 2"] = in_query_or_key[..., 1::2]
    # 注意一下unsqueeze(-1)
    # 举个例子 a = torch.tensor([10, 20, 30])
    # a.unsqueeze(-1)
    # [[10], [20], [30]]
    # torch.arrange的用法和Python range用法相似，都是左界（闭），右界（开），步长
    device = token_positions.device
    # 确保 arange 生成在同一设备上
    angles = token_positions.unsqueeze(-1) / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
    # token_positions.unsqueeze(-1) 的形状是 (..., sequence_length, 1)
    # torch.arange(0, d_k, 2).float() / d_k 的形状是 (d_k / 2,)
    # 它们相除后，形状是 (..., sequence_length, d_k / 2) 这里token_positions.unsqueeze(-1)的最后一维被拓展
    # 很神奇吧？
    x_out: Float[Tensor, " ... sequence_length d_k / 2"] = torch.cos(angles) * x - torch.sin(angles) * y
    y_out: Float[Tensor, " ... sequence_length d_k / 2"] = torch.sin(angles) * x + torch.cos(angles) * y
    return torch.stack([x_out, y_out], dim=-1).flatten(-2)
    # stack用于配对；flatten(-2)用于把末尾两维压成一维

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    # 拿数据 注意上界不要写错 不要多减1
    random_start = np.random.randint(0, len(dataset) - context_length, size = batch_size)
    inputs: npt.NDArray = np.array([dataset[i : i + context_length] for i in random_start]) # (batch_size, context_length)
    labels: npt.NDArray = np.array([dataset[i + 1 : i + context_length + 1] for i in random_start]) # (batch_size, context_length)
    inputs_tensor: Int[Tensor, "batch_size context_length"] = torch.tensor(inputs, device = device)
    labels_tensor: Int[Tensor, "batch_size context_length"] = torch.tensor(labels, device = device)
    return inputs_tensor, labels_tensor

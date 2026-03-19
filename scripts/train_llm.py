from dataclasses import dataclass, asdict
from cs336_basics.utils import cross_entropy, get_batch
from cs336_basics.model import transformer_lm
from cs336_basics.optimizer import AdamW
from cs336_basics.bpe_tokenizer_optimized_3 import Tokenizer
from cs336_basics.pretokenization import find_chunk_boundaries
import torch
import numpy as np
import os
import argparse
import multiprocessing
import itertools
from torch import Tensor
from jaxtyping import Float, Int

MODEL_CONFIGS = {
    "vocab_size": 10000,
    "context_length": 256,
    "d_model": 512,
    "d_ff": 1344,
    "num_layers": 4,
    "num_heads": 16,
    "rope_theta": 10000.0
}

def init_weights(config: dict) -> dict[str, Tensor]:
    d_model = config["d_model"]
    d_ff = config["d_ff"]
    vocab_size = config["vocab_size"]
    num_layers = config["num_layers"]
    
    # 线性
    def w_init(rows, cols):
        return (torch.randn(rows, cols) * 0.02).cuda().requires_grad_(True)
    
    # 归一化
    def ln_init(size):
        return torch.ones(size).cuda().requires_grad_(True)
    weights = {}
    
    # Embeddings & LM Head
    weights["token_embeddings.weight"] = w_init(vocab_size, d_model)
    weights["lm_head.weight"] = w_init(vocab_size, d_model)
    
    # Transformer Layers
    for i in range(num_layers):
        # Attention
        weights[f"layers.{i}.attn.q_proj.weight"] = w_init(d_model, d_model)
        weights[f"layers.{i}.attn.k_proj.weight"] = w_init(d_model, d_model)
        weights[f"layers.{i}.attn.v_proj.weight"] = w_init(d_model, d_model)
        weights[f"layers.{i}.attn.output_proj.weight"] = w_init(d_model, d_model)
        
        # FFN
        weights[f"layers.{i}.ffn.w1.weight"] = w_init(d_ff, d_model)
        weights[f"layers.{i}.ffn.w2.weight"] = w_init(d_model, d_ff)
        weights[f"layers.{i}.ffn.w3.weight"] = w_init(d_ff, d_model)
        
        # RMSNorms
        weights[f"layers.{i}.ln1.weight"] = ln_init(d_model)
        weights[f"layers.{i}.ln2.weight"] = ln_init(d_model)
    
    # Final Norm
    weights["ln_final.weight"] = ln_init(d_model)
    
    return weights

def generate_shit(batch_size: int, context_length: int, vocab_size: int) -> Int[Tensor, "batch_size context_length"]:
    return torch.randint(0, vocab_size, (batch_size, context_length)).cuda()

def _encode_worker(file_path, start, end, tokenizer_path):
    from cs336_basics.bpe_tokenizer_optimized_3 import Tokenizer
    tokenizer = Tokenizer.load(tokenizer_path)
    with open(file_path, 'rb') as f:
        f.seek(start)
        chunk_data = f.read(end - start)
    text = chunk_data.decode('utf-8', errors='replace')
    return tokenizer.encode(text)

def load_checkpoint(path: str) -> tuple[dict[str, Tensor], dict]:
    print(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location="cpu")
    # 把权重搬到 GPU 并开启梯度
    weights = {k: v.cuda().requires_grad_(True) for k, v in checkpoint['weights'].items()}
    return weights, checkpoint['config']

def train_llm(num_steps: int, corpus_path: str, tokenizer_path: str, save_path: str, checkpoint_path: str = None):
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer.load(tokenizer_path)
    
    # encoding结果缓存
    dataset_cache_path = corpus_path + ".npy"
    if os.path.exists(dataset_cache_path):
        print(f"Found cached dataset at {dataset_cache_path}, loading instantly...")
        dataset = np.load(dataset_cache_path)
    else:
        print(f"Encoding corpus from {corpus_path} (this might take a while)...")
        # 将生成的的列表用np.array转成int32并保存
        ids = parallel_encode(tokenizer, corpus_path, tokenizer_path)
        dataset = np.array(ids, dtype=np.int32)
        print(f"Dataset ready! Total tokens: {len(dataset)}")
        print(f"Saving dataset cache to {dataset_cache_path}...")
        np.save(dataset_cache_path, dataset)

    # 查已有文件 已加入文件大小检查
    if checkpoint_path and os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 0:
        weights, config = load_checkpoint(checkpoint_path)
    else:
        config = MODEL_CONFIGS
        weights = init_weights(config)
    optimizer = AdamW(params = weights.values(), lr = 1e-4, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0.1)
    for step in range(num_steps):
        # 喂饭
        input, target = get_batch(dataset, batch_size = 32, context_length = config["context_length"], device = "cuda")
        # 显式转换为long
        input, target = input.long(), target.long()        
        logits: Float[Tensor, "batch_size context_length vocab_score"] = transformer_lm(weights = weights, in_indices = input, **config)    
        loss: Float[Tensor, ""] = cross_entropy(logits.view(-1, config["vocab_size"]), target.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")        
        # 每100步存档一次
        if step > 0 and step % 100 == 0:
            ckpt_name = f"checkpoint_step_{step}.pt"
            torch.save({'weights': weights, 'config': config}, ckpt_name)
            print(f"Periodic checkpoint saved: {ckpt_name}")
    # 保存
    print(f"Training finished! Saving weights to {save_path}...")
    torch.save({'weights': weights, 'config': config}, save_path)
    print("All saved! Mission Accomplished.")

def parallel_encode(tokenizer, file_path, tokenizer_path):
    num_cpu = multiprocessing.cpu_count()
    # 获取边界
    split_bytes: bytes = tokenizer.special_tokens[0].encode('utf-8')
    boundaries: list[int] = find_chunk_boundaries(file_path, split_bytes)
    # 构建任务，只传路径
    tasks = [(file_path, start, end, tokenizer_path) 
                for start, end in zip(boundaries[:-1], boundaries[1:])]
    print(f"Parallel Encoding... dispatching to {num_cpu} cores.")
    with multiprocessing.Pool(processes = num_cpu) as pool:
        # 使用 starmap 把 tasks 里的元组分拆传给 _encode_worker
        results = pool.starmap(_encode_worker, tasks)
    return list(itertools.chain.from_iterable(results))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--input", type=str, required=True, help="Path to text corpus")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json")
    parser.add_argument("--output", type=str, default="model_weights.pt", help="Where to save the final weights")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for resuming")
    args = parser.parse_args()
    
    train_llm(num_steps=args.steps, 
              corpus_path=args.input, 
              tokenizer_path=args.tokenizer,
              save_path=args.output,
              checkpoint_path=args.checkpoint)

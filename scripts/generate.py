import torch
import argparse
from cs336_basics.model import transformer_lm
from cs336_basics.bpe_tokenizer_optimized_3 import Tokenizer
from torch import Tensor
from jaxtyping import Float, Int

# This is an AI-generated script.

def load_checkpoint(path: str) -> tuple[dict[str, Tensor], dict]:
    print(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location="cuda")
    weights = {k: v.cuda().requires_grad_(False) for k, v in checkpoint['weights'].items()}
    return weights, checkpoint['config']

@torch.no_grad()
def generate(
    prompt: str,
    weights: dict[str, Tensor],
    config: dict,
    tokenizer: Tokenizer,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 10,
) -> str:
    # encode the prompt
    ids = tokenizer.encode(prompt)
    context_length = config["context_length"]
    
    print(f"\n[Prompt]: {prompt}", end="")
    
    # recursive generating
    for _ in range(max_new_tokens):
        # crop the context to prevent it from exceeding the model's context_length
        context = ids[-context_length:]
        in_tensor: Int[Tensor, "1 context_len"] = torch.tensor([context], dtype=torch.long, device="cuda")
        
        logits: Float[Tensor, "1 context_len vocab_size"] = transformer_lm(
            weights=weights, 
            in_indices=in_tensor, 
            **config
        )
        next_token_logits = logits[0, -1, :] / temperature
        # Top-K
        if top_k is not None:
            v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            next_token_logits[next_token_logits < v[-1]] = -float('Inf')
        
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        
        ids.append(next_token)
        
        # 实时逐字打印出刚生成的词。先缓存一个字再打印。
        print(tokenizer.decode([next_token]), end="", flush=True)

    print("\n")
    return tokenizer.decode(ids)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Once upon a time, there was a little girl named Lily. She loved to", help="Starting prompt")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to your trained .pt model")
    parser.add_argument("--tokenizer", type=str, default="/home/maxim/tiny_stories/ts1.json", help="Path to your tokenizer")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Creativity vs. strictness (0.1 ~ 1.5)")
    parser.add_argument("--top_k", type=int, default=10, help="Top K sampling parameter")
    args = parser.parse_args()

    tokenizer = Tokenizer.load(args.tokenizer)
    weights, config = load_checkpoint(args.checkpoint)

    # 启动生成
    generate(
        prompt=args.prompt,
        weights=weights,
        config=config,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k
    )

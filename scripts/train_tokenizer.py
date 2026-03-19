import os
import sys
import argparse

# Add the project root to sys.path
sys.path.append(os.getcwd())

from cs336_basics.bpe_tokenizer_optimized_3 import Tokenizer

def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument("--input", type=str, default="data/TinyStoriesV2-GPT4-valid.txt", 
                        help="Path to training text (default: validation set for debug)")
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--save_path", type=str, default="tokenizer.json")
    parser.add_argument("--max_bytes", type=int, default=None, help="Limit input size")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return
        
    print(f"Training tokenizer on {args.input}...")
    print(f"Vocab size: {args.vocab_size}")
    
    # Initialize tokenizer with special tokens
    tokenizer = Tokenizer(special_tokens=["<|endoftext|>"])
    
    # Train
    tokenizer.train(args.input, args.vocab_size, max_bytes=args.max_bytes)
    
    # Save
    print(f"Saving tokenizer to {args.save_path}...")
    tokenizer.save(args.save_path)
    print("Done!")

if __name__ == "__main__":
    main()

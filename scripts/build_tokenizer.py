from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import os
import argparse

def build_tokenizer(data_path, vocab_size, save_path):
    print(f"Building Tokenizer from {data_path}...")
    
    # Initialize BPE Tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    )
    
    # Train
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return

    tokenizer.train(files=[data_path], trainer=trainer)
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    print(f"Success! Tokenizer saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to training data text file")
    parser.add_argument("--vocab_size", type=int, default=1024, help="Vocabulary size")
    parser.add_argument("--save_path", type=str, default="tokenizers/hp.json", help="Where to save the json")
    
    args = parser.parse_args()
    build_tokenizer(args.data, args.vocab_size, args.save_path)
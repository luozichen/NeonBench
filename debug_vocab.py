import sys
sys.path.append('.')
from tokenizers import Tokenizer
from models.neon238 import Neon238

tokenizer = Tokenizer.from_file('tokenizers/wiki103_tok5.json')
vocab_size = tokenizer.get_vocab_size()

c = {'vocab_size': vocab_size, 'd_model': 272, 'n_layers': 4, 'n_head': 4, 'd_ff': 1072, 'block_size': 1024, 'device': 'cpu'}
m = Neon238(c)

print(f"Vocab size from tokenizer: {vocab_size}")

total_grad_p = sum(p.numel() for p in m.parameters() if p.requires_grad)
emb_numel = m.token_emb.weight.numel()
c_val = total_grad_p - emb_numel

print(f"Total Grad Params: {total_grad_p:,}")
print(f"Embedding Numel: {emb_numel:,}")
print(f"Reported Non-Embed: {c_val:,}")

import sys
sys.path.append('.')

from train import get_config
from models.neon233 import Neon233
from models.neon238 import Neon238

def test_model(name, ModelClass):
    c = get_config(name)
    c['vocab_size'] = 32000
    m = ModelClass(c)
    
    total = sum(p.numel() for p in m.parameters())
    grad = sum(p.numel() for p in m.parameters() if p.requires_grad)
    emb = m.token_emb.weight.numel()
    
    print(f"--- {name} ---")
    print(f"Total Params: {total:,}")
    print(f"Grad Params:  {grad:,}")
    print(f"Embed Size:   {emb:,}")
    print(f"Train Parity Reports: {grad - emb:,}")

test_model('neon233', Neon233)
test_model('neon238', Neon238)

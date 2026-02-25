import torch
import torch.nn as nn
import sys
import os
from tokenizers import Tokenizer

sys.path.append(os.getcwd())
from train import get_config
from models.neon185 import Neon185

from models.neon230 import Neon230
from models.neon231 import Neon231
from models.neon232 import Neon232
from models.neon233 import Neon233
from models.neon234 import Neon234
from models.neon235 import Neon235
from models.neon236 import Neon236
from models.neon237 import Neon237
from models.neon238 import Neon238
from models.neon239 import Neon239
from models.neon240 import Neon240
from models.neon241 import Neon241
from models.neon242 import Neon242

def count_non_embed(model):
    shared_params = model.token_emb.weight.numel()
    return sum(p.numel() for p in model.parameters()) - shared_params

vocab_size = 8192 # wiki103_tok5
target = 5004528 # neon185 non-embed

def check(name, ModelClass, d_ff):
    cfg = get_config(name)
    cfg.update({
        'vocab_size': vocab_size, 
        'd_ff': d_ff, 
        'n_layers': 4, 
        'd_model': 272,
        'block_size': 256,
        'n_head': 4
    })
    m = ModelClass(cfg)
    p = count_non_embed(m)
    print(f"{name} (d_ff={d_ff}) Non-Embed: {p:,} (Diff: {p - target})")

print(f"Target Non-Embed (Neon185): {target:,}\n")
check("neon230", Neon230, 1170)
check("neon231", Neon231, 834)
check("neon232", Neon232, 1170)
check("neon233", Neon233, 1170)
check("neon234", Neon234, 1170)
check("neon235", Neon235, 1170)
check("neon236", Neon236, 1170)
check("neon237", Neon237, 1170)
check("neon238", Neon238, 1072)
check("neon239", Neon239, 1072)
check("neon240", Neon240, 1072)
check("neon241", Neon241, 1072)
check("neon242", Neon242, 1072)

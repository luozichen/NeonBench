"""Neon030: neon002 (RMSNorm + GELU) scaled to ~3M params via d_ff increase.
Isolates the RMSNorm effect: neon029 (LayerNorm) vs neon030 (RMSNorm) at equal params."""
from models.neon002 import Neon002 as Neon030

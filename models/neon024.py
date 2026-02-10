"""Neon024: Same architecture as Neon023 (8-layer neon016), trained with LayerDrop=0.1.
This is not a separate architecture — it reuses Neon023 with a different training config."""
from models.neon023 import Neon023 as Neon024

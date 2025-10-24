from __future__ import annotations
import os
import tensorflow as tf

def save_model_artifacts(model, out_dir: str):
    os.makedirs(out_dir, exist_ok = True)
    model.save(os.path.join(out_dir, "model.keras"))

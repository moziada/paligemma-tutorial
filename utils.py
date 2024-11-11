from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os
import re

def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # Find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # ... and load them one by one in the tensors dictionary
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load the model's config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    # Create the model using the configuration
    model = PaliGemmaForConditionalGeneration(config).to(device)

    # check if all weights are mappable
    check_unmatched_weights(model, os.path.join(model_path, "model.safetensors.index.json"))

    # Load the state dict of the model
    model.load_state_dict(tensors, strict=False)

    # Tie weights
    model.tie_weights()

    return (model, tokenizer)

def formatIndexing(attr_chain):
    """
    Format a dot-separated attribute chain to include square brackets around numeric parts.
    
    Args:
        attr_chain (str): The attribute chain, e.g., 'language_model.model.layers.10.mlp.gate_proj'.
    
    Returns:
        str: The formatted attribute chain with numeric parts in square brackets, e.g.,
             'language_model.model.layers[10].mlp.gate_proj'.
    """
    # Use regex to find numbers and wrap them with brackets
    formatted = re.sub(r'\.(\d+)', r'[\1]', attr_chain)
    return formatted

def check_unmatched_weights(model, safetensors_json_path):
    with open(safetensors_json_path, 'r') as file:
        data = json.load(file)
    for k in data['weight_map'].keys():
        k = '.'.join(k.split('.')[:-1])    # remove `weight` or `bias` from the end of the chain str
        k = formatIndexing(k)
        i = '.'.join(k.split('.')[:-1])     # all chain except last attribute
        attr = k.split('.')[-1]             # last attribute
        if not hasattr(eval('model.' + i), attr):
            raise Warning(f"Attribute {attr} not found in model {i}")
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

def load_base_model(model_id, use_8bit=False):
    kwargs = {"device_map": "auto"}
    kwargs = {"device_map": "auto"}

    if use_8bit and torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["quantization_config"] = quant_config
    elif use_8bit and not torch.cuda.is_available():
        print("⚠️ Requested 8-bit, but CUDA not available. Falling back to float32 on CPU.")

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    return model, tokenizer
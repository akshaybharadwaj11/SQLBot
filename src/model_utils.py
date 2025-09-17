from transformers import AutoModelForCausalLM, AutoTokenizer

def load_base_model(model_id, use_8bit=False):
    kwargs = {"device_map": "auto"}
    if use_8bit:
        kwargs.update({"load_in_8bit": True})
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    return model, tokenizer
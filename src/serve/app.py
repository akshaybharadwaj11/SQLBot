import argparse
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import uvicorn
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Using NVIDIA CUDA GPU")
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU (slow)")

app = FastAPI()

class Req(BaseModel):
    nl: str
    max_tokens: int = 256

@app.on_event("startup")
def load_model():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(app.state.tokenizer_id)
    kwargs = {"device_map": "auto"}

    if torch.cuda.is_available():
        print("✅ CUDA detected. Loading model in 8-bit mode.")
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["quantization_config"] = quant_config
    else:
        print("⚠️ CUDA not available. Loading model in float32 (CPU mode). "
              "This may be slow for large models.")
    base_model = AutoModelForCausalLM.from_pretrained(
        app.state.base_model, **kwargs
    )
    model = PeftModel.from_pretrained(base_model, app.state.adapter_path)
    model.to(device)
    model.eval()

@app.post("/generate_sql")
def generate(req: Req):
    input_ids = tokenizer(req.nl, return_tensors="pt").input_ids.to(model.device)
    out = model.generate(input_ids, max_new_tokens=req.max_tokens, do_sample=False)
    return {"sql": tokenizer.decode(out[0], skip_special_tokens=True)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--tokenizer_id", default=None)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    app.state.base_model = args.base_model
    app.state.adapter_path = args.adapter_path
    app.state.tokenizer_id = args.tokenizer_id or args.base_model

    uvicorn.run(app, host="0.0.0.0", port=args.port)
import argparse
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

app = FastAPI()

class Req(BaseModel):
    nl: str
    max_tokens: int = 256

@app.on_event("startup")
def load_model():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(app.state.tokenizer_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        app.state.base_model, device_map="auto", load_in_8bit=True
    )
    model = PeftModel.from_pretrained(base_model, app.state.adapter_path)
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

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)
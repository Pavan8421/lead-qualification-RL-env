# scripts/try_adapter.py
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = "outputs/adapters/step-000050"

tok = AutoTokenizer.from_pretrained(ADAPTER)
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER).eval()

prompt = tok.apply_chat_template(
    [{"role": "user", "content": "Lead: enterprise, IT, Director. Action?"}],
    tokenize=False, add_generation_prompt=True,
)
ids = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**ids, max_new_tokens=16, do_sample=False)
print(tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True))
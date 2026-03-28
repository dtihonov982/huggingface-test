import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
print(f"BF16: {torch.cuda.is_bf16_supported()}")

model_name = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="cuda"
)

inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=20)

print(tokenizer.decode(output[0], skip_special_tokens=True))
print(f"VRAM used: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print("ALL CHECKS PASSED")
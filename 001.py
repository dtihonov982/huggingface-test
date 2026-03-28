# test_env.py
import torch

# 1. CUDA & GPU check
print("=== Environment Check ===")
print(f"PyTorch version:  {torch.__version__}")
print(f"CUDA available:   {torch.cuda.is_available()}")
print(f"CUDA version:     {torch.version.cuda}")
print(f"GPU count:        {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name:         {torch.cuda.get_device_name(0)}")
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU memory:       {mem:.1f} GB")
    print(f"BF16 supported:   {torch.cuda.is_bf16_supported()}")

# 2. Quick tensor ops on GPU
print("\n=== GPU Tensor Test ===")
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
y = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
z = x @ y  # matmul
print(f"Matmul result shape: {z.shape}, dtype: {z.dtype}")
print("GPU tensor ops: OK")

# 3. HuggingFace model test (small model, ~500MB)
print("\n=== HuggingFace Model Test ===")
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "microsoft/phi-2"  # small 2.7B model
print(f"Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True,
)

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=20)

result = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Prompt:   {prompt}")
print(f"Output:   {result}")
print(f"\nMem used: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print("\n=== ALL CHECKS PASSED ===")
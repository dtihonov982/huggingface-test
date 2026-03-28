import torch
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
)
pipe.to("cuda")

prompt = "A cat astronaut floating in space, digital art, highly detailed"
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("test_sdxl.png")

print(f"VRAM used: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print("Image saved to test_sdxl.png")
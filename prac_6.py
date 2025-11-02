from diffusers import StableDiffusionPipeline
import torch

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model (use float16 only if GPU is available)
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# Text prompt
prompt = "A futuristic cityscape at sunset with flying cars"

# Generate image
image = pipe(prompt, guidance_scale=7.5).images[0]

# Save and show
image.save("generated.png")
image.show()

print(f"✅ Image generated successfully on {device.upper()}")

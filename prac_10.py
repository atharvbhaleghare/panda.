from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.ao.quantization as quant  # new import to avoid warning

# Load pre-trained GPT-2
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Quantize the model
model_q = quant.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# Encode input
inp = tokenizer("AI can", return_tensors="pt")

# Generate text with better sampling parameters
out = model_q.generate(inp["input_ids"], max_length=60, temperature=0.8, top_p=0.9, repetition_penalty=1.2)

print("\nGenerated Text:\n")
print(tokenizer.decode(out[0], skip_special_tokens=True))
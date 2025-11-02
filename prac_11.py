from transformers import pipeline

# ⿡ Load a pre-trained text generation model
generator = pipeline("text-generation", model="gpt2")

# ⿢ Generate sentences for bias analysis
prompts = [
    "The nurse said that",
    "The engineer said that"
]

# ⿣ Check model outputs
for p in prompts:
    result = generator(p, max_length=15, num_return_sequences=1)
    print(f"\nPrompt: {p}")
    print("Generated:", result[0]['generated_text'])

# ⿤ Simple mitigation: use neutral or balanced prompts
neutral_prompt = "The person said that"
neutral_result = generator(neutral_prompt, max_length=15, num_return_sequences=1)
print("\nAfter mitigation:")
print("Generated:", neutral_result[0]['generated_text'])

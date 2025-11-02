from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import math, torch

# Load small dataset & tokenizer
tok = AutoTokenizer.from_pretrained("distilgpt2")
tok.pad_token = tok.eos_token
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]").map(
    lambda x: tok(x["text"], truncation=True, padding="max_length", max_length=64), batched=True
)
ds = ds.remove_columns(["text"]).train_test_split(test_size=0.1)

# Model + LoRA setup
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
lora = LoraConfig(r=4, lora_alpha=16, target_modules=["c_attn"], task_type="CAUSAL_LM")
model = get_peft_model(model, lora)

# Training
args = TrainingArguments("out", per_device_train_batch_size=4, num_train_epochs=1, logging_steps=10)
trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["test"], tokenizer=tok)
trainer.train()

# Evaluate & Perplexity
loss = trainer.evaluate()["eval_loss"]
print("Perplexity:", math.exp(loss))

# Generate text
prompt = "Artificial intelligence is"
ids = tok(prompt, return_tensors="pt").input_ids
out = model.generate(ids, max_new_tokens=40, do_sample=True, top_p=0.9)
print(tok.decode(out[0], skip_special_tokens=True))

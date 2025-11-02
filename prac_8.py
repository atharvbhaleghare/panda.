# Install required packages (run once)
# pip install transformers datasets evaluate --quiet

from transformers import pipeline
import evaluate

# Load summarization pipeline (pre-trained T5 model)
summarizer = pipeline("summarization", model="t5-small")

# Input text (you can change this)
text = """Artificial Intelligence is transforming industries by automating tasks, 
improving decision-making, and creating new opportunities for innovation."""

# Generate summary
summary = summarizer(text, max_length=30, min_length=5, do_sample=False)[0]['summary_text']

# Evaluate using ROUGE metric
rouge = evaluate.load("rouge")
results = rouge.compute(predictions=[summary], references=[text])

print("Original Text:\n", text)
print("\nGenerated Summary:\n", summary)
print("\nROUGE Evaluation:\n", results)

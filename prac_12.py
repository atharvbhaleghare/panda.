from transformers import pipeline

agent = pipeline("text-generation", model="gpt2")

def ai_agent(name, message):
    response = agent(message, max_length=40, truncation=True)[0]['generated_text']
    print(f"{name}: {response}\n")
    return response

msg = "Hello! What do you think about AI?"
for i in range(1):
    msg = ai_agent("Agent 1", msg)
    msg = ai_agent("Agent 2",msg)
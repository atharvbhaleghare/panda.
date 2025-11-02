# Experiment 9
# NOTE: Requires pip install sentence-transformers torch (for embeddings)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. Document Corpus (Knowledge Base)
documents = [
    "The capital of France is Paris.",
    "The Eiffel Tower is in Paris.",
    "Machine learning is a subfield of AI.",
    "Python is a popular programming language."
]
query = "Where is the Eiffel Tower?"

# 2. Create Embeddings (Simplified Vector DB with S-BERT)
model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = model.encode(documents)
query_embedding = model.encode([query])

# 3. Retrieve relevant documents (FAISS is replaced by cosine similarity)
similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
top_doc_index = np.argmax(similarities)
retrieved_doc = documents[top_doc_index]

# 4. Generate contextually accurate text response (LLM step is conceptual)
print(f"Query: {query}")
print(f"Retrieved Context (from 'Vector DB'): {retrieved_doc}")

# Conceptual LLM step: LLM takes the query and the context to generate the final answer.
final_answer = f"According to the context, '{retrieved_doc}', the answer is related to Paris."
print(f"Generated Answer: {final_answer}")
from gensim.models import Word2Vec
import faiss
import numpy as np

# Small text corpus
sentences = [
    ["ai", "is", "transforming", "the", "world"],
    ["machine", "learning", "is", "a", "part", "of", "ai"],
    ["deep", "learning", "uses", "neural", "networks"],
    ["data", "science", "involves", "statistics", "and", "programming"]
]

# Train Word2Vec model
model = Word2Vec(sentences, vector_size=50, min_count=1, epochs=100)

# Prepare FAISS index
words = list(model.wv.key_to_index.keys())
vectors = np.array([model.wv[w] for w in words]).astype('float32')

index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

# Query similar words
query = model.wv["ai"].reshape(1, -1)
_, idx = index.search(query, 3)

print("Query: ai")
print("Most similar words:", [words[i] for i in idx[0]])

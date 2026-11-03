# PS16: Scaled dot-product self-attention with a causal mask (the core of GPT / Transformers), numpy only
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)

words = ["the", "cat", "sat", "on", "mat"]
T, d = len(words), 8
X = rng.normal(size=(T, d))                         # pretend these are the word embeddings
Wq, Wk, Wv = (rng.normal(size=(d, d)) for _ in range(3))

Q, K, V = X @ Wq, X @ Wk, X @ Wv                    # queries, keys, values
scores = Q @ K.T / np.sqrt(d)                       # how much each word looks at every other word
mask = np.triu(np.ones((T, T)), 1).astype(bool)     # causal mask: a word cannot see the FUTURE
scores[mask] = -1e9
att = np.exp(scores - scores.max(1, keepdims=True)); att /= att.sum(1, keepdims=True)   # softmax
out = att @ V                                       # weighted mix of values

print("Attention weights (rows = current word, cols = words it looks at):")
print(np.round(att, 2))
print("Output shape:", out.shape)

plt.imshow(att, cmap="Blues"); plt.xticks(range(T), words); plt.yticks(range(T), words)
plt.colorbar(); plt.title("causal self-attention"); plt.show()

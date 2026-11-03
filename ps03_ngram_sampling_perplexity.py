# PS3: Bigram language model + temperature / top-k / top-p sampling + perplexity (numpy only)
import numpy as np
np.random.seed(0)

corpus = ("artificial intelligence is the study of machines that learn . "
          "generative ai is a part of artificial intelligence that can create text images and music . "
          "language models learn to predict the next word . "
          "a language model can generate text from a prompt .").split()
vocab = sorted(set(corpus)); V = len(vocab)
w2i = {w: i for i, w in enumerate(vocab)}

# count bigrams (+ tiny smoothing so no probability is exactly 0)
counts = np.full((V, V), 0.1)
for a, b in zip(corpus, corpus[1:]):
    counts[w2i[a], w2i[b]] += 1
P = counts / counts.sum(1, keepdims=True)   # P[prev] = distribution of next word

def sample_next(prev, temp=1.0, top_k=0, top_p=1.0):
    logits = np.log(P[prev]) / temp          # temperature: <1 sharper, >1 flatter
    probs = np.exp(logits - logits.max()); probs /= probs.sum()
    if top_k > 0:                            # keep only k best words
        probs[np.argsort(probs)[::-1][top_k:]] = 0
        probs /= probs.sum()
    if top_p < 1.0:                          # keep smallest set whose prob adds up to p
        order = np.argsort(probs)[::-1]
        cut = np.searchsorted(np.cumsum(probs[order]), top_p) + 1
        keep = np.zeros_like(probs); keep[order[:cut]] = probs[order[:cut]]
        probs = keep / keep.sum()
    return np.random.choice(V, p=probs)

def generate(prompt, n=12, **kw):
    out = prompt.split()
    for _ in range(n):
        out.append(vocab[sample_next(w2i[out[-1]], **kw)])
    return " ".join(out)

for kw in [dict(temp=0.3), dict(temp=1.0), dict(temp=2.0), dict(temp=1.0, top_k=3), dict(temp=1.0, top_p=0.8)]:
    print(kw, "->", generate("language", **kw))

# perplexity = exp(average negative log-likelihood); lower = better model
nll = -np.mean([np.log(P[w2i[a], w2i[b]]) for a, b in zip(corpus, corpus[1:])])
print("\nLoss (NLL):", round(nll, 3), " Perplexity:", round(float(np.exp(nll)), 3))

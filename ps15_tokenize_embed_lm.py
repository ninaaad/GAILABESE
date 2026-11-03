# PS15: Tokenization vs embedding + tiny neural language model; loss & perplexity vs epochs (numpy)
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)

text = ("artificial intelligence is changing the world . generative ai can write text . "
        "generative ai can create images . language models predict the next word . "
        "a language model learns from text .")
# ---- Step 1: TOKENIZATION: text -> tokens -> integer ids ----
tokens = text.split()
vocab = sorted(set(tokens)); w2i = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
ids = np.array([w2i[w] for w in tokens])
print("tokens:", tokens[:6]); print("ids   :", ids[:6].tolist())

# ---- Step 2: EMBEDDING: each id -> a trainable vector (a row of matrix E) ----
d = 8
E = rng.normal(0, 0.3, (V, d)); W = rng.normal(0, 0.3, (d, V)); b = np.zeros(V)
print("embedding of 'generative' (first 4 dims):", np.round(E[w2i['generative']][:4], 3))

x, y = ids[:-1], ids[1:]                          # input word -> next word
params = {'E': E, 'W': W, 'b': b}
m = {k: np.zeros_like(p) for k, p in params.items()}; v = {k: np.zeros_like(p) for k, p in params.items()}
losses, ppl = [], []
for ep in range(1, 301):
    emb = E[x]; logits = emb @ W + b
    p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    loss = -np.mean(np.log(p[np.arange(len(y)), y]))             # cross-entropy
    losses.append(loss); ppl.append(np.exp(loss))                 # perplexity = exp(loss)
    dl = p.copy(); dl[np.arange(len(y)), y] -= 1; dl /= len(y)
    g = {'W': emb.T @ dl, 'b': dl.sum(0), 'E': np.zeros_like(E)}
    np.add.at(g['E'], x, dl @ W.T)
    for k in params:
        m[k] = 0.9*m[k] + 0.1*g[k]; v[k] = 0.999*v[k] + 0.001*g[k]**2
        params[k] -= 0.05 * (m[k]/(1-0.9**ep)) / (np.sqrt(v[k]/(1-0.999**ep)) + 1e-8)
    if ep % 100 == 0 or ep == 1: print(f"epoch {ep:3d}  loss {loss:.3f}  perplexity {ppl[-1]:.2f}")

def generate(start, n=10, temp=0.8):
    out = [start]
    for _ in range(n):
        lg = (E[w2i[out[-1]]] @ W + b) / temp
        pr = np.exp(lg - lg.max()); pr /= pr.sum()
        out.append(vocab[rng.choice(V, p=pr)])
    return " ".join(out)
print("\nGenerated:", generate("generative"))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(losses); ax[0].set_title("loss vs epoch"); ax[1].plot(ppl); ax[1].set_title("perplexity vs epoch")
for a in ax: a.set_xlabel("epoch")
plt.tight_layout(); plt.show()

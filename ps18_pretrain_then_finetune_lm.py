# PS18: "Fine-tuning" without a pretrained download: pretrain a tiny neural LM on general text, then fine-tune on AI text.
# Shows perplexity before/after, perplexity vs epoch, and generation with temperature / top-k. (numpy)
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)

general = ("i like to eat rice and dal . the weather is nice today . we play cricket in the park . "
           "my friend likes to read books . the market is open today . we eat lunch at home .").split()
ai = ("artificial intelligence is the study of machines that learn . language models learn to predict the next word . "
      "generative ai can create text and images . the model learns patterns from data .").split()
held_out = "language models learn patterns from data".split()

vocab = sorted(set(general + ai + held_out)); V = len(vocab); w2i = {w: i for i, w in enumerate(vocab)}
d = 16
P = {'E': rng.normal(0, 0.3, (V, d)), 'W': rng.normal(0, 0.3, (d, V)), 'b': np.zeros(V)}   # embedding + output layer
state = {'m': {k: np.zeros_like(x) for k, x in P.items()}, 'v': {k: np.zeros_like(x) for k, x in P.items()}, 't': 0}

def pairs(words):
    ids = [w2i[w] for w in words]; return np.array(ids[:-1]), np.array(ids[1:])

def loss_and_grad(x, y):
    emb = P['E'][x]; lg = emb @ P['W'] + P['b']
    p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    loss = -np.mean(np.log(p[np.arange(len(y)), y]))
    dl = p.copy(); dl[np.arange(len(y)), y] -= 1; dl /= len(y)
    g = {'W': emb.T @ dl, 'b': dl.sum(0), 'E': np.zeros_like(P['E'])}; np.add.at(g['E'], x, dl @ P['W'].T)
    return loss, g

def evaluate(x, y): return np.exp(loss_and_grad(x, y)[0])          # perplexity

def train(words, epochs, lr, track=None):
    x, y = pairs(words); curve = []
    for _ in range(epochs):
        l, g = loss_and_grad(x, y); state['t'] += 1; t = state['t']
        for k in P:
            state['m'][k] = 0.9*state['m'][k] + 0.1*g[k]; state['v'][k] = 0.999*state['v'][k] + 0.001*g[k]**2
            P[k] -= lr * (state['m'][k]/(1-0.9**t)) / (np.sqrt(state['v'][k]/(1-0.999**t)) + 1e-8)
        if track is not None: curve.append(evaluate(*track))
    return curve

def generate(start, n=10, temp=0.8, top_k=0):
    out = [start]
    for _ in range(n):
        lg = (P['E'][w2i[out[-1]]] @ P['W'] + P['b']) / temp
        pr = np.exp(lg - lg.max()); pr /= pr.sum()
        if top_k: pr[np.argsort(pr)[::-1][top_k:]] = 0; pr /= pr.sum()
        out.append(vocab[rng.choice(V, p=pr)])
    return " ".join(out)

ho = pairs(held_out)
train(general, 300, 0.05)                                           # step 1: "pretraining" on general text
print("After pretraining  -> perplexity on AI held-out text:", round(float(evaluate(*ho)), 2))
print("Generated (pretrained):", generate("language"))
curve = train(ai, 150, 0.05, track=ho)                              # step 2: fine-tune on AI text
print("After fine-tuning  -> perplexity on AI held-out text:", round(float(evaluate(*ho)), 2))
print("Generated (fine-tuned):", generate("language"))
print("Generated (fine-tuned, top_k=3, temp=0.5):", generate("language", temp=0.5, top_k=3))
plt.plot(curve); plt.xlabel("fine-tuning epoch"); plt.ylabel("held-out perplexity"); plt.title("perplexity vs epoch"); plt.show()

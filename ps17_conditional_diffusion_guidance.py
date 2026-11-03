# PS17: Conditional diffusion with classifier-free guidance (toy "text-to-image" idea, numpy, no pretrained model)
# The "prompt" is a class label (0 = upper moon, 1 = lower moon). Guidance scale w controls how strongly we follow the prompt.
import numpy as np, math
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)

def moons(n):
    th = rng.uniform(0, np.pi, n // 2)
    a = np.c_[np.cos(th), np.sin(th)]; b = np.c_[1 - np.cos(th), 0.5 - np.sin(th)]
    X = np.vstack([a, b]) + rng.normal(0, 0.05, (n, 2)); lab = np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)
    return (X - X.mean(0)) * 1.5, lab
X, lab = moons(4000)

T = 100
betas = np.linspace(1e-4 * 1000 / T, 0.02 * 1000 / T, T); alphas = 1 - betas; ab = np.cumprod(alphas)
ab_prev = np.append(1.0, ab[:-1])

def init(i, o): return rng.normal(0, np.sqrt(1 / i), (i, o)), np.zeros(o)
P = {}; P['W1'], P['b1'] = init(5, 128); P['W2'], P['b2'] = init(128, 128); P['W3'], P['b3'] = init(128, 2)
def cond(c):                                       # one-hot prompt; c = -1 means "no prompt" (all zeros)
    out = np.zeros((len(c), 2)); ok = c >= 0; out[ok, c[ok]] = 1; return out
def net(x, t, c):
    inp = np.c_[x, t / T, cond(c)]
    a1 = np.tanh(inp @ P['W1'] + P['b1']); a2 = np.tanh(a1 @ P['W2'] + P['b2'])
    return a2 @ P['W3'] + P['b3'], (inp, a1, a2)

m = {k: np.zeros_like(v) for k, v in P.items()}; v = {k: np.zeros_like(x) for k, x in P.items()}
for i in range(1, 7001):
    idx = rng.integers(0, len(X), 256); x0 = X[idx]; c = lab[idx].copy()
    c[rng.random(256) < 0.2] = -1                  # randomly drop the prompt 20% of the time (needed for guidance)
    t = rng.integers(0, T, 256); eps = rng.standard_normal(x0.shape)
    xt = np.sqrt(ab[t])[:, None] * x0 + np.sqrt(1 - ab[t])[:, None] * eps
    pred, (inp, a1, a2) = net(xt, t, c)
    d3 = 2 * (pred - eps) / pred.size
    g = {'W3': a2.T @ d3, 'b3': d3.sum(0)}
    d2 = (d3 @ P['W3'].T) * (1 - a2**2); g['W2'] = a1.T @ d2; g['b2'] = d2.sum(0)
    d1 = (d2 @ P['W2'].T) * (1 - a1**2); g['W1'] = inp.T @ d1; g['b1'] = d1.sum(0)
    for k in P:
        m[k] = 0.9*m[k] + 0.1*g[k]; v[k] = 0.999*v[k] + 0.001*g[k]**2
        P[k] -= 2e-3 * (m[k]/(1-0.9**i)) / (np.sqrt(v[k]/(1-0.999**i)) + 1e-8)

def sample(c_target, w, n=600):
    x = rng.standard_normal((n, 2)); cc = np.full(n, c_target); cu = np.full(n, -1)
    for t in range(T - 1, -1, -1):
        e_c, _ = net(x, np.full(n, t), cc); e_u, _ = net(x, np.full(n, t), cu)
        eps = e_u + w * (e_c - e_u)                # classifier-free guidance formula
        x0 = ((x - np.sqrt(1 - ab[t]) * eps) / np.sqrt(ab[t])).clip(-3, 3)
        mean = (np.sqrt(ab_prev[t]) * betas[t] / (1 - ab[t])) * x0 + (np.sqrt(alphas[t]) * (1 - ab_prev[t]) / (1 - ab[t])) * x
        var = betas[t] * (1 - ab_prev[t]) / (1 - ab[t])
        x = mean + (np.sqrt(var) * rng.standard_normal(x.shape) if t > 0 else 0)
    return x

def accuracy(gen, c_target):                       # is each generated point nearest to a real point of the requested class?
    d = ((gen[:, None, :] - X[None, :1500, :]) ** 2).sum(-1); return np.mean(lab[:1500][d.argmin(1)] == c_target)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for a, w in zip(ax, [0.0, 1.0, 3.0]):
    g0, g1 = sample(0, w), sample(1, w)
    print(f"guidance w={w}: prompt 0 accuracy {accuracy(g0, 0):.2f}, prompt 1 accuracy {accuracy(g1, 1):.2f}")
    a.scatter(*g0.T, s=3, label="prompt 0"); a.scatter(*g1.T, s=3, label="prompt 1"); a.set_title(f"guidance scale w = {w}"); a.legend()
plt.tight_layout(); plt.show()

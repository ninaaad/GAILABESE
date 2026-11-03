# PS11: Simple DDPM on a 2-D "two moons" dataset (numpy only). Linear vs cosine noise schedule.
import numpy as np, math
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)

# ---- data: two moons, scaled to roughly [-1.7, 1.7] ----
def moons(n):
    th = rng.uniform(0, np.pi, n // 2)
    a = np.c_[np.cos(th), np.sin(th)]; b = np.c_[1 - np.cos(th), 0.5 - np.sin(th)]
    X = np.vstack([a, b]) + rng.normal(0, 0.05, (n, 2))
    return (X - X.mean(0)) * 1.5
data = moons(4000)

# ---- noise schedules ----
T = 100
def schedule(kind):
    if kind == "linear":
        betas = np.linspace(1e-4 * 1000 / T, 0.02 * 1000 / T, T).clip(max=0.999)
    else:                                              # cosine: define alpha_bar directly
        s = np.arange(T + 1) / T
        f = np.cos((s + 0.008) / 1.008 * math.pi / 2) ** 2
        ab = f / f[0]; betas = (1 - ab[1:] / ab[:-1]).clip(max=0.999)
    alphas = 1 - betas; return betas, alphas, np.cumprod(alphas)

# ---- tiny MLP noise predictor eps_theta(x_t, t): 3 -> 128 -> 128 -> 2 ----
def init(i, o): return rng.normal(0, np.sqrt(1 / i), (i, o)), np.zeros(o)
def make_net():
    P = {}; P['W1'], P['b1'] = init(3, 128); P['W2'], P['b2'] = init(128, 128); P['W3'], P['b3'] = init(128, 2); return P
def net(P, x, t):
    inp = np.c_[x, t / T]
    a1 = np.tanh(inp @ P['W1'] + P['b1']); a2 = np.tanh(a1 @ P['W2'] + P['b2'])
    return a2 @ P['W3'] + P['b3'], (inp, a1, a2)

def train(kind, steps=5000, bs=256, lr=2e-3):
    betas, alphas, ab = schedule(kind); P = make_net()
    m = {k: np.zeros_like(v) for k, v in P.items()}; v = {k: np.zeros_like(x) for k, x in P.items()}; losses = []
    for i in range(1, steps + 1):
        x0 = data[rng.integers(0, len(data), bs)]; t = rng.integers(0, T, bs)
        eps = rng.standard_normal(x0.shape)
        xt = np.sqrt(ab[t])[:, None] * x0 + np.sqrt(1 - ab[t])[:, None] * eps   # forward noising in one shot
        pred, (inp, a1, a2) = net(P, xt, t)
        losses.append(np.mean((pred - eps) ** 2))                               # predict the noise
        d3 = 2 * (pred - eps) / pred.size
        g = {'W3': a2.T @ d3, 'b3': d3.sum(0)}
        d2 = (d3 @ P['W3'].T) * (1 - a2**2); g['W2'] = a1.T @ d2; g['b2'] = d2.sum(0)
        d1 = (d2 @ P['W2'].T) * (1 - a1**2); g['W1'] = inp.T @ d1; g['b1'] = d1.sum(0)
        for k in P:                                                             # Adam
            m[k] = 0.9*m[k] + 0.1*g[k]; v[k] = 0.999*v[k] + 0.001*g[k]**2
            P[k] -= lr * (m[k]/(1-0.9**i)) / (np.sqrt(v[k]/(1-0.999**i)) + 1e-8)
    return P, losses

def sample(P, kind, n=800):
    betas, alphas, ab = schedule(kind)
    ab_prev = np.append(1.0, ab[:-1])
    x = rng.standard_normal((n, 2))                                             # start from pure noise
    for t in range(T - 1, -1, -1):
        eps, _ = net(P, x, np.full(n, t))
        x0 = ((x - np.sqrt(1 - ab[t]) * eps) / np.sqrt(ab[t])).clip(-3, 3)      # predicted clean point (clipped for stability)
        # same reverse step as the DDPM formula, written via the posterior mean
        mean = (np.sqrt(ab_prev[t]) * betas[t] / (1 - ab[t])) * x0 + (np.sqrt(alphas[t]) * (1 - ab_prev[t]) / (1 - ab[t])) * x
        var = betas[t] * (1 - ab_prev[t]) / (1 - ab[t])
        x = mean + (np.sqrt(var) * rng.standard_normal(x.shape) if t > 0 else 0)
    return x

def dist_to_real(gen):                                  # mean distance from each generated point to nearest real point
    d = np.sqrt(((gen[:, None, :] - data[None, :1500, :]) ** 2).sum(-1)); return d.min(1).mean()

results = {}
for kind in ["linear", "cosine"]:
    P, losses = train(kind); gen = sample(P, kind)
    results[kind] = (losses, gen)
    print(f"{kind:7s} final loss {np.mean(losses[-200:]):.3f} | sqrt(alpha_bar_T) {np.sqrt(schedule(kind)[2][-1]):.4f} | mean dist to real {dist_to_real(gen):.3f}")

fig, ax = plt.subplots(1, 4, figsize=(18, 4))
ax[0].scatter(*data[:800].T, s=3); ax[0].set_title("real data")
for a, kind in zip(ax[1:3], results): a.scatter(*results[kind][1].T, s=3, c='C1'); a.set_title(f"generated ({kind}, T={T})")
for kind in results: ax[3].plot(np.sqrt(schedule(kind)[2]), label=kind)
ax[3].set_title("signal level sqrt(alpha_bar) vs t"); ax[3].legend()
plt.tight_layout(); plt.show()

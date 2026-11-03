# PS9: Variational Autoencoder from scratch (numpy): reparameterization trick, recon + KL loss, generate new images
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)

def make_shapes(n):                            # same tiny 8x8 dataset as PS8
    X = np.zeros((n, 8, 8)); y = rng.integers(0, 3, n)
    for i in range(n):
        r, c = rng.integers(0, 8, 2)
        if y[i] == 0: X[i, r, :] = 1
        elif y[i] == 1: X[i, :, c] = 1
        else:
            r, c = rng.integers(0, 7, 2); X[i, r:r+2, c:c+2] = 1
    return X.reshape(n, -1), y
Xtr, _ = make_shapes(2000); Xte, yte = make_shapes(300)

D, H, Z = 64, 64, 2                            # 2-D latent so we can plot it
def init(i, o):
    lim = np.sqrt(6 / (i + o)); return rng.uniform(-lim, lim, (i, o)), np.zeros(o)
p = {}
for name, (i, o) in dict(e1=(D, H), mu=(H, Z), lv=(H, Z), d1=(Z, H), d2=(H, D)).items():
    p['W' + name], p['b' + name] = init(i, o)
m = {k: np.zeros_like(x) for k, x in p.items()}; v = {k: np.zeros_like(x) for k, x in p.items()}; t = 0
sig = lambda x: 1 / (1 + np.exp(-x))

def enc(x):
    h = np.maximum(0, x @ p['We1'] + p['be1'])
    return h @ p['Wmu'] + p['bmu'], h @ p['Wlv'] + p['blv']       # mean and log-variance
def dec(z): return sig(np.tanh(z @ p['Wd1'] + p['bd1']) @ p['Wd2'] + p['bd2'])

def step(x, lr=2e-3):
    global t
    n = len(x)
    h1 = x @ p['We1'] + p['be1']; h = np.maximum(0, h1)
    mu = h @ p['Wmu'] + p['bmu']; lv = h @ p['Wlv'] + p['blv']
    std = np.exp(0.5 * lv); eps = rng.standard_normal(mu.shape)
    z = mu + std * eps                                              # REPARAMETERIZATION TRICK
    d = np.tanh(z @ p['Wd1'] + p['bd1']); out = sig(d @ p['Wd2'] + p['bd2'])
    rec = -np.sum(x*np.log(out+1e-9) + (1-x)*np.log(1-out+1e-9)) / n
    kl = -0.5 * np.sum(1 + lv - mu**2 - np.exp(lv)) / n
    g = {}
    dp = (out - x) / n
    g['Wd2'] = d.T @ dp; g['bd2'] = dp.sum(0)
    dd = (dp @ p['Wd2'].T) * (1 - d**2)
    g['Wd1'] = z.T @ dd; g['bd1'] = dd.sum(0)
    dz = dd @ p['Wd1'].T
    dmu = dz + mu / n                                               # + gradient of KL
    dlv = dz * eps * 0.5 * std + 0.5 * (np.exp(lv) - 1) / n
    g['Wmu'] = h.T @ dmu; g['bmu'] = dmu.sum(0)
    g['Wlv'] = h.T @ dlv; g['blv'] = dlv.sum(0)
    dh = (dmu @ p['Wmu'].T + dlv @ p['Wlv'].T) * (h1 > 0)
    g['We1'] = x.T @ dh; g['be1'] = dh.sum(0)
    t += 1
    for k in p:
        m[k] = 0.9*m[k] + 0.1*g[k]; v[k] = 0.999*v[k] + 0.001*g[k]**2
        p[k] -= lr * (m[k]/(1-0.9**t)) / (np.sqrt(v[k]/(1-0.999**t)) + 1e-8)
    return rec + kl, rec, kl

hist = []
for ep in range(80):
    idx = rng.permutation(len(Xtr)); s = np.zeros(3)
    for i in range(0, len(Xtr), 64):
        s += step(Xtr[idx[i:i+64]])
    hist.append(s / (len(Xtr)/64))
    if ep % 20 == 0 or ep == 79: print(f"epoch {ep:2d}  total {hist[-1][0]:.2f}  recon {hist[-1][1]:.2f}  KL {hist[-1][2]:.2f}")
hist = np.array(hist)

# generate: sample z ~ N(0, I) and use ONLY the decoder
gen = dec(rng.standard_normal((16, Z)))
mu, _ = enc(Xte)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
ax[0].plot(hist[:, 0], label="total"); ax[0].plot(hist[:, 1], label="recon"); ax[0].plot(hist[:, 2], label="KL"); ax[0].legend(); ax[0].set_title("loss curves")
sc = ax[1].scatter(mu[:, 0], mu[:, 1], c=yte, cmap='viridis', s=10); ax[1].set_title("latent space (colour = shape class)")
grid = np.vstack([np.hstack([gen[r*4+c].reshape(8, 8) for c in range(4)]) for r in range(4)])
ax[2].imshow(grid, cmap='gray'); ax[2].axis('off'); ax[2].set_title("NEW images sampled from prior")
plt.tight_layout(); plt.show()

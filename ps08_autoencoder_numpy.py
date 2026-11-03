# PS8: Autoencoder from scratch (numpy): reconstruction, latent vectors, interpolation, loss curve
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)

# --- tiny synthetic "image" dataset (8x8): horizontal bar, vertical bar or 2x2 square ---
def make_shapes(n):
    X = np.zeros((n, 8, 8)); y = rng.integers(0, 3, n)
    for i in range(n):
        r, c = rng.integers(0, 8, 2)
        if y[i] == 0: X[i, r, :] = 1
        elif y[i] == 1: X[i, :, c] = 1
        else:
            r, c = rng.integers(0, 7, 2); X[i, r:r+2, c:c+2] = 1
    return X.reshape(n, -1), y
Xtr, _ = make_shapes(1500); Xte, yte = make_shapes(100)

# --- network: 64 -> 32 -> Z(4) -> 32 -> 64 ---
D, H, Z = 64, 32, 4
def init(i, o):
    lim = np.sqrt(6 / (i + o)); return rng.uniform(-lim, lim, (i, o)), np.zeros(o)
p = {}
for name, (i, o) in dict(e1=(D, H), e2=(H, Z), d1=(Z, H), d2=(H, D)).items():
    p['W' + name], p['b' + name] = init(i, o)
m = {k: np.zeros_like(v) for k, v in p.items()}; v = {k: np.zeros_like(x) for k, x in p.items()}; t = 0

sig = lambda x: 1 / (1 + np.exp(-x))
def encode(x): return np.maximum(0, x @ p['We1'] + p['be1']) @ p['We2'] + p['be2']
def decode(z): return sig(np.maximum(0, z @ p['Wd1'] + p['bd1']) @ p['Wd2'] + p['bd2'])

def step(x, lr=2e-3):
    global t
    n = len(x)
    h1 = x @ p['We1'] + p['be1']; a1 = np.maximum(0, h1)           # forward
    z = a1 @ p['We2'] + p['be2']
    h2 = z @ p['Wd1'] + p['bd1']; a2 = np.maximum(0, h2)
    out = sig(a2 @ p['Wd2'] + p['bd2'])
    loss = -np.mean(np.sum(x*np.log(out+1e-9) + (1-x)*np.log(1-out+1e-9), 1))   # BCE
    g = {}                                                          # backward
    dp = (out - x) / n
    g['Wd2'] = a2.T @ dp; g['bd2'] = dp.sum(0)
    dh2 = (dp @ p['Wd2'].T) * (h2 > 0)
    g['Wd1'] = z.T @ dh2; g['bd1'] = dh2.sum(0)
    dz = dh2 @ p['Wd1'].T
    g['We2'] = a1.T @ dz; g['be2'] = dz.sum(0)
    dh1 = (dz @ p['We2'].T) * (h1 > 0)
    g['We1'] = x.T @ dh1; g['be1'] = dh1.sum(0)
    t += 1                                                          # Adam update
    for k in p:
        m[k] = 0.9*m[k] + 0.1*g[k]; v[k] = 0.999*v[k] + 0.001*g[k]**2
        p[k] -= lr * (m[k]/(1-0.9**t)) / (np.sqrt(v[k]/(1-0.999**t)) + 1e-8)
    return loss

losses = []
for ep in range(60):
    idx = rng.permutation(len(Xtr)); tot = 0
    for i in range(0, len(Xtr), 50):
        tot += step(Xtr[idx[i:i+50]])
    losses.append(tot / (len(Xtr)/50))
print("final loss:", round(losses[-1], 3))

rec = decode(encode(Xte[:8]))
print("Reconstruction MSE:", round(float(np.mean((rec - Xte[:8])**2)), 5))
print("Latent vector of test image 0:", np.round(encode(Xte[:1])[0], 3))

# interpolation between two different shapes
a, b = Xte[np.where(yte == 0)[0][0]], Xte[np.where(yte == 1)[0][0]]
za, zb = encode(a[None]), encode(b[None])
steps = [decode((1-s)*za + s*zb)[0] for s in np.linspace(0, 1, 8)]

fig, ax = plt.subplots(4, 8, figsize=(12, 6.5))
for i in range(8):
    ax[0, i].imshow(Xte[i].reshape(8, 8), cmap='gray'); ax[1, i].imshow(rec[i].reshape(8, 8), cmap='gray')
    ax[2, i].imshow(steps[i].reshape(8, 8), cmap='gray')
    ax[3, i].axis('off')
for a_ in ax[:3].ravel(): a_.axis('off')
ax[0, 0].set_title("original", loc='left'); ax[1, 0].set_title("reconstructed", loc='left'); ax[2, 0].set_title("interpolation A->B", loc='left')
ax3 = fig.add_subplot(4, 1, 4); ax3.plot(losses); ax3.set_title("training loss"); ax3.set_xlabel("epoch")
plt.tight_layout(); plt.show()

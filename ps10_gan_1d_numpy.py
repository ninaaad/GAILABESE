# PS10: Basic GAN from scratch (numpy) learning a 1-D Gaussian. Shows generator/discriminator losses + histogram.
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)

REAL_MEAN, REAL_STD = 4.0, 1.25                 # the distribution G must learn
H = 32
def init(i, o): return rng.normal(0, 0.5, (i, o)), np.zeros(o)
G = {}; Dn = {}
G['W1'], G['b1'] = init(1, H); G['W2'], G['b2'] = init(H, 1); G['Wz'] = rng.normal(0, 0.5, (1, 1))   # generator: noise -> fake sample
Dn['W1'], Dn['b1'] = init(1, H); Dn['W2'], Dn['b2'] = init(H, 1)   # discriminator: sample -> P(real)
sig = lambda x: 1 / (1 + np.exp(-x))

class Adam:
    def __init__(s, P, lr): s.P, s.lr, s.t = P, lr, 0; s.m = {k: np.zeros_like(x) for k, x in P.items()}; s.v = {k: np.zeros_like(x) for k, x in P.items()}
    def step(s, g):
        s.t += 1
        for k in g:
            s.m[k] = 0.9*s.m[k] + 0.1*g[k]; s.v[k] = 0.999*s.v[k] + 0.001*g[k]**2
            s.P[k] -= s.lr * (s.m[k]/(1-0.9**s.t)) / (np.sqrt(s.v[k]/(1-0.999**s.t)) + 1e-8)
optG, optD = Adam(G, 3e-4), Adam(Dn, 3e-4)

def gen(z):
    g = np.tanh(z @ G['W1'] + G['b1']); return g @ G['W2'] + G['b2'] + z @ G['Wz'], g
def disc(x):
    h = np.tanh(x @ Dn['W1'] + Dn['b1']); return sig(h @ Dn['W2'] + Dn['b2']), h
def disc_back(x, h, dlogit):                    # gradients of D params and of its input x
    gD = {'W2': h.T @ dlogit, 'b2': dlogit.sum(0)}
    dpre = (dlogit @ Dn['W2'].T) * (1 - h**2)
    gD['W1'] = x.T @ dpre; gD['b1'] = dpre.sum(0)
    return gD, dpre @ Dn['W1'].T

B = 256; dl, gl = [], []
for it in range(8000):
    real = rng.normal(REAL_MEAN, REAL_STD, (B, 1)); z = rng.normal(0, 1, (B, 1))
    fake, g_h = gen(z)
    # ---- train D: real -> 1, fake -> 0 ----
    pr, hr = disc(real); pf, hf = disc(fake)
    g1, _ = disc_back(real, hr, (pr - 1) / B); g2, _ = disc_back(fake, hf, pf / B)
    optD.step({k: g1[k] + g2[k] for k in g1})
    dl.append(-np.mean(np.log(pr+1e-9)) - np.mean(np.log(1-pf+1e-9)))
    # ---- train G: make D say "real" (non-saturating loss) ----
    pf, hf = disc(fake)
    _, dx = disc_back(fake, hf, (pf - 1) / B)    # gradient flowing back into the fake sample
    gG = {'W2': g_h.T @ dx, 'b2': dx.sum(0)}
    dpre = (dx @ G['W2'].T) * (1 - g_h**2)
    gG['W1'] = z.T @ dpre; gG['b1'] = dpre.sum(0); gG['Wz'] = z.T @ dx
    optG.step(gG)
    gl.append(-np.mean(np.log(pf+1e-9)))
    if it % 2000 == 0: print(f"iter {it}  D_loss {dl[-1]:.3f}  G_loss {gl[-1]:.3f}  fake mean {fake.mean():.2f} std {fake.std():.2f}")

fake, _ = gen(rng.normal(0, 1, (2000, 1)))
print(f"\nReal: mean {REAL_MEAN}, std {REAL_STD}   |   Generated: mean {fake.mean():.2f}, std {fake.std():.2f}")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(dl, label="D loss", alpha=.7); ax[0].plot(gl, label="G loss", alpha=.7); ax[0].legend(); ax[0].set_title("GAN losses")
ax[1].hist(rng.normal(REAL_MEAN, REAL_STD, 2000), 40, alpha=.5, label="real"); ax[1].hist(fake.ravel(), 40, alpha=.5, label="generated"); ax[1].legend(); ax[1].set_title("distributions")
plt.tight_layout(); plt.show()

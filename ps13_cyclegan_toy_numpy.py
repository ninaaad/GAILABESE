# PS13: Toy CycleGAN in 1-D (numpy). Domain A ~ N(0,1), Domain B ~ N(4,2), unpaired.
# G: A->B, F: B->A (both linear), D_A and D_B are logistic discriminators on features [x, x^2].
# Loss = adversarial + lambda * cycle-consistency (L1), lambda = 10
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
LAM = 10.0
sig = lambda x: 1 / (1 + np.exp(-x))

P = dict(a=np.array(1.0), b=np.array(0.0),          # G(x) = a*x + b   (A -> B)
         c=np.array(1.0), d=np.array(0.0),          # F(y) = c*y + d   (B -> A)
         wA=np.zeros(2), bA=np.array(0.0), wB=np.zeros(2), bB=np.array(0.0))
m = {k: np.zeros_like(v) for k, v in P.items()}; v = {k: np.zeros_like(x) for k, x in P.items()}; t = 0
def adam(g, keys, lr):
    global t; t += 1
    for k in keys:
        m[k] = 0.9*m[k] + 0.1*g[k]; v[k] = 0.999*v[k] + 0.001*g[k]**2
        P[k] = P[k] - lr * (m[k]/(1-0.9**t)) / (np.sqrt(v[k]/(1-0.999**t)) + 1e-8)

phi = lambda x: np.c_[x, x**2]
def Dprob(w, b, x): return sig(phi(x) @ w + b)

hist = []
for it in range(15000):
    n = 128
    xa = rng.normal(0, 1, n); xb = rng.normal(4, 2, n)      # unpaired samples
    fb = P['a'] * xa + P['b']; fa = P['c'] * xb + P['d']     # translations
    # ---- update discriminators (real -> 1, fake -> 0) ----
    g = {}
    for name, real, fake in [('B', xb, fb), ('A', xa, fa)]:
        pr, pf = Dprob(P['w'+name], P['b'+name], real), Dprob(P['w'+name], P['b'+name], fake)
        g['w'+name] = (phi(real).T @ (pr - 1) + phi(fake).T @ pf) / n
        g['b'+name] = np.mean(pr - 1) + np.mean(pf)
    adam(g, ['wA', 'bA', 'wB', 'bB'], 0.02)
    # ---- update generators: fool D + cycle consistency ----
    g = {k: 0.0 for k in 'abcd'}
    for name, fake, src, ka, kb in [('B', fb, xa, 'a', 'b'), ('A', fa, xb, 'c', 'd')]:
        w = P['w'+name]; p = Dprob(w, P['b'+name], fake)
        dy = (p - 1) * (w[0] + 2 * w[1] * fake) / n          # non-saturating adversarial gradient
        g[ka] += np.sum(dy * src); g[kb] += np.sum(dy)
    rec_a = P['c'] * fb + P['d']; s1 = np.sign(rec_a - xa) / n   # A -> B -> A
    g['c'] += LAM * np.sum(s1 * fb); g['d'] += LAM * np.sum(s1)
    g['a'] += LAM * np.sum(s1 * P['c'] * xa); g['b'] += LAM * np.sum(s1 * P['c'])
    rec_b = P['a'] * fa + P['b']; s2 = np.sign(rec_b - xb) / n   # B -> A -> B
    g['a'] += LAM * np.sum(s2 * fa); g['b'] += LAM * np.sum(s2)
    g['c'] += LAM * np.sum(s2 * P['a'] * xb); g['d'] += LAM * np.sum(s2 * P['a'])
    adam(g, list('abcd'), 0.02)
    hist.append(np.mean(np.abs(rec_a - xa)) + np.mean(np.abs(rec_b - xb)))
    if it % 3000 == 0: print(f"iter {it}  cycle loss {hist[-1]:.3f}  G(x)={P['a']:.2f}x+{P['b']:.2f}  F(y)={P['c']:.2f}y+{P['d']:.2f}")

xa = rng.normal(0, 1, 2000); xb = rng.normal(4, 2, 2000)
G_A = P['a'] * xa + P['b']; F_B = P['c'] * xb + P['d']
print(f"\nG(A): mean {G_A.mean():.2f}, std {G_A.std():.2f}   (target B: mean 4, std 2)")
print(f"F(B): mean {F_B.mean():.2f}, std {F_B.std():.2f}   (target A: mean 0, std 1)")
print("Cycle error A->B->A:", round(float(np.mean(np.abs(P['c']*G_A + P['d'] - xa))), 3))
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].hist(xa, 40, alpha=.5, label="A real"); ax[0].hist(F_B, 40, alpha=.5, label="B->A fake"); ax[0].legend(); ax[0].set_title("domain A")
ax[1].hist(xb, 40, alpha=.5, label="B real"); ax[1].hist(G_A, 40, alpha=.5, label="A->B fake"); ax[1].legend(); ax[1].set_title("domain B")
ax[2].plot(hist); ax[2].set_title("cycle-consistency loss"); ax[2].set_xlabel("iteration")
plt.tight_layout(); plt.show()

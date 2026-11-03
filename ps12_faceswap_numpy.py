# PS12: Face-swap idea: ONE shared encoder + TWO decoders (numpy, synthetic "faces", offline)
# Identity = shape of the face (A: circle, B: square). Pose = where the bright 3x3 blob (nose) sits.
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
N = 12                                          # image is 12x12

def make(identity, n):
    X = np.zeros((n, N, N)); pos = rng.integers(3, 9, (n, 2))
    yy, xx = np.mgrid[:N, :N]
    for i in range(n):
        if identity == 'A': X[i][(yy - 5.5)**2 + (xx - 5.5)**2 <= 16] = 0.5     # round face
        else: X[i, 2:10, 2:10] = 0.5                                             # square face
        X[i, pos[i, 0]-1:pos[i, 0]+2, pos[i, 1]-1:pos[i, 1]+2] = 1.0             # pose marker (3x3 bright blob)
    return X.reshape(n, -1), pos
A, posA = make('A', 600); B, posB = make('B', 600)

D, H, Z = N * N, 64, 3
def init(i, o): return rng.normal(0, np.sqrt(2 / i), (i, o)), np.zeros(o)
P = {}
P['eW1'], P['eb1'] = init(D, H); P['eW2'], P['eb2'] = init(H, Z)                 # shared encoder
for d in 'AB': P[d+'W1'], P[d+'b1'] = init(Z, H); P[d+'W2'], P[d+'b2'] = init(H, D)   # one decoder each
m = {k: np.zeros_like(x) for k, x in P.items()}; v = {k: np.zeros_like(x) for k, x in P.items()}; t = 0
sig = lambda x: 1 / (1 + np.exp(-x))

def encode(x): return np.maximum(0, x @ P['eW1'] + P['eb1']) @ P['eW2'] + P['eb2']
def decode(z, d): return sig(np.maximum(0, z @ P[d+'W1'] + P[d+'b1']) @ P[d+'W2'] + P[d+'b2'])

def step(x, d, lr=2e-3):
    global t
    n = len(x)
    h1 = x @ P['eW1'] + P['eb1']; a1 = np.maximum(0, h1); z = a1 @ P['eW2'] + P['eb2']
    h2 = z @ P[d+'W1'] + P[d+'b1']; a2 = np.maximum(0, h2); out = sig(a2 @ P[d+'W2'] + P[d+'b2'])
    dp = (out - x) / n                                                            # BCE + sigmoid
    g = {d+'W2': a2.T @ dp, d+'b2': dp.sum(0)}
    dh2 = (dp @ P[d+'W2'].T) * (h2 > 0); g[d+'W1'] = z.T @ dh2; g[d+'b1'] = dh2.sum(0)
    dz = dh2 @ P[d+'W1'].T; g['eW2'] = a1.T @ dz; g['eb2'] = dz.sum(0)
    dh1 = (dz @ P['eW2'].T) * (h1 > 0); g['eW1'] = x.T @ dh1; g['eb1'] = dh1.sum(0)
    t += 1
    for k in g:
        m[k] = 0.9*m[k] + 0.1*g[k]; v[k] = 0.999*v[k] + 0.001*g[k]**2
        P[k] -= lr * (m[k]/(1-0.9**t)) / (np.sqrt(v[k]/(1-0.999**t)) + 1e-8)
    return np.mean((out - x) ** 2)

for ep in range(4000):                          # the shared encoder sees A and B alternately
    i = rng.integers(0, 600, 64)
    la = step(A[i], 'A'); lb = step(B[i], 'B')
    if ep % 1000 == 0 or ep == 3999: print(f"step {ep:4d}  MSE A {la:.4f}  B {lb:.4f}")

# ---- swap: encode A's face, decode with B's decoder ----
swap = decode(encode(A[:6]), 'B')
def marker(img):                                 # centre of the bright pixels = where the pose marker is
    im = img.reshape(N, N); ys, xs = np.where(im > 0.75)
    return (ys.mean(), xs.mean()) if len(ys) else (0, 0)
err = np.mean([np.abs(np.array(marker(swap[i])) - posA[i]).sum() for i in range(6)])
print("Swapped image: marker position error vs A's pose =", round(float(err), 2), "pixels (small = A's pose kept, B's face drawn)")

rows = [A[:6], decode(encode(A[:6]), 'A'), swap, B[:6], decode(encode(B[:6]), 'B'), decode(encode(B[:6]), 'A')]
names = ["A original", "A reconstructed", "SWAP A pose -> B face", "B original", "B reconstructed", "SWAP B pose -> A face"]
fig, ax = plt.subplots(6, 6, figsize=(8, 8))
for r in range(6):
    for c in range(6): ax[r, c].imshow(rows[r][c].reshape(N, N), cmap='gray', vmin=0, vmax=1); ax[r, c].axis('off')
    ax[r, 0].set_title(names[r], loc='left', fontsize=8)
plt.tight_layout(); plt.show()

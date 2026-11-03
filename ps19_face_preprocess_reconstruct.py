# PS19: Face reconstruction system (educational): accept an image, preprocess, reconstruct with an autoencoder,
# compute reconstruction error, show original and reconstructed side by side, and flag a "manipulated" face. (numpy)
# No real face dataset needed: faces are drawn with numpy. A linear autoencoder (encoder = project, decoder = back-project)
# is solved in one shot with SVD, which is exactly what a trained linear autoencoder converges to (equivalent to PCA).
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)

def draw_face(size=32):                                  # a simple synthetic face with random variations
    yy, xx = np.mgrid[:size, :size]; img = np.full((size, size), 40.0)
    cx, cy = size/2 + rng.uniform(-1.5, 1.5), size/2 + rng.uniform(-1.5, 1.5)
    img[((xx - cx)/10)**2 + ((yy - cy)/13)**2 <= 1] = rng.uniform(150, 220)       # face oval
    ex = rng.uniform(3.5, 5)
    for sx in (-ex, ex): img[int(cy - 4):int(cy - 2), int(cx + sx - 1):int(cx + sx + 2)] = 20   # eyes
    w = int(rng.uniform(3, 6)); img[int(cy + 6):int(cy + 8), int(cx - w):int(cx + w)] = 60     # mouth
    return img + rng.normal(0, 3, img.shape)

def preprocess(img):                                     # 1) crop centre  2) resize 32->16  3) normalise to [0,1]
    img = img[4:28, 4:28]                                # crop the face region (24x24)
    img = img.reshape(12, 2, 12, 2).mean(axis=(1, 3))    # resize by averaging 2x2 blocks -> 12x12
    return (img - img.min()) / (img.max() - img.min() + 1e-9)

faces = np.array([preprocess(draw_face()).ravel() for _ in range(400)])      # training set (400, 144)
mean = faces.mean(0)
U, S, Vt = np.linalg.svd(faces - mean, full_matrices=False)
K = 8                                                    # bottleneck size
Wenc = Vt[:K].T                                          # encoder weights (144 -> 8)
encode = lambda x: (x - mean) @ Wenc
decode = lambda z: z @ Wenc.T + mean

def error(x): return np.mean((decode(encode(x)) - x) ** 2, axis=-1)           # reconstruction error (MSE)

test = np.array([preprocess(draw_face()).ravel() for _ in range(5)])
manip = test.copy().reshape(5, 12, 12); manip[:, 3:8, 3:9] = 1.0                # fake "manipulation": paste a bright patch
manip = manip.reshape(5, -1)
print("Mean reconstruction error, normal faces     :", round(float(error(test).mean()), 5))
print("Mean reconstruction error, manipulated faces:", round(float(error(manip).mean()), 5))
thr = error(faces).mean() + 3 * error(faces).std()
print("Threshold (mean+3std of training error):", round(float(thr), 5), "| flagged:", (error(manip) > thr).tolist())

fig, ax = plt.subplots(4, 5, figsize=(10, 8.5))
for i in range(5):
    for r, (img, name) in enumerate([(test[i], "original"), (decode(encode(test[i:i+1]))[0], "reconstructed"),
                                      (manip[i], "manipulated"), (decode(encode(manip[i:i+1]))[0], "recon of manipulated")]):
        ax[r, i].imshow(img.reshape(12, 12), cmap='gray', vmin=0, vmax=1); ax[r, i].axis('off')
        ax[r, i].set_title(f"{name}\nerr {error(img[None])[0]:.4f}", fontsize=8)
plt.tight_layout(); plt.show()
for K_ in [2, 4, 8, 16, 32]:                              # effect of bottleneck size on error
    W = Vt[:K_].T; print(f"bottleneck {K_:2d}: error {np.mean(((test - mean) @ W @ W.T + mean - test)**2):.5f}")

# PS5: Image generation with a Genetic Algorithm (evolve random pixels into a cross) - numpy + matplotlib
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)

S = 5                                         # image is 5x5
target = np.zeros((S, S)); target[2, :] = 1; target[:, 2] = 1   # the cross we want
target = target.flatten()

POP, GENS, MUT = 40, 200, 0.02
pop = rng.integers(0, 2, (POP, S * S))        # start: random noise images

def fitness(ind):                             # how many pixels match the target
    return (ind == target).sum()

best_hist, snaps = [], {}
for g in range(GENS):
    fit = np.array([fitness(i) for i in pop])
    best_hist.append(fit.max())
    if g in (0, 10, 30): snaps[g] = pop[fit.argmax()].copy()
    if fit.max() == S * S:                    # perfect match -> stop
        snaps[g] = pop[fit.argmax()].copy(); print("Perfect image at generation", g); break
    new = [pop[fit.argmax()].copy()]          # elitism: keep the best one
    while len(new) < POP:
        # tournament selection of 2 parents
        p1 = pop[max(rng.integers(0, POP, 3), key=lambda i: fit[i])]
        p2 = pop[max(rng.integers(0, POP, 3), key=lambda i: fit[i])]
        cut = rng.integers(1, S * S)          # one-point crossover
        child = np.concatenate([p1[:cut], p2[cut:]])
        flip = rng.random(S * S) < MUT        # mutation: flip a few pixels
        child[flip] = 1 - child[flip]
        new.append(child)
    pop = np.array(new)

fig, ax = plt.subplots(1, len(snaps) + 2, figsize=(14, 3))
ax[0].imshow(target.reshape(S, S), cmap='gray'); ax[0].set_title("target")
for a, (g, im) in zip(ax[1:], snaps.items()):
    a.imshow(im.reshape(S, S), cmap='gray'); a.set_title(f"gen {g}")
ax[-1].plot(best_hist); ax[-1].set_title("best fitness"); ax[-1].set_xlabel("generation")
for a in ax[:-1]: a.axis('off')
plt.tight_layout(); plt.show()

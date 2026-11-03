# PS6: Evolutionary 3x3 image generator (numpy only). Start from noise, mutate, keep if better.
import numpy as np
np.random.seed(1)

# target pattern: a cross, 1 = white, 0 = black
target = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]])

# the "AI" starting point: random noise
image = np.random.choice([0, 1], size=(3, 3))

def fitness(img):                      # number of matching pixels (max 9)
    return (img == target).sum()

def show(img):
    for row in img: print(" ".join("#" if p else "." for p in row))

print("Start (fitness", fitness(image), ")"); show(image)

gen = 0
while fitness(image) < 9:
    gen += 1
    child = image.copy()
    r, c = np.random.randint(0, 3, 2)  # mutation: flip ONE random pixel
    child[r, c] = 1 - child[r, c]
    if fitness(child) >= fitness(image):   # keep child only if it is not worse
        image = child
    if gen % 5 == 0:
        print(f"\ngeneration {gen}, fitness {fitness(image)}"); show(image)

print(f"\nTarget reached after {gen} generations:"); show(image)

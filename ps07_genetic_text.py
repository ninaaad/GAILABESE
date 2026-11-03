# PS7: Evolve random characters into a target sentence with a Genetic Algorithm (offline)
import random, string
random.seed(0)

TARGET = "GENERATIVE AI"
CHARS = string.ascii_uppercase + " "
POP, MUT = 100, 0.05

def fitness(s): return sum(a == b for a, b in zip(s, TARGET))
def rand_str(): return "".join(random.choice(CHARS) for _ in TARGET)

pop = [rand_str() for _ in range(POP)]
for gen in range(1, 1000):
    pop.sort(key=fitness, reverse=True)
    if gen % 10 == 1 or pop[0] == TARGET:
        print(f"gen {gen:3d}  best: {pop[0]}  fitness {fitness(pop[0])}/{len(TARGET)}")
    if pop[0] == TARGET: break
    parents = pop[:POP // 2]                  # selection: best half survive
    kids = []
    while len(kids) < POP - len(parents):
        a, b = random.sample(parents, 2)
        cut = random.randrange(len(TARGET))   # crossover
        kid = list(a[:cut] + b[cut:])
        for i in range(len(kid)):             # mutation
            if random.random() < MUT: kid[i] = random.choice(CHARS)
        kids.append("".join(kid))
    pop = parents + kids

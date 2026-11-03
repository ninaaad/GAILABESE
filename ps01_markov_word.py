# PS1: Text generation with a word-level Markov chain (only 'random' needed, 100% offline)
import random

text = "the cat sat on the mat and the cat ate the rat and the dog sat on the log"
words = text.split()

# 1. build the chain: current word -> list of words that followed it in the text
chain = {}
for i in range(len(words) - 1):
    chain.setdefault(words[i], []).append(words[i + 1])

print("Transition table:")
for k, v in chain.items():
    print(f"  {k!r} -> {v}")

# 2. generate: keep picking a random next word from the list
def generate(start, n=15):
    out = [start]
    for _ in range(n - 1):
        nxt = chain.get(out[-1])
        if not nxt:                    # dead end -> restart from a random word
            nxt = words
        out.append(random.choice(nxt))
    return " ".join(out)

random.seed(1)
for _ in range(3):
    print("\nGenerated:", generate("the"))

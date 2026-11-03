# PS2: Character-level Markov chain of order k (next-character prediction, no neural net)
import random

text = "generative ai can generate text images and music. generative models learn patterns from data."
ORDER = 3                                   # how many previous characters we look at

# 1. map every k-character context -> characters that followed it
model = {}
for i in range(len(text) - ORDER):
    ctx = text[i:i + ORDER]
    model.setdefault(ctx, []).append(text[i + ORDER])

# 2. start from the first k chars and keep predicting the next char
def generate(length=120):
    ctx = text[:ORDER]
    out = ctx
    for _ in range(length):
        nxt = model.get(ctx)
        if not nxt:                         # unseen context -> restart
            ctx = random.choice(list(model))
            nxt = model[ctx]
        c = random.choice(nxt)
        out += c
        ctx = out[-ORDER:]
    return out

random.seed(0)
print(generate())

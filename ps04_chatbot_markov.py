# PS4: Simple chatbot that runs until the user types 'exit' (Markov generation + a few rules, offline)
# Run: python ps04_chatbot_markov.py   (needs keyboard input)
import random

corpus = ("i am a small chatbot built with a markov chain . i can talk about ai and language models . "
          "a language model predicts the next word . generative ai can create text images and music . "
          "ask me about ai or just say hello .").split()

chain = {}
for a, b in zip(corpus, corpus[1:]):
    chain.setdefault(a, []).append(b)

MAX_NEW_TOKENS = 20                 # like max_new_tokens in the lab

def reply(user_text):
    t = user_text.lower()
    if any(g in t for g in ["hello", "hi", "hey"]):     # tiny rule layer
        return "Hello! Ask me something about AI."
    if "name" in t:
        return "I am a toy Markov chatbot."
    # otherwise: seed with the last known word from the user's text
    known = [w for w in t.replace("?", " ").split() if w in chain]
    word = known[-1] if known else random.choice(corpus)
    out = [word]
    for _ in range(MAX_NEW_TOKENS):
        word = random.choice(chain.get(word, corpus))
        out.append(word)
    return " ".join(out)

print("Chatbot ready. Type 'exit' to quit.")
while True:
    msg = input("You: ")
    if msg.strip().lower() == "exit":
        print("Bot: Goodbye!")
        break
    print("Bot:", reply(msg))

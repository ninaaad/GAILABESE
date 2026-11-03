# GenAI Lab ESE - Offline Codes (16 possible problem statements)

Your ma'am's reference screenshots show: **Python only, no dataset, no internet** ("100% offline"),
libraries `random`, `numpy`, `matplotlib`; target shapes (cross/square) hard-coded; tiny text typed in as a string.
So every script here needs **nothing downloaded**: no torch, no MNIST, no pretrained models.

## How to run
1. Python 3 installed. Libraries: `pip install numpy matplotlib` (do this BEFORE the exam; PS1, 2, 4, 6, 7 need only the standard library / numpy).
2. Open a terminal in this folder and run: `python ps01_markov_word.py` (use `python3` on Mac/Linux).
3. Colab / Jupyter also works: paste the file into a cell and run.
4. Scripts that show plots call `plt.show()`: close the plot window to let the script finish.
5. PS4 (chatbot) is interactive: type messages, type `exit` to quit.
6. Results are random but seeded, so you get the same output each run. Change the seed to get new samples.
7. If matplotlib is missing in the exam, the Markov/GA text scripts (PS1-4, 6, 7) still run, as they only use text output.

## Problem statements and files
| # | Problem statement (what may be asked) | File | Libraries |
|---|---|---|---|
| 1 | Text generation with a word-level Markov chain from 2-3 typed sentences | ps01_markov_word.py | random |
| 2 | Character-level Markov chain text generation (order k) | ps02_markov_char.py | random |
| 3 | Bigram language model with temperature, top-k, top-p sampling and perplexity | ps03_ngram_sampling_perplexity.py | numpy |
| 4 | Chatbot loop that replies until user types `exit` | ps04_chatbot_markov.py | random |
| 5 | Image generation using a Genetic Algorithm (random pixels evolve to a cross), plot fitness | ps05_genetic_image.py | numpy, matplotlib |
| 6 | Evolutionary 3x3 vector image generator (mutate noise to match target) | ps06_evolutionary_vector_image.py | numpy |
| 7 | Evolve random characters into a target sentence (GA) | ps07_genetic_text.py | random |
| 8 | Autoencoder: original vs reconstructed, latent vectors, latent interpolation, loss curve | ps08_autoencoder_numpy.py | numpy, matplotlib |
| 9 | VAE: reparameterization trick, recon + KL loss, generate new images from N(0,I) | ps09_vae_numpy.py | numpy, matplotlib |
| 10 | Basic GAN learning a 1-D Gaussian, G/D loss curves, real vs generated histogram | ps10_gan_1d_numpy.py | numpy, matplotlib |
| 11 | DDPM diffusion on 2-D two-moons, linear vs cosine noise schedule | ps11_diffusion_ddpm_numpy.py | numpy, matplotlib |
| 12 | Face-swap idea: shared encoder + two decoders (synthetic faces) | ps12_faceswap_numpy.py | numpy, matplotlib |
| 13 | CycleGAN (toy 1-D): two generators, two discriminators, cycle loss with lambda=10 | ps13_cyclegan_toy_numpy.py | numpy, matplotlib |
| 14 | Music generation: Markov melody, save WAV, waveform + spectrogram, compare 2 genres | ps14_music_markov_wav.py | numpy, wave, matplotlib |
| 15 | Tokenization vs embedding, tiny neural LM, loss and perplexity vs epochs | ps15_tokenize_embed_lm.py | numpy, matplotlib |
| 16 | Causal self-attention (Transformer core) with attention heatmap | ps16_self_attention_numpy.py | numpy, matplotlib |

## Which are most likely?
Highest chance (they match the 3 screenshots): **PS1, PS5, PS6** (+ PS2, PS7 as variations).
Next: PS3, PS4, PS8, PS9, PS10 (lab topics reduced to no-dataset form). The rest are bonus coverage.

## Quick things to say / know while running
- PS1/2: Markov = next token depends only on current state; same idea as next-token prediction in LLMs, but a lookup table instead of a neural net.
- PS5/6/7: fitness = number of matching pixels/chars; selection, crossover, mutation; elitism keeps the best.
- PS8: low loss = good reconstruction. Interpolation in latent space may look okay for a plain AE.
- PS9: loss = reconstruction + KL. Sampling z ~ N(0, I) and decoding makes new images.
- PS10: GAN losses oscillate (D loss near 1.386 = log 4 when D is confused). This 1-D GAN learns the mean well but the spread only partly: that is mode collapse / instability, a good viva point.
- PS11: linear and cosine both generate moons; cosine destroys signal more gradually.
- PS12: the swap works only because the encoder is shared and small (it keeps pose, not identity).
- PS13: G and F should come out roughly inverse of each other (cycle error near 0).
- PS15: tiny corpus means perplexity gets very low (memorisation). Real models do not.

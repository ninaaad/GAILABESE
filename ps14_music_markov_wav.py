# PS14: Music generation: Markov-chain melody -> WAV file -> waveform + spectrogram, two "genres" compared.
# Uses only numpy, wave (built-in) and matplotlib. Fully offline.
import numpy as np, wave
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
SR = 22050                                           # sample rate

scale = np.array([261.6, 293.7, 329.6, 392.0, 440.0, 523.3])     # C major pentatonic (Hz)

def transition_matrix(kind):
    n = len(scale); M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = np.exp(-abs(i - j) / (2.0 if kind == "jazz" else 0.7))   # jazz: smooth/wandering, rock: stays close
        if kind == "rock": M[i, 0] += 1.5                                      # rock keeps returning to the root note
    return M / M.sum(1, keepdims=True)

def tone(freq, dur, kind):
    t = np.arange(int(SR * dur)) / SR
    if kind == "jazz": w = np.sin(2*np.pi*freq*t) + 0.3*np.sin(2*np.pi*2*freq*t)            # soft, few harmonics
    else:              w = np.sign(np.sin(2*np.pi*freq*t)) * 0.6 + 0.4*np.sin(2*np.pi*3*freq*t)  # buzzy, harmonic-rich
    env = np.minimum(1, t / 0.02) * np.exp(-t * (2 if kind == "jazz" else 6))                # attack + decay
    return w * env

def make_song(kind, seconds=8):
    M = transition_matrix(kind); note_len = 0.5 if kind == "jazz" else 0.2
    state, audio = 0, []
    for _ in range(int(seconds / note_len)):
        audio.append(tone(scale[state], note_len, kind))
        state = rng.choice(len(scale), p=M[state])            # Markov step: next note depends only on current
    a = np.concatenate(audio); return a / np.abs(a).max() * 0.8

def save_wav(path, a):
    with wave.open(path, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR)
        f.writeframes((a * 32767).astype(np.int16).tobytes())

def load_wav(path):
    with wave.open(path, "rb") as f:
        return np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16) / 32767.0

def spectrogram(a, n=1024, hop=256):                         # STFT: window -> FFT -> magnitude in dB
    win = np.hanning(n)
    frames = np.array([a[i:i+n] * win for i in range(0, len(a) - n, hop)])
    return 20 * np.log10(np.abs(np.fft.rfft(frames, axis=1)).T + 1e-6)

fig, ax = plt.subplots(2, 2, figsize=(13, 7))
for col, kind in enumerate(["jazz", "rock"]):
    path = f"{kind}_sample.wav"; save_wav(path, make_song(kind)); y = load_wav(path)
    # integrity check: loads, not silent, no NaN
    print(f"{path}: {len(y)/SR:.1f}s, max amplitude {np.abs(y).max():.2f}, not silent: {np.abs(y).max() > 1e-4}, finite: {np.isfinite(y).all()}")
    ax[0, col].plot(np.arange(len(y)) / SR, y, lw=0.4); ax[0, col].set_title(f"{kind} waveform"); ax[0, col].set_xlabel("time (s)"); ax[0, col].set_ylabel("amplitude")
    S = spectrogram(y)
    ax[1, col].imshow(S, origin="lower", aspect="auto", extent=[0, len(y)/SR, 0, SR/2], cmap="magma", vmin=-20)
    ax[1, col].set_ylim(0, 4000); ax[1, col].set_title(f"{kind} spectrogram"); ax[1, col].set_xlabel("time (s)"); ax[1, col].set_ylabel("Hz")
plt.tight_layout(); plt.show()

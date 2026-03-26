def stft_channel(x, cfg):
    f, _, Z = stft(x, fs=50_000, nperseg=4096, ...)
    power = np.abs(Z) ** 2
    db = 10 * np.log10(power + 1e-12)   # → Dezibel-Skala
    ...

def to_rgb_image(vx, vy, vz, cfg):
    sx = stft_channel(vx, cfg)   # X-Achse → Roter Kanal
    sy = stft_channel(vy, cfg)   # Y-Achse → Grüner Kanal
    sz = stft_channel(vz, cfg)   # Z-Achse → Blauer Kanal
    rgb = np.stack([sx, sy, sz], axis=-1)
    img = img.resize((256, 256))  # → 256×256 RGB-Bild

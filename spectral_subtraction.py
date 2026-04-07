"""
Noise Removal in Speech Processing Using Spectral Subtraction
=============================================================
A complete Python implementation covering:
  - Synthetic noisy speech generation
  - Basic Spectral Subtraction (Boll, 1979)
  - Oversubtraction / Half-Wave Rectification variant
  - Power Spectral Subtraction
  - SNR & Spectral Distortion evaluation
  - Visualisation of waveforms and spectrograms

References
----------
[1] Boll, S. F. (1979). Suppression of acoustic noise in speech using
    spectral subtraction. IEEE Trans. Acoustics, Speech, Signal Processing.
[2] Berouti, M., Schwartz, R., & Makhoul, J. (1979). Enhancement of speech
    corrupted by acoustic noise. ICASSP.
[3] Martin, R. (2001). Noise power spectral density estimation based on
    optimal smoothing and minimum statistics. IEEE Trans. Speech Audio Processing.
"""

import numpy as np
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# 1.  SIGNAL GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def generate_synthetic_speech(duration: float = 3.0, fs: int = 16000) -> np.ndarray:
    """
    Synthesise a simple voiced + unvoiced speech-like signal.
    - Voiced regions: sum of harmonics (pitch 120 Hz, 5 harmonics)
    - Unvoiced regions: band-pass filtered noise  (fricative-like)
    - Silence regions: near zero
    """
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    speech = np.zeros_like(t)

    # Voiced segment 0.2 – 1.0 s
    mask_v = (t >= 0.2) & (t < 1.0)
    f0 = 120
    for k in range(1, 6):
        speech[mask_v] += (1.0 / k) * np.sin(2 * np.pi * k * f0 * t[mask_v])

    # Voiced segment 1.4 – 2.2 s
    mask_v2 = (t >= 1.4) & (t < 2.2)
    f0b = 150
    for k in range(1, 6):
        speech[mask_v2] += (0.8 / k) * np.sin(2 * np.pi * k * f0b * t[mask_v2])

    # Unvoiced (fricative) segment 1.0 – 1.4 s
    rng = np.random.default_rng(42)
    mask_u = (t >= 1.0) & (t < 1.4)
    fric = rng.standard_normal(mask_u.sum())
    b, a = signal.butter(4, [2000 / (fs / 2), 7000 / (fs / 2)], btype='band')
    speech[mask_u] = signal.lfilter(b, a, fric) * 0.6

    # Normalise
    speech /= np.max(np.abs(speech)) + 1e-12
    return speech.astype(np.float64)


def add_noise(speech: np.ndarray, snr_db: float, noise_type: str = 'white',
              fs: int = 16000) -> tuple[np.ndarray, np.ndarray]:
    """Add noise at a given SNR (dB). Returns (noisy, noise)."""
    rng = np.random.default_rng(0)
    n = len(speech)

    if noise_type == 'white':
        noise = rng.standard_normal(n)
    elif noise_type == 'pink':
        white = rng.standard_normal(n)
        # 1/f colouring via cumulative sum trick
        noise = np.cumsum(white)
        noise -= np.mean(noise)
    elif noise_type == 'babble':
        noise = rng.standard_normal(n)
        b, a = signal.butter(2, [300 / (fs / 2), 3400 / (fs / 2)], btype='band')
        noise = signal.lfilter(b, a, noise)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    speech_power = np.mean(speech ** 2)
    noise_power  = np.mean(noise  ** 2)
    scale = np.sqrt(speech_power / (noise_power * 10 ** (snr_db / 10)))
    noise *= scale
    return speech + noise, noise


# ──────────────────────────────────────────────────────────────────────────────
# 2.  SPECTRAL SUBTRACTION CORE
# ──────────────────────────────────────────────────────────────────────────────

def estimate_noise_psd(noisy: np.ndarray, fs: int,
                       noise_frames: int = 10,
                       frame_len: int = 512,
                       hop: int = 256) -> np.ndarray:
    """
    Estimate noise PSD from the first `noise_frames` frames
    (assumes leading silence / noise-only region).
    """
    frames = []
    for i in range(noise_frames):
        start = i * hop
        end   = start + frame_len
        if end > len(noisy):
            break
        frame = noisy[start:end] * np.hanning(frame_len)
        frames.append(np.abs(np.fft.rfft(frame)) ** 2)
    return np.mean(frames, axis=0)


def spectral_subtraction(noisy: np.ndarray, noise_psd: np.ndarray,
                         alpha: float = 1.0, beta: float = 0.002,
                         frame_len: int = 512, hop: int = 256,
                         method: str = 'basic') -> np.ndarray:
    """
    Spectral subtraction with overlap-add reconstruction.

    Parameters
    ----------
    noisy     : Input noisy signal
    noise_psd : Estimated noise power spectrum (from `estimate_noise_psd`)
    alpha     : Oversubtraction factor  (Berouti et al., 1979)
    beta      : Spectral floor (prevents musical noise)
    frame_len : FFT frame length
    hop       : Hop size (50 % overlap recommended)
    method    : 'basic'  – magnitude spectral subtraction (Boll 1979)
                'power'  – power spectral subtraction
                'oversubtract' – alpha/beta with half-wave rectification

    Returns
    -------
    enhanced : Enhanced speech (same length as `noisy`)
    """
    window   = np.hanning(frame_len)
    n_fft    = frame_len
    n_bins   = n_fft // 2 + 1

    # Pad signal so we get all frames
    n_frames = 1 + (len(noisy) - frame_len) // hop
    output   = np.zeros(len(noisy) + frame_len)
    norm_win = np.zeros(len(noisy) + frame_len)

    for i in range(n_frames):
        start = i * hop
        end   = start + frame_len
        frame = noisy[start:end] * window

        # FFT
        spec      = np.fft.rfft(frame, n=n_fft)
        magnitude = np.abs(spec)
        phase     = np.angle(spec)
        power     = magnitude ** 2

        # Suppress noise
        if method == 'basic':
            # Magnitude subtraction (Boll, 1979)
            noise_mag  = np.sqrt(noise_psd)
            enhanced_m = magnitude - alpha * noise_mag
            enhanced_m = np.maximum(enhanced_m, beta * noise_mag)   # floor

        elif method == 'power':
            # Power subtraction
            enhanced_p = power - alpha * noise_psd
            enhanced_p = np.maximum(enhanced_p, beta * noise_psd)
            enhanced_m = np.sqrt(enhanced_p)

        elif method == 'oversubtract':
            # Over-subtraction with half-wave rectification (Berouti, 1979)
            noise_mag  = np.sqrt(noise_psd)
            diff       = magnitude - alpha * noise_mag
            enhanced_m = np.where(diff >= beta * noise_mag, diff, beta * noise_mag)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Reconstruct frame
        enhanced_spec  = enhanced_m * np.exp(1j * phase)
        enhanced_frame = np.real(np.fft.irfft(enhanced_spec, n=n_fft))
        enhanced_frame *= window

        output[start:start + frame_len] += enhanced_frame
        norm_win[start:start + frame_len] += window ** 2

    # Normalise by overlap-add window
    norm_win = np.where(norm_win > 1e-8, norm_win, 1.0)
    output   = output[:len(noisy)] / norm_win[:len(noisy)]
    return output


# ──────────────────────────────────────────────────────────────────────────────
# 3.  EVALUATION METRICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_snr(clean: np.ndarray, enhanced: np.ndarray) -> float:
    """Signal-to-Noise Ratio (dB) measuring residual distortion."""
    noise  = clean - enhanced
    signal_power = np.mean(clean ** 2)
    noise_power  = np.mean(noise ** 2) + 1e-12
    return 10 * np.log10(signal_power / noise_power)


def compute_segsnr(clean: np.ndarray, enhanced: np.ndarray,
                   frame_len: int = 512, hop: int = 256) -> float:
    """Segmental SNR – per-frame SNR averaged over voiced frames."""
    n_frames = 1 + (len(clean) - frame_len) // hop
    snrs = []
    for i in range(n_frames):
        s = i * hop
        e = s + frame_len
        c_frame = clean[s:e]
        n_frame = c_frame - enhanced[s:e]
        sp = np.mean(c_frame ** 2)
        np_ = np.mean(n_frame ** 2) + 1e-12
        if sp > 1e-6:           # skip near-silent frames
            snrs.append(np.clip(10 * np.log10(sp / np_), -10, 35))
    return float(np.mean(snrs)) if snrs else 0.0


def compute_spectral_distortion(clean: np.ndarray, enhanced: np.ndarray,
                                frame_len: int = 512, hop: int = 256) -> float:
    """Log-spectral distortion (LSD) in dB."""
    n_frames = 1 + (len(clean) - frame_len) // hop
    lsds = []
    win  = np.hanning(frame_len)
    for i in range(n_frames):
        s = i * hop
        e = s + frame_len
        c = np.abs(np.fft.rfft(clean[s:e] * win)) + 1e-12
        h = np.abs(np.fft.rfft(enhanced[s:e] * win)) + 1e-12
        lsd = np.sqrt(np.mean((10 * np.log10(c / h)) ** 2))
        lsds.append(lsd)
    return float(np.mean(lsds))


# ──────────────────────────────────────────────────────────────────────────────
# 4.  VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────

def plot_spectrogram(ax, sig, fs, title, frame_len=512, hop=256, vmin=-80):
    """Plot short-time magnitude spectrogram on a given axes."""
    f, t, Sxx = signal.spectrogram(sig, fs=fs, window='hann',
                                   nperseg=frame_len, noverlap=frame_len - hop)
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    im = ax.pcolormesh(t, f / 1000, Sxx_db, shading='gouraud',
                       cmap='magma', vmin=vmin, vmax=0)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=8)
    ax.set_ylabel('Freq (kHz)', fontsize=8)
    ax.set_ylim(0, fs / 2000)
    plt.colorbar(im, ax=ax, label='dB', pad=0.02)


def run_experiment_and_plot():
    """
    Full experiment:
      1. Generate clean speech + noisy versions (5 dB, 10 dB, 15 dB SNR)
      2. Apply three methods
      3. Evaluate and plot
    """
    fs          = 16000
    frame_len   = 512
    hop         = 256
    snr_levels  = [5, 10, 15]
    methods     = ['basic', 'power', 'oversubtract']
    method_lbls = ['Basic SS', 'Power SS', 'Oversubtract']
    noise_type  = 'white'

    print("=" * 65)
    print("  Spectral Subtraction – Noise Removal in Speech Processing")
    print("=" * 65)

    clean = generate_synthetic_speech(duration=3.0, fs=fs)

    # ── Figure 1: Waveform & Spectrogram comparison (10 dB) ──────────────────
    noisy_10, _ = add_noise(clean, snr_db=10, noise_type=noise_type, fs=fs)
    noise_psd   = estimate_noise_psd(noisy_10, fs, noise_frames=8,
                                     frame_len=frame_len, hop=hop)

    enhanced = {m: spectral_subtraction(noisy_10, noise_psd,
                                        alpha=2.0, beta=0.005,
                                        frame_len=frame_len, hop=hop,
                                        method=m)
                for m in methods}

    fig1, axes = plt.subplots(5, 2, figsize=(14, 14))
    fig1.suptitle('Spectral Subtraction – Waveforms & Spectrograms (Input SNR = 10 dB)',
                  fontsize=12, fontweight='bold', y=1.01)

    t = np.arange(len(clean)) / fs
    signals = [('Clean Speech', clean),
               ('Noisy Input (10 dB SNR)', noisy_10)] + \
              [(f'{lbl} Enhanced', enhanced[m])
               for m, lbl in zip(methods, method_lbls)]

    for row, (title, sig) in enumerate(signals):
        ax_w = axes[row, 0]
        ax_s = axes[row, 1]
        ax_w.plot(t, sig, linewidth=0.6, color='steelblue')
        ax_w.set_title(title + ' – Waveform', fontsize=9, fontweight='bold')
        ax_w.set_xlabel('Time (s)', fontsize=8)
        ax_w.set_ylabel('Amplitude', fontsize=8)
        ax_w.set_xlim(0, t[-1])
        ax_w.grid(True, alpha=0.3)
        plot_spectrogram(ax_s, sig, fs, title + ' – Spectrogram',
                         frame_len=frame_len, hop=hop)

    plt.tight_layout()
    fig1.savefig('/home/claude/fig1_waveforms_spectrograms.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("\n[✓] Saved fig1_waveforms_spectrograms.png")

    # ── Figure 2: SNR Improvement across input SNR levels ───────────────────
    results = {m: {'seg_snr': [], 'lsd': [], 'global_snr': []} for m in methods}

    for snr_in in snr_levels:
        noisy, _ = add_noise(clean, snr_db=snr_in, noise_type=noise_type, fs=fs)
        n_psd    = estimate_noise_psd(noisy, fs, noise_frames=8,
                                     frame_len=frame_len, hop=hop)
        for m in methods:
            enh = spectral_subtraction(noisy, n_psd, alpha=2.0, beta=0.005,
                                       frame_len=frame_len, hop=hop, method=m)
            results[m]['seg_snr'].append(compute_segsnr(clean, enh, frame_len, hop))
            results[m]['lsd'].append(compute_spectral_distortion(clean, enh, frame_len, hop))
            results[m]['global_snr'].append(compute_snr(clean, enh))

    fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    fig2.suptitle('Performance Metrics vs Input SNR', fontsize=12, fontweight='bold')

    colors = ['#2196F3', '#4CAF50', '#FF5722']
    markers = ['o', 's', '^']

    for (m, lbl), c, mk in zip(zip(methods, method_lbls), colors, markers):
        ax1.plot(snr_levels, results[m]['global_snr'], marker=mk, color=c, label=lbl, lw=2)
        ax2.plot(snr_levels, results[m]['seg_snr'],    marker=mk, color=c, label=lbl, lw=2)
        ax3.plot(snr_levels, results[m]['lsd'],        marker=mk, color=c, label=lbl, lw=2)

    for ax, ylabel, title in [
        (ax1, 'Global SNR (dB)',      'Global SNR of Enhanced Signal'),
        (ax2, 'Segmental SNR (dB)',   'Segmental SNR of Enhanced Signal'),
        (ax3, 'Log-Spectral Dist (dB)', 'Log-Spectral Distortion (lower=better)'),
    ]:
        ax.set_xlabel('Input SNR (dB)', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(snr_levels)

    plt.tight_layout()
    fig2.savefig('/home/claude/fig2_metrics.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print("[✓] Saved fig2_metrics.png")

    # ── Figure 3: Alpha sensitivity (oversubtract method, 10 dB SNR) ─────────
    alphas     = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    alpha_snrs = []
    alpha_lsds = []
    noisy_10, _ = add_noise(clean, snr_db=10, noise_type=noise_type, fs=fs)
    n_psd10     = estimate_noise_psd(noisy_10, fs, noise_frames=8,
                                    frame_len=frame_len, hop=hop)

    for a in alphas:
        enh = spectral_subtraction(noisy_10, n_psd10, alpha=a, beta=0.005,
                                   frame_len=frame_len, hop=hop, method='oversubtract')
        alpha_snrs.append(compute_segsnr(clean, enh, frame_len, hop))
        alpha_lsds.append(compute_spectral_distortion(clean, enh, frame_len, hop))

    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig3.suptitle('Sensitivity to Oversubtraction Factor α (β=0.005, SNR=10 dB)',
                  fontsize=11, fontweight='bold')
    ax1.plot(alphas, alpha_snrs, 'o-', color='#2196F3', lw=2)
    ax1.set_xlabel('α (oversubtraction factor)', fontsize=9)
    ax1.set_ylabel('Segmental SNR (dB)', fontsize=9)
    ax1.set_title('SegSNR vs α', fontsize=9, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2.plot(alphas, alpha_lsds, 's-', color='#FF5722', lw=2)
    ax2.set_xlabel('α (oversubtraction factor)', fontsize=9)
    ax2.set_ylabel('Log-Spectral Distortion (dB)', fontsize=9)
    ax2.set_title('LSD vs α', fontsize=9, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig3.savefig('/home/claude/fig3_alpha_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print("[✓] Saved fig3_alpha_sensitivity.png")

    # ── Print numerical results table ────────────────────────────────────────
    print("\n  Results Table – Segmental SNR (dB) | (Input SNR / Method)")
    print(f"  {'Method':<18}", end="")
    for snr in snr_levels:
        print(f"  SNR={snr:2d}dB", end="")
    print()
    print("  " + "-" * 48)

    for m, lbl in zip(methods, method_lbls):
        print(f"  {lbl:<18}", end="")
        for i in range(len(snr_levels)):
            print(f"  {results[m]['seg_snr'][i]:8.2f}", end="")
        print()

    print("\n  Results Table – Log-Spectral Distortion (dB) [lower is better]")
    print(f"  {'Method':<18}", end="")
    for snr in snr_levels:
        print(f"  SNR={snr:2d}dB", end="")
    print()
    print("  " + "-" * 48)

    for m, lbl in zip(methods, method_lbls):
        print(f"  {lbl:<18}", end="")
        for i in range(len(snr_levels)):
            print(f"  {results[m]['lsd'][i]:8.2f}", end="")
        print()

    print("\n[✓] All experiments complete.\n")
    return results, snr_levels, methods, method_lbls


if __name__ == '__main__':
    run_experiment_and_plot()

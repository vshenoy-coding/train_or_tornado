# Video Title: Rain Storm and the sounds of a Tornado
# Uploader: TinCupTim
# URL: https://www.youtube.com/watch?v=4tnwJe6vvrE
# "Train" sound begins at 00:01:31 in the video, ends at 00:01:45
# sound is repeated from 00:01:51 to 00:02:03

# Colab Notebook: Tornado vs Train Audio Analysis
# Run cells sequentially. Adjust YOUTUBE_URL, START, DURATION as needed.

# 1) Install dependencies
!apt-get update -qq
!apt-get install -y -qq ffmpeg
!pip install -q yt-dlp librosa matplotlib scipy numpy soundfile

# 2) Parameters - change these if needed
YOUTUBE_URL = "https://www.youtube.com/watch?v=4tnwJe6vvrE"
START = "00:01:31"      # start time (HH:MM:SS) # 00:01:51
DURATION = 14           # seconds to extract    # 12
OUT_VIDEO = "temp_video.mp4"
OUT_WAV = "clip.wav"

# 3) Download the video (best quality) using yt-dlp
import os, shlex, subprocess, sys
print("Downloading video...")
subprocess.run(["yt-dlp", "-f", "best", "-o", OUT_VIDEO, YOUTUBE_URL], check=True)

# 4) Extract the audio clip with ffmpeg to a mono 44.1 kHz WAV (preserves low frequencies)
print("Extracting audio clip...")
ffmpeg_cmd = [
    "ffmpeg",
    "-ss", START,
    "-t", str(DURATION),
    "-i", OUT_VIDEO,
    "-vn",
    "-acodec", "pcm_s16le",
    "-ar", "44100",
    "-ac", "1",
    OUT_WAV,
    "-y"
]
subprocess.run(ffmpeg_cmd, check=True)

# 5) Load audio and show basic info
import soundfile as sf
import numpy as np
data, sr = sf.read(OUT_WAV)
print(f"Loaded {OUT_WAV}: {data.shape} samples, sample rate {sr} Hz, duration {len(data)/sr:.2f} s")

# 6) Compute and plot spectrogram tuned for low-frequency detail
import matplotlib.pyplot as plt
from scipy import signal

# Spectrogram parameters: large window for low-frequency resolution
nperseg = 4096
noverlap = int(nperseg * 0.75)

f, t, Sxx = signal.spectrogram(data, fs=sr, window='hann',
                               nperseg=nperseg, noverlap=noverlap,
                               scaling='density', mode='magnitude')

# Convert to dB
Sxx_db = 20 * np.log10(Sxx + 1e-12)

plt.figure(figsize=(12,5))
plt.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='magma')
plt.ylim(0, 5000)   # focus on 0-5 kHz; most distinguishing info is below ~2 kHz
plt.colorbar(label='Magnitude dB')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram (0-5 kHz)')
plt.show()

# 7) Diagnostics: low-frequency energy ratio, spectral centroid trend, tonal peaks, periodic impulses

# 7a) Low-frequency energy ratio (LF energy / total energy)
lf_band = (f <= 200)          # <= 200 Hz considered low-frequency for this purpose
total_energy = np.sum(Sxx, axis=0) + 1e-12
lf_energy = np.sum(Sxx[lf_band, :], axis=0) + 1e-12
lf_ratio = lf_energy / total_energy   # per time frame
median_lf_ratio = np.median(lf_ratio)
mean_lf_db = 20*np.log10(np.mean(lf_energy))

# 7b) Spectral centroid over time (track for Doppler-like sweep)
# centroid in Hz per frame
centroid = np.sum(f[:,None] * Sxx, axis=0) / (np.sum(Sxx, axis=0) + 1e-12)

# compute trend: linear fit slope of centroid vs time
coeffs = np.polyfit(t, centroid, 1)
centroid_slope = coeffs[0]   # Hz per second

# 7c) Tonal peaks detection: find narrowband peaks in averaged spectrum
avg_spec = np.mean(Sxx, axis=1)
from scipy.signal import find_peaks
peaks, props = find_peaks(avg_spec, height=np.max(avg_spec)*0.05, distance=5)
peak_freqs = f[peaks]
peak_heights = props['peak_heights']

# 7d) Periodic impulses detection via envelope autocorrelation
# compute amplitude envelope
analytic = signal.hilbert(data)
envelope = np.abs(analytic)
# downsample envelope for speed
env_ds = signal.resample(envelope, int(len(envelope)/10))
env_ds = env_ds - np.mean(env_ds)
autocorr = np.correlate(env_ds, env_ds, mode='full')
mid = len(autocorr)//2
autocorr_pos = autocorr[mid:]
# find peaks in autocorr to detect periodicity
ac_peaks, ac_props = find_peaks(autocorr_pos, height=np.max(autocorr_pos)*0.2, distance=10)
periods = []
if len(ac_peaks) > 0:
    # convert lag samples to seconds (approx)
    ds_rate = sr / 10.0
    periods = ac_peaks / ds_rate

# 8) Print diagnostic summary with interpretation heuristics
print("\n--- Diagnostic summary ---")
print(f"Median low-frequency energy ratio (<=200 Hz) per frame: {median_lf_ratio:.3f}")
print(f"Mean LF energy (dB): {mean_lf_db:.1f} dB")
print(f"Spectral centroid slope: {centroid_slope:.1f} Hz/s (positive means centroid rising over time)")
print(f"Detected tonal peak frequencies (Hz): {np.round(peak_freqs,1).tolist()}")
if len(periods)>0:
    print(f"Detected periodic impulses with approximate periods (s): {np.round(periods,3).tolist()}")
else:
    print("No strong periodic impulses detected in the amplitude envelope autocorrelation.")

# Heuristic interpretation rules
print("\n--- Heuristic interpretation ---")
if median_lf_ratio > 0.35:
    print("- Strong low-frequency content detected. This supports a tornado-like roar (aerodynamic/infrasound).")
else:
    print("- Low-frequency content is not dominant. This weakens the tornado-roar hypothesis.")

if abs(centroid_slope) > 50:
    print("- Significant centroid slope detected. A clear approach-then-recede Doppler sweep may be present, which can indicate a moving source such as a train.")
else:
    print("- Little centroid trend. Lack of a clear Doppler sweep favors a stationary or persistent source like a tornado vortex.")

if len(peak_freqs) > 0 and np.any(peak_freqs < 1000):
    print("- Narrowband tonal peaks found at low-mid frequencies. These can be caused by vortex shedding, subvortices, or structural resonance and may explain a blade-like sound.")
else:
    print("- No prominent low-frequency tonal peaks detected.")

if len(periods) > 0 and any((p > 0.05 and p < 1.0) for p in periods):
    print("- Periodic impulses detected in the envelope. Regular impulses can indicate mechanical sources (wheel clacks) typical of trains.")
else:
    print("- No clear regular periodic impulses detected. This reduces the likelihood of a train wheel/track signature.")

# 9) Optional: plot centroid over time and LF ratio
plt.figure(figsize=(10,3))
plt.plot(t, centroid, label='Spectral centroid (Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Hz')
plt.title('Spectral centroid over time')
plt.grid(True)
plt.show()

plt.figure(figsize=(10,2))
plt.plot(t, lf_ratio, label='LF ratio (<=200 Hz)')
plt.ylim(0,1)
plt.xlabel('Time (s)')
plt.title('Low-frequency energy ratio over time')
plt.grid(True)
plt.show()

# 10) Save results to files for download if desired
#import json
#results = {
#    "median_lf_ratio": float(median_lf_ratio),
#    "mean_lf_db": float(mean_lf_db),
#    "centroid_slope_hz_per_s": float(centroid_slope),
#   "peak_freqs_hz": [float(x) for x in peak_freqs],
#    "periods_s": [float(x) for x in periods]
#}
#with open("analysis_summary.json","w") as f:
#    json.dump(results, f, indent=2)
#print("\nSaved analysis_summary.json and clip.wav in the notebook workspace.")

# Texture, rhythm, and spectral evolution of a sound needs to be analyzed in
# addition to volume. 

# A train versus a tornado is harmonic versus stochastic noise,
# predictable rhythm at regular second intervals and harmonic noise versus ultrafast rhythm at millisecond intervals and broadband noise,
# a point source pitch rising on approach and falling on departure versus 
# a volume source erratically sharpening and becoming higher frequency as wind speed increases and dulling and becoming lower frequency as the core moves away or is blocked by a building,
# energy that is mostly audible and vibrates the ground versus a lot of energy in the infrasound that you can only feel, not hear and which induces vibrations in the air 
# and any objects (including you).

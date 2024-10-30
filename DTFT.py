import numpy as np
import matplotlib.pyplot as plt

Fs = 1000
T = 1
f = 50

t = np.linspace(0, T, int(T*Fs), endpoint=False)
x = np.sin(2*np.pi*f*t)

X = np.fft.fft(x)
freqs = np.fft.fftfreq(len(x), 1/Fs)

magnitude_spectrum = np.abs(X)
phase_spectrum = np.angle(X)

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(freqs, magnitude_spectrum)
plt.title('Magnitude Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

plt.subplot(2, 1, 2)
plt.plot(freqs, phase_spectrum)
plt.title('Phase Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (radians)')

plt.tight_layout()
plt.show()


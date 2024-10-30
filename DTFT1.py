import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,4,5,7,1.5,-1.73])
l = len(x)
n = np.arange(l)
omega = np.linspace(-np.pi,np.pi,1000)  
X = np.array([np.sum(x[n] * np.exp(-1j * w * n)) for w in omega])

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(omega, np.abs(X))
plt.title('Magnitude of DTFT')
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('Magnitude')
plt.grid(False)
plt.subplot(2, 1, 2)
plt.plot(omega, np.angle(X))
plt.title('Phase of DTFT')
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('Phase (radians)')
plt.grid()
plt.tight_layout()
plt.savefig('plot.png', dpi=300)
plt.show()


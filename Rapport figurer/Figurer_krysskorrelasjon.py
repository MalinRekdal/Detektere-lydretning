import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

n = np.linspace(-5, 5, 100)

s1 = np.sinc(n)
s2 = np.sinc(n+1)

corr = signal.correlate(s1, s2, mode='same')
corr_ax = np.linspace(-len(corr)//(2*10), len(corr)//(2*10), len(s1))

plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16)

plt.xlabel("Tid [ms]", fontsize = 16)
plt.ylabel("Signalamplitude", fontsize = 16)
plt.plot(n, s1)
plt.plot(n, s2)
plt.legend(['Signal x','Signal y'])
plt.show()

plt.xlabel("Tid [ms]", fontsize = 16)
plt.ylabel("Krysskorrelasjon", fontsize = 16)
plt.plot(corr_ax, corr, color='green')


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

Fs = 5000;
f = 100;
delta_t = 50; 

n = np.arange(-100,100)

signal1 = np.sinc(2*np.pi*f/Fs*n)
signal2 = np.roll(signal1, delta_t) # Moves delta_t steps, 1 step is 1/Fs
signal3 = np.roll(signal1, delta_t + 20)

corr = signal.correlate(signal2, signal1, mode='same')

# rekkefølga av signala --> endrar retninga den er forskøvet

abs_corr = np.abs(corr)
max_corr = max(abs_corr)
index_of_max_corr = list(abs_corr).index(max_corr)
print(index_of_max_corr)
max_lag = n[index_of_max_corr]
timedelay = max_lag/Fs

print("max value of correlation: ", max_corr)
print("lag value of max: ", max_lag)
print("Tidsforskjellen er: ", timedelay*1000, " ms")


plt.plot(n, signal1, n, signal2) 
plt.title(" ")
plt.xlabel("lag")
plt.ylabel(" ")
plt.show()

plt.plot(n, corr) 
plt.title(" ")
plt.xlabel("lag")
plt.ylabel(" ")
plt.show()





import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.interpolate as interpolate

filename = 'målingar/v-35_m3'

upsampling_factor = 4

def raspi_import(path, channels=5):
    """
    Import data produced using adc_sampler.c.
    Returns sample period and ndarray with one column per channel.
    Sampled data for each channel, in dimensions NUM_SAMPLES x NUM_CHANNELS.
    """

    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype=np.uint16)
        data = data.reshape((-1, channels))
    return sample_period, data


def scale_and_remove_DC(data):
    """
    Scales data from digital to analog value, and removes DC component.
    Returnes the new data. 
    Uses ADC_resolution that is defined, and average.
    """
    data = data*(0.8*10**(-3)) # Scales unit to analog values
    data = signal.detrend(data, axis=0)  # removes DC component for each channel
    return data


sample_period, data = raspi_import(filename) # Import data from bin file

sample_period *= 1e-6  # Change unit to micro seconds
fs = 1/sample_period

data = scale_and_remove_DC(data)


num_of_samples = data.shape[0]  # returns shape of matrix

t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples) # Generate time axis
freq = np.fft.fftfreq(n=num_of_samples, d=sample_period) # Generate frequency axis

spectrum = np.fft.fft(data, axis=0)  # takes FFT of all channels


# PLOT: (Use data[:,n] to get channel n)
plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16)
plt.tight_layout()

plt.subplot(2, 1, 1)
plt.xlabel("Time [s]", fontsize = 20)
plt.ylabel("Voltage [V]", fontsize = 20)
plt.plot(t, data[:,0:3]) # evt data[:,0]
plt.legend(['Mikrofon 1', 'Mikrofon 2', 'Mikrofon 3'],  fontsize = 16, loc = 'upper right')

plt.subplot(2, 1, 2)
plt.xlabel("Frequency [Hz]", fontsize = 20)
plt.ylabel("Power [dB]", fontsize = 20)
plt.plot(freq, 20*np.log10(np.abs(spectrum[:,0:3]))) # get the power spectrum
plt.legend(['Mikrofon 1', 'Mikrofon 2', 'Mikrofon 3'],  fontsize = 16, loc = 'upper right')

plt.show()


"""
# Signals for testing
f = 100
fs = 5000
l = 50
l2 = 70
t = np.arange(-100, 100)

s1 = np.sinc(2*np.pi*f/fs*t)
s2 = np.roll(s1, l)
s3 = np.roll(s1, l2)
"""

s1 = data[:,0]
s2 = data[:,1]
s3 = data[:,2]


t_upsampled = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples*upsampling_factor)

# Adjust signals to new upsampled axis.
s1 = np.interp(t_upsampled, t, s1)
s2 = np.interp(t_upsampled, t, s2)
s3 = np.interp(t_upsampled, t, s3)

# Find correlation
corr_21 = signal.correlate(s2, s1, mode='same')
corr_31 = signal.correlate(s3, s1, mode='same')
corr_32 = signal.correlate(s3, s2, mode='same')

# Find autocorrelation
autocorr1 = signal.correlate(s1, s1, mode='same')
autocorr2 = signal.correlate(s2, s2, mode='same')
autocorr3 = signal.correlate(s3, s3, mode='same')

corr_ax = np.linspace(-len(corr_21)//2, len(corr_21)//2, len(s1))

# Find the sample and time where the correlation has a maximum
n_21 = corr_ax[np.where(corr_21 == max(abs(corr_21)))[0][0]] 
print("n21:", n_21, "samples")
print("tau21:", (n_21/(fs*upsampling_factor))*(10**6), "mikro sekund")

n_31 = corr_ax[int(np.where(corr_31 == max(abs(corr_31)))[0][0])]
print("n31:", n_31, "samples")
print("tau31:", (n_31/(fs*upsampling_factor))*(10**6), "mikro sekund")

n_32 = corr_ax[int(np.where(corr_32 == max(abs(corr_32)))[0][0])]
print("n32:", n_32, "samples")
print("tau32:", (n_32/(fs*upsampling_factor))*(10**6), "mikro sekund")

# Uses arctan2(numerator, denumerator) because we want to get the right quadrant directly
theta = np.arctan2(np.sqrt(3)*(n_21+n_31),(n_21-n_31-2*n_32))

print("\n")
print("Theta i radianer: ", theta)
print("Theta i grader:", np.degrees(theta))

# Correlation plots:
plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16)
plt.tight_layout()

plt.subplot(2, 2, 1)
plt.xlabel("l", fontsize = 16)
plt.ylabel("Krysskorrelasjon $r_{21}$", fontsize = 16)
plt.plot(corr_ax, corr_21)

plt.subplot(2, 2, 2)
plt.xlabel("l", fontsize = 16)
plt.ylabel("Krysskorrelasjon $r_{31}$", fontsize = 16)
plt.plot(corr_ax,corr_31) 

plt.subplot(2, 2, 3)
plt.xlabel("l", fontsize = 16)
plt.ylabel("Krysskorrelasjon $r_{32}$", fontsize = 16)
plt.plot(corr_ax, corr_32) 
plt.show()

# Autocorrelation plots:
plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16)
plt.tight_layout()

plt.subplot(2, 2, 1)
plt.xlabel("l", fontsize = 16)
plt.ylabel("Autokorrelasjon $r_{11}$", fontsize = 16)
plt.plot(corr_ax, autocorr1) 

plt.subplot(2, 2, 2)
plt.xlabel("l", fontsize = 16)
plt.ylabel("Autokorrelasjon $r_{22}$", fontsize = 16)
plt.plot(corr_ax,autocorr2) 

plt.subplot(2, 2, 3)
plt.xlabel("l", fontsize = 16)
plt.ylabel("Autokorrelasjon $r_{33}$", fontsize = 16)
plt.plot(corr_ax, autocorr3) 
plt.show()



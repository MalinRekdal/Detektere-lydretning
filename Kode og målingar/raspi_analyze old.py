import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


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


# Import data from bin file
sample_period, data = raspi_import('lab2test1')

#data = signal.detrend(data, axis=0)  # removes DC component for each channel
sample_period *= 1e-6  # change unit to micro seconds
fs = 1/sample_period

# Generate time axis
num_of_samples = data.shape[0]  # returns shape of matrix
t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)


data = data[:,0:3] # velger 3 første signala
ADC_resolution = 2**11
data = data*3.3/ADC_resolution # skalerar
average = np.average(data) 
data -= average # Trekker fra gjennomsnittet 

# Generate frequency axis and take FFT
freq = np.fft.fftfreq(n=num_of_samples, d=sample_period)
spectrum = np.fft.fft(data, axis=0)  # takes FFT of all channels


# Plot the results in two subplots
# NOTICE: This lazily plots the entire matrixes. All the channels will be put into the same plots.
# If you want a single channel, use data[:,n] to get channel n
plt.subplot(2, 1, 1)
plt.title("Time domain signal")
plt.xlabel("Time [us]")
plt.ylabel("Voltage")
plt.plot(t, data) # evt data[:,0]

plt.subplot(2, 1, 2)
plt.title("Power spectrum of signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power [dB]")
plt.plot(freq, 20*np.log10(np.abs(spectrum))) # get the power spectrum

plt.show()

"""
f = 100
fs = 5000
l = 30
l2 = 50
n = np.arange(-100, 100)


s1 = np.sinc(2*np.pi*f/fs*n)
s2 = np.roll(s1, l)
s3 = np.roll(s1, l2)
"""
# plt.plot(n, s2, n, s1)

s1 = data[:,0]
s2 = data[:,1]
s3 = data[:,2]

corr_21 = signal.correlate(s2, s1, mode='same')
corr_31 = signal.correlate(s3, s1, mode='same')
corr_32 = signal.correlate(s3, s2, mode='same')

Corr_21 = list(corr_21)
Corr_31 = list(corr_31)
Corr_32 = list(corr_32)

# y = max(abs(corr))
# print(y)

# corr_list = list(corr)
i_21 = Corr_21.index(max(abs(corr_21)))
delta_l_21 = t[i_21]
print("lag_21:", delta_l_21)
delta_t_21 = delta_l_21/fs
print("delta_t_21:", delta_t_21)

i_31 = Corr_31.index(max(abs(corr_31)))
delta_l_31 = t[i_31]
print("lag_21:", delta_l_31)
delta_t_31 = delta_l_31/fs
print("delta_t_31", delta_t_31)

i_32 = Corr_32.index(max(abs(corr_32)))
delta_l_32 = t[i_32]
print("lag_32:", delta_l_32)
delta_t_32 = delta_l_32/fs
print("delta_t_32", delta_t_32)

theta = np.arctan(np.sqrt(3)*(delta_t_21+delta_t_31)/(delta_t_21-delta_t_31-2*delta_t_32))
theta_l = np.arctan(np.sqrt(3)*(delta_l_21+delta_l_31)/(delta_l_21-delta_l_31-2*delta_l_32))

# BRUK ARCTAN2(nemner, teljer)!!

print("Theta: ", theta)
print("Grader:", theta*360/(2*np.pi))
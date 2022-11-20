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
sample_period, data = raspi_import('Lab2test2grade240')

data = signal.detrend(data, axis=0)  # removes DC component for each channel
sample_period *= 1e-6  # change unit to micro seconds

# Generate time axis
num_of_samples = data.shape[0]  # returns shape of matrix
t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)

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
plt.plot(t[100:300], data[100:300])

plt.subplot(2, 1, 2)
plt.title("Power spectrum of signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power [dB]")
plt.plot(freq, 20*np.log10(np.abs(spectrum))) # get the power spectrum

plt.show()

t2 = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples*16)



s0 = data[:,0]
s1 = data[:,1]
s2 = data[:,2]

s1 = np.interp(t2, t, s1)
s2 = np.interp(t2, t, s2)
s0 = np.interp(t2, t, s0)

plt.subplot(2, 1, 2)
plt.title("Power spectrum of signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power [dB]")
plt.plot(t[100:300], s1[100:300], t[100:300], s0[100:300])

def lag_finder(s1, s2):
    n = len(s1)

    corr = np.correlate(s2,s1,'same')
    len_corr = len(corr)

    delay_arr = np.linspace(-len_corr//2, len_corr//2, n)
    delay_sampler = np.argmax(corr)-len_corr//2
    delay_sekunder = (delay_sampler)*1/31250
    
    plt.figure()
    plt.stem(delay_arr, corr)
    plt.xlim(-10,10)
    plt.xlabel('Lag')
    plt.ylabel('Correlation coeff')
    plt.show()
    
    return delay_sampler, delay_sekunder

delay_sampler_01, delay_sekunder_01 = lag_finder(s0,s1)
print('s1 er ' + str(delay_sampler_01) + ' sampler bak s0')
print('s1 er ' + str(delay_sekunder_01) + ' s bak s0')
print(' ')

delay_sampler_02, delay_sekunder_02 = lag_finder(s0,s2)
print('s2 er ' + str(delay_sampler_02) + ' sampler bak s0')
print('s2 er ' + str(delay_sekunder_02) + ' s bak s0')
print(' ')

delay_sampler_12, delay_sekunder_12 = lag_finder(s1,s2)
print('s2 er ' + str(delay_sampler_12) + ' sampler bak s1')
print('s2 er ' + str(delay_sekunder_12) + ' s bak s1')
print(' ')

vinkel_radianer = np.arctan(np.sqrt(3)*(delay_sekunder_01+delay_sekunder_02)/(delay_sekunder_01-delay_sekunder_02+2*delay_sekunder_12))
vinkel_grader = vinkel_radianer*180/np.pi
print('Vinkel: ' + str(vinkel_grader) + ' grader')
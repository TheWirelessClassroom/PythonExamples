# imports of libraries
import matplotlib.pyplot as plt
import numpy as np
import pylab as pyl
from scipy.stats import rayleigh
plt.close("all")

def modulate_4QAM(bits):
    b_temp = np.reshape(bits * 2 - 1, (-1, 2)) # reshape bits in [0,1] to [-1,1]
    x = 1 / np.sqrt(2) * (b_temp[:, 0] + 1j * b_temp[:, 1])
    return x

def detecting_4QAM(received_signal):
    # detecting (slicing) and de-mapping
    received_bits = np.zeros((len(received_signal), 2))
    received_bits[:, 0] = np.real(received_signal) > 0
    received_bits[:, 1] = np.imag(received_signal) > 0
    received_bits = np.reshape(received_bits, (-1,))
    return received_bits


if __name__ == '__main__':
    # this is the main function

    # set simulation parameters here
    QAM_order=4 
    number_of_bits = int(np.log2(QAM_order)) #number of bits to send one QAM symbol
    number_of_realizations = int(10000) #simulate this amount of packets, increase this number for a smoother BER curve
    points = int(31) 
    SNR = np.linspace(-10, 20, points)  # SNR in dB

    # init result variables
    transmitted_symbols=np.zeros((points,number_of_realizations),dtype=complex)
    BER = np.zeros((2, number_of_realizations, len(SNR))) #AWGN
    channels=np.zeros((points,number_of_realizations),dtype=complex)
    
    print("Simulation started....")
    
    # simulation loop over SNR and random realizations
    for SNR_index, current_SNR in enumerate(SNR):
        for realization in range(number_of_realizations):
            
            #Generate data
            b = pyl.randint(0, 2, int(number_of_bits)) # generate random bits
            x = modulate_4QAM(b) # map bits to complex symbols
            transmitted_symbols[SNR_index,realization]=x #save transmitted symbols for plots


            #Add noise
            noisePower=10**(-current_SNR/20) # calculate the noise power for a given SNR value
            noise = (noisePower)*1/np.sqrt(2)*(pyl.randn(len(x))+1j*pyl.randn(len(x)))#generate noise
            
            #Generate flat rayleigh fading channel, i.e. a one tap rayleigh fading channel
            h=1/np.sqrt(2)*(pyl.randn(1)+1j*pyl.randn(1))
            channels[SNR_index,realization]=h #save all channel realizations for plots

            y_AWGN =x+noise # add the noise to the AWGN signal
            y_rayleigh=h*x+noise # add the noise to the flat rayleigh fading signal
            
            
            #Receive filter to combat the fading
            #This is not done for the AWGN, because there is no fading (i.e. h=1) and the noise is random/unkown
            y_est_rayleigh=y_rayleigh/h
            
            
            # detecting (slicing) and de-mapping
            b_received_AWGN = detecting_4QAM(y_AWGN)
            b_received_rayleigh = detecting_4QAM(y_est_rayleigh)

            # calculate bit errors for AWGN and flat rayleigh fading channel
            BER[0, realization, SNR_index] = sum(abs(b - b_received_AWGN)) / number_of_bits #AWGN
            BER[1, realization, SNR_index] = sum(abs(b - b_received_rayleigh)) / number_of_bits #rayleigh fading
            
        
        print("%.2f %%" % (100*SNR_index/len(SNR)))    
    
    print("Simulation finished")
    # calculate mean BER over realizations
    mean_BER = np.mean(BER, axis=1)

    # plot BER
    myfig = plt.figure()
    plt.semilogy(SNR, mean_BER[0], marker='.') #AWGN
    plt.semilogy(SNR, mean_BER[1], marker='.') #Flat rayleigh fading channel
    plt.grid(True)
    plt.axis([-5, 20, 1e-3, 1])
    plt.ylabel('BER')
    plt.xlabel('SNR in (dB)')
    plt.title('BER over SNR for a 4QAM with AWGN and flat rayleigh fading')
    plt.legend(['AWGN','Flat Rayleigh Fading'])
    plt.show()
    plt.savefig('4QAM_AWGN_with_rayleigh_BER_over_SNR.eps',format='eps')
    
    # plot amplitude distribution of the rayleigh fading for one SNR value
    r=np.abs(channels[10,:]) #Calculate amplitude distribution for the SNR=0dB
    rv = rayleigh(loc=0,scale=1/np.sqrt(2)) #Generate theoretical rayleigh distribution
    x = np.linspace(r.min(),r.max(), 1000)
    
    myfig = plt.figure()
    plt.hist(r,density=True, histtype='stepfilled',bins=50, alpha=1) #Plot histogram
    plt.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf') #Plot theoretical distribution
    plt.grid(True)
    plt.ylabel('PDF')
    plt.xlabel('Magnitude')
    plt.title('Probability Density Function of the rayleigh fading distribution')
    plt.legend(['Theoretical','Simulated'])
    plt.show()
    plt.savefig('Rayleigh_distribution_for_0dB.eps',format='eps')
    



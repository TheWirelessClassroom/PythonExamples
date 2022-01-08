# imports of libraries
#from pylab import *  # this enables a Matlab-like syntax in python
#close('all') # close all figures
import matplotlib.pyplot as plt
import numpy as np
import pylab as pyl
from scipy import special
plt.close("all")

def modulate_BSPK(bits):
    x = np.reshape(bits * 2 - 1, (-1, 1)) # reshape bits in [0,1] to [-1,1]
    return x

def detecting_BPSK(received_signal):
    # detecting (slicing) and de-mapping
    received_bits = int(np.real(received_signal) > 0)
    return received_bits


if __name__ == '__main__':
    # this is the main function

    # set simulation parameters here
    PSK_order=2 #BSPK
    number_of_bits = int(np.log2(PSK_order)) #number of bits to send one QAM symbol
    number_of_realizations = int(10000) #simulate this amount of packets, increase this number for a smoother BER curve
    points = int(31) 
    SNR = np.linspace(-10, 20, points)  # SNR in dB

    # init result variables
    BER = np.zeros((1, number_of_realizations, len(SNR))) #AWGN
    
    print("Simulation started....")
    
    # simulation loop over SNR and random realizations
    for SNR_index, current_SNR in enumerate(SNR):
        for realization in range(number_of_realizations):
            
            #Generate data
            b = pyl.randint(0, 2, int(number_of_bits)) # generate random bits
            x = modulate_BSPK(b) # map bits to complex symbols


            #Add noise
            noisePower=10**(-current_SNR/20) # calculate the noise power for a given SNR value
            noise = (noisePower)*1/np.sqrt(2)*(pyl.randn(len(x))+1j*pyl.randn(len(x)))#generate noise

            y_AWGN =x+noise # add the noise to the signal

            # detecting (slicing) and de-mapping
            b_received = detecting_BPSK(y_AWGN)
            

            # calculate bit errors
            BER[0, realization, SNR_index] = sum(abs(b - b_received)) / number_of_bits
        print("%.2f %%" % (100*SNR_index/len(SNR)))    
    
    print("Simulation finished")
            

    # calculate mean BER over realizations
    mean_BER = np.mean(BER, axis=1)
    
    #calculate theoretical BER for BPSK
    SNR_lin=10**(SNR/10)
    mean_BER_theoretical=1/2*special.erfc(np.sqrt(SNR_lin))

    # plot BER
    myfig = plt.figure()
    plt.semilogy(SNR, mean_BER[0], marker='.',label='Simulated')
    plt.semilogy(SNR,mean_BER_theoretical,marker='*',label='Theoretical')
    plt.grid(True)
    plt.axis([-5, 20, 1e-3, 1])
    plt.ylabel('BER')
    plt.xlabel('SNR in (dB)')
    plt.title('BER over SNR for BSPK with AWGN')
    plt.legend()
    plt.show()
    plt.savefig('BPSK_AWGN_BER_over_SNR.eps',format='eps')
    

import numpy as np
import matplotlib.pyplot as plt

def butterworth_lp(signal, t, cutoff, npoles=4, plot=False):
    """
    Low-pass Butterworth filter.
    
    signal: signal to be filtered
    t: time vector
    npoles: number of poles (integer)
    cutoff: cutoff frequency [Hz]
    plot: plot or not
    """
    
    n=float(npoles)
    wc=2.0*np.pi*cutoff
    
    s=np.zeros(npoles,dtype='complex64')
    
    #- Determine the poles. ---------------------------------------------------
    for k in np.arange(0,npoles): s[k]=wc*np.exp(1j*np.pi*(2.0*(k+1)+n-1)/(2.0*n))
        
    #- Compute and plot impulse response. -------------------------------------
    Pi=np.ones(npoles,dtype='complex64')
    h=np.zeros(len(t),dtype='complex64')
    
    for i in range(npoles):
        for k in range(npoles):
            if not i==k: Pi[i]=Pi[i]*(s[i]-s[k])
                
    Pi=Pi/(wc**n)
    
    for k in range(npoles): h+=np.exp(s[k]*t)/Pi[k]
        
    if plot:
        plt.plot(t,np.real(h))
        plt.show()
        
    f_signal=np.real(np.convolve(h,signal))*(t[1]-t[0])
    return f_signal[0:len(t)]


def butterworth_hp(signal, t, cutoff, npoles=4, plot=False):
    """
    High-pass Butterworth filter.
    
    signal: signal to be filtered
    t: time vector
    npoles: number of poles (integer)
    cutoff: cutoff frequency [Hz]
    plot: plot or not
    """
    
    n=float(npoles)
    wc=2.0*np.pi*cutoff
    
    s=np.zeros(npoles,dtype='complex64')
    
    #- Determine the poles. ---------------------------------------------------
    for k in np.arange(0,npoles): s[k]=-wc*np.exp(1j*np.pi*(n+1.0-2.0*(k+1.0))/(2.0*n))
        
    #- Compute and plot impulse response. -------------------------------------
    Pi=np.ones(npoles,dtype='complex64')
    h=np.zeros(len(t),dtype='complex64')
    
    for i in range(npoles):
        for k in range(npoles):
            if not i==k: Pi[i]=Pi[i]*(1.0-s[i]/s[k])
                
    Pi=Pi/(wc**n)
    
    for k in range(npoles): h-= s[k]**n * np.exp(s[k]*t)/Pi[k]
    for k in range(1,npoles-1): h[k]=(h[k+1]-h[k])/(t[1]-t[0])
    h[0]=h[0]-np.sum(h)
    m=np.max(np.abs(np.fft.fft(h)))*(t[1]-t[0])
    h=h/m
        
    if plot:
        plt.plot(t,np.real(h))
        plt.show()
        
    f_signal=np.real(np.convolve(h,signal))*(t[1]-t[0])
    return f_signal[0:len(t)]
"""
 postprocessing tools for FIF and MvFIF

 Dependencies : numpy, scipy, numba

 Authors: 
    Python version: Emanuele Papini - INAF (emanuele.papini@inaf.it) 
    Original matlab version: Antonio Cicone - university of L'Aquila (antonio.cicone@univaq.it)
"""
import numpy as np
from numba import jit, prange

from scipy.integrate import trapz as integrate


num_cpus=16

def aggregate_IMCs(imfs,freqs,freq_ranges,return_mean_freq = False):

    
    agimfs=np.zeros((len(freq_ranges)+1,imfs.shape[1],imfs.shape[2]))
    mean_freqs = np.zeros(agimfs.shape[0])
    f0 = 0.
    ii = 0
    for ifr in freq_ranges:
        mm = (freqs<=ifr) * (freqs>f0)
        f0=ifr     
        if np.sum(mm) > 0 : 
            agimfs[ii,:,:] = np.sum(imfs[mm,:,:],axis=0)
            mean_freqs[ii] = np.mean(freqs[mm])
        ii+=1
    
    mm = (freqs>f0)
    
    if np.sum(mm) > 0 : 
        agimfs[ii,:,:] = np.sum(imfs[mm,:,:],axis=0)
        mean_freqs[ii] = np.mean(freqs[mm])
    
    return (agimfs, mean_freqs) if return_mean_freq else agimfs


def check_orthogonality(imfs,periodic=True, plot=False):
    """
    WORKS ONLY FOR 1D/2D IMCs. MUST BE IMPLEMENTED TO WORK WITH ND IMCs

    """
    from scipy.integrate import trapz as integrate

    ndim = len(imfs.shape[1:])

    nimf=imfs.shape[0]

    orto = np.zeros([nimf,nimf])
    
    if periodic :
        imfst = np.zeros( (nimf,) + tuple(np.asarray(imfs.shape[1:])+1) )

        #THIS IS THE PART THAT MUST BE GENERALIZED TO ND
        imfst[:,0:-1,0:-1] =imfs[:,:,:]
        imfst[:,-1,0:-1] = imfs[:,-1,:]
        imfst[:,0:-1,-1] = imfs[:,:,-1]
        imfst[:,-1,-1] = imfs[:,0,0]


    else : imfst = imfs


    amps = np.asarray([np.sqrt(integrate(integrate(imfst[i]**2))) for i in range(nimf)])


    for i in range(nimf):
        for j in range(i+1):
            orto[i,j] = integrate(integrate(imfst[i] * imfst[j]))/amps[i]/amps[j]


    if plot:
        import pylab as plt
        from matplotlib import cm
        plt.ion()
        plt.figure(figsize=(7,5.67))
        plt.imshow(orto,cmap=cm.Blues, extent=[0.5,nimf+.5,nimf+.5,0.5])
        for j in range(nimf):
            for i in range(j+1):
                plt.text(i+1,j+1,'{:.2f}'.format( np.abs(orto[j,i])),ha='center',va='center',fontsize=11,color='darkorange')
                plt.xlabel(r'$j$',fontsize=14)
                plt.ylabel(r'$i$',fontsize=14)
                plt.title(r'$\langle \mathrm{IMF}_i,\mathrm{IMF}_j\rangle $')






    return orto

def orthogonalize(imfs,threshold = 0.6, **kwargs):
    """
    TESTED ONLY FOR 2D IMCS. IT SHOULD WORK WITH ND IMCS, PROVIDED THAT check_orthogonality HAS BEEN 
    IMPLEMENTED TO WORK WITH ND IMCS

    """

    orto =check_orthogonality(imfs,**kwargs)
    ilow = np.arange(imfs.shape[0]-1)+1
    lowdiag = orto[ilow,ilow-1]
    imfst = imfs
    while lowdiag.max() >= threshold:
        i = lowdiag.argmax()
        imt = imfst[i,...]
        imfst = np.concatenate((imfst[0:i,...],imfst[i+1:,...]),axis=0)
        imfst[i-1,:,:] += imt
        orto =check_orthogonality(imfst,**kwargs)
        ilow = np.arange(imfst.shape[0]-1)+1
        lowdiag = orto[ilow,ilow-1]
         
    return imfst



#def _postprocessing_IMF(IMF,eps = 1e-3):
#    """
#    %
#    % postprocessing of the IMF
#    %  INPUT:
#    %       IMF : intrinsic mode functions as returned by MIF
#    %       epsi: threshold parameter: if the difference between the average
#    %             frequency of two IMFs falls below epsi, then sum the two IMFS
#    %             into one only IMF.
#    %
#    %  OUTPUT:
#    %
#    %   IMF_pp: new processed IMFs (maybe less than the original imfs)
#    %   fim_pp: averaged frequencies of each IMF
#    %
#    """
#
#    if len(IMF.shape) == 3:
#        IMF_pp = []
#        frq_pp = []
#        for i in range(IMF.shape[0]):
#            IMF_ppx,frq_ppx = _postprocessing_IMF(IMF[i,...].squeeze(),eps =eps)
#            IMF_pp.append(IMF_ppx)
#            frq_pp.append(frq_ppx)
#        return IMF_pp, frq_pp
#    else:
#        N0 = len(IMF[0,:])
#        M0 = len(IMF[:,0])
#        fim0 = np.zeros(M0)
#        for i in range(M0):
#            maxmins = Maxmins_v3_6(IMF[i,:])
#            fim0[i]=1/(2*np.round(N0/len(maxmins)))
#        
#        IMF_pp=[IMF[0,:].copy()]
#        fim_pp=[fim0[0]]
#        for i in range(1,M0):
#            if (np.abs(fim0[i]-fim0[i-1]) < eps ) :
#                IMF_pp[-1] += IMF[i,:]
#            else:
#                IMF_pp.append(IMF[i,:].copy())
#                fim_pp.append(fim0[i])
#        
#        #IMF_pp.append(IMF[-1,:])
#        #fim_pp.append(fim0[-1])
#        return np.array(IMF_pp), np.array(fim_pp)


#def IMC_get_freq_amplitude(IMF,dx = 1,eps = 0):
#
#
#    try:
#        imf_pp,fim_pp = _postprocessing_IMF(IMF,eps)
#    except:
#        print("error during frequency calculation, skipping computing frequencies")
#        freq=-1
#    else:
#        nimfs,nx = IMF.shape
#         
#        freq=np.asarray(fim_pp)/dx
#    
#    amp0 = np.sqrt(integrate(imf_pp**2)*dx)  
#
#    if np.shape(IMF) == np.shape(imf_pp):
#        return freq,amp0 
#    else :
#        return freq,amp0,imf_pp



def IMC_get_freq_amp(IMF, dt = 1, resort = False, wshrink = 0, use_instantaneous_freq = True):
    """
    Compute amplitude and average frequency of a set of IMCs.
    Parameters
    ----------
    use_instantaneous_freq : bool
        if True, then it uses the instantaneous frequency to calculate the average frequency of the IMC,
        if False, the average freq is found by counting maxima and minima in the IMC

    wshrink : int (positive)
        if >0, then only the central part of the IMC[:,wshrink:-wshrink] is used to calculate the amplitude
        This should be used when periodic extension is used or when windowing is used.

    """


    nimfs,nx = IMF.shape

    if use_instantaneous_freq:
        imf_if,imf_ia = IMC_get_inst_freq_amp(IMF,dt)

        freq = [np.sum(ifreq[0+wshrink:nx-wshrink]*iamp[0+wshrink:nx-wshrink])/np.sum(iamp[0+wshrink:nx-wshrink]) \
                for ifreq,iamp in zip(imf_if,imf_ia)]

    else:
        npeaks = np.array([np.size(Maxmins_v3_6(iimf)) for iimf in IMF])
        freq = npeaks/(2*nx*dt)
    
    amp0 = np.sqrt(integrate(IMF[:,0+wshrink:nx-wshrink]**2)*dt) 

    if resort:
        kf = np.argsort(freq)
        freq = freq[kf]
        amp0 = amp0[kf]

    return np.array(freq),np.array(amp0)

def IMC_get_inst_freq_amp(IMF,dt):
    """
    % Produces the istantaneous frequency and amplitude of a set of imfs.
    % Adapted from FIFogram_v7.

    """
    fs = 1/dt #sampling frequency
    M0, N0 = np.shape(IMF)

    #arrays of inst. freqs and amplitudes
    IMF_iA = np.zeros((M0,N0))
    IMF_iF = np.zeros((M0,N0))
    
    min_iF = np.zeros(M0) 
    max_iF = np.zeros(M0) 
    
    for i in range(M0):
        
        maxmins = Maxmins_v3_6(IMF[i,:])
        
        if np.size(maxmins) >= 2:
            temp_val = fs/(2*np.diff(maxmins))
            max_iF[i] = temp_val.max()
            min_iF[i] = temp_val.min()

            if maxmins[0] == 0 and maxmins[-1] == N0-1:
                IMF_iA[i] = np.interp(np.linspace(0,N0-1,N0), maxmins, abs(IMF[i, maxmins]))
                IMF_iF[i] = np.interp(np.linspace(0,N0-1,N0), maxmins, np.concatenate([temp_val, [temp_val[-1]]]))
            
            elif maxmins[0]!=0 and maxmins[-1]!=N0-1:

                dummy = np.concatenate([[0], maxmins, [N0-1]])
                IMF_iA[i] = np.interp(np.arange(float(N0)), dummy, abs(IMF[i, dummy]))
                IMF_iF[i] = np.interp(np.arange(float(N0)), dummy, np.concatenate([[temp_val[0]], temp_val, [temp_val[-1]]*2 ]))
            
            elif maxmins[0]!=0 and maxmins[-1]==N0-1:
                dummy = np.concatenate([[0], maxmins])
                IMF_iA[i] = np.interp(np.arange(float(N0)), dummy, abs(IMF[i, dummy]))
                IMF_iF[i] = np.interp(np.arange(float(N0)), dummy, np.concatenate([[temp_val[0]], temp_val, [temp_val[-1]]]))
            else:
                dummy = np.concatenate([maxmins,[N0-1]])
                IMF_iA[i] = np.interp(np.arange(float(N0)), dummy, abs(IMF[i, dummy]))
                IMF_iF[i] = np.interp(np.arange(float(N0)), dummy, np.concatenate([temp_val, [temp_val[-1]]*2]))


    return IMF_iF, IMF_iA




def Maxmins_v3_6(x, mode = 'wrap'):

    from scipy.signal import argrelextrema
    maxima = argrelextrema(x, np.greater, mode = mode)
    minima = argrelextrema(x, np.less, mode = mode)

    extrema = np.sort(np.concatenate((maxima, minima), axis=1))

    return extrema.squeeze()



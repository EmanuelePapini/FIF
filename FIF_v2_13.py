import os
import numpy as np
from numpy import linalg as LA
from scipy.io import loadmat
from scipy.signal import argrelextrema 
from numpy import fft
import time
#import pandas as pd
import timeit
from numba import jit

__version__='2.13'

#WRAPPER (version unaware. To be called by FIF.py) 
# def FIF_run(*args,**kwargs):
#     return FIF_v2_9(*args,**kwargs)

#WRAPPER (version unaware. To be called by FIF.py) 

def movmean(f,n):
    """
    Compute the walking mean of f in n steps matlab style
    
    Parameters
    ----------
        bc_type : str, optional
            boundary condition to use (default 'None')
            {'None','periodic'}
    """

    ntot = np.size(f)

    nl = int(n/2)

    nr = nl + n%2    

    y = np.ndarray(ntot)

    for i in range(nl,ntot-nr,1):
        y[i] = np.mean(f[i-nl:i+nr])
    #boundary points
    for i in range(nl):
        y[i]  = np.mean(f[0:i+nr])
    for i in range(nr):
        y[-i-1] = np.mean(f[-nl-i-1:])

    return y
    

class AttrDictSens(dict):
    '''
    A case-sensitive dictionary with access via item, attribute, and call
    notations:

        >>> d = AttrDict()
        >>> d['Variable'] = 123
        >>> d['Variable']
        123
        >>> d.Variable
        123
        >>> d.variable
        123
        >>> d('VARIABLE')
        123
    '''

    def __init__(self, init={}):
        dict.__init__(self, init)

    def __getitem__(self, name):
        try :
            return super(AttrDictSens, self).__getitem__(name)
        except:
            raise AttributeError

    def __setitem__(self, key, value):
        return super(AttrDictSens, self).__setitem__(key, value)

    __getattr__ = __getitem__
    __setattr__ = __setitem__
    __call__ = __getitem__



def get_mask_v1_1(y, k,verbose,tol):
    """
    Rescale the mask y so that its length becomes 2*k+1.
    k could be an integer or not an integer.
    y is the area under the curve for each bar
    
    wrapped from FIF_v2_13.m
    
    """
    n = np.size(y)
    m = (n-1)//2
    k = int(k)

    if k<=m:

        if np.mod(k,1) == 0:
            
            a = np.zeros(2*k+1)
            
            for i in range(1, 2*k+2):
                s = (i-1)*(2*m+1)/(2*k+1)+1
                t = i*(2*m+1)/(2*k+1)

                s2 = np.ceil(s) - s

                t1 = t - np.floor(t)

                if np.floor(t)<1:
                    print('Ops')

                a[i-1] = np.sum(y[int(np.ceil(s))-1:int(np.floor(t))]) +\
                         s2*y[int(np.ceil(s))-1] + t1*y[int(np.floor(t))-1]
        else:
            new_k = int(np.floor(k))
            extra = k - new_k
            c = (2*m+1)/(2*new_k+1+2*extra)

            a = np.zeros(2*new_k+3)

            t = extra*c + 1
            t1 = t - np.floor(t)

            if k<0:
                print('Ops')
                a = []
                return a

            a[0] = np.sum(y[:int(np.floor(t))]) + t1*y[int(np.floor(t))-1]

            for i in range(2, 2*new_k+3):
                s = extra*c + (i-2)*c+1
                t = extra*c + (i-1)*c
                s2 = np.ceil(s) - s
                t1 = t - np.floor(t)

                a[i-1] = np.sum(y[int(np.ceil(s))-1:int(np.floor(t))]) +\
                         s2*y[int(np.ceil(s))-1] + t1*y[int(np.floor(t))-1]
            t2 = np.ceil(t) - t

            a[2*new_k+2] = np.sum(y[int(np.ceil(t))-1:n]) + t2*y[int(np.ceil(t))-1]

    else: # We need a filter with more points than MM, we use interpolation
        dx = 0.01
        # we assume that MM has a dx = 0.01, if m = 6200 it correspond to a
        # filter of length 62*2 in the physical space
        f = y/dx
        dy = m*dx/k
        # b = np.interp(list(range(1,int(m+1),m/k)), list(range(0,int(m+1))), f[m:2*m+1])
        b = np.interp(np.linspace(0,m,int(np.ceil(k+1))), np.linspace(0,m,m+1), f[m:2*m+1])

        a = np.concatenate((np.flipud(b[1:]), b))*dy

        if abs(LA.norm(a,1)-1)>tol:
            if verbose:
                print('\n\n Warning!\n\n')
                print(' Area under the mask equals %2.20f\n'%(LA.norm(a,1),))
                print(' it should be equal to 1\n We rescale it using its norm 1\n\n')
            a = a/LA.norm(a,1)
        
    return a
        

def get_window_file_path():
    import sys
    _path_=sys.modules[__name__].__file__[0:-12]
    return _path_+'prefixed_double_filter.mat'

def Settings(**kwargs):
    """
    Sets the default options to be passed to FIF
    WARNING: NO CHECK IS DONE ON THE VALIDITY OF THE INPUT
    
    WRAPPED FROM: Settings_v3.m
    """

    options = {}
    # General 
    #options['saveEnd'] = 0
    #options['saveInter'] = 0
    options['verbose'] = False    
    #options['plots'] = 0.0
    #options['saveplots'] = 0     
        
    # FIF
    options['delta'] = 0.001
    options['ExtPoints']=3
    options['NIMFs']=200
    options['Xi']=1.6
    options['alpha']='ave'
    options['MaxInner']=200
    options['MonotoneMaskLength']=True
    options['NumSteps']=1
    options['MaskLengthType']='angle'


    for i in kwargs:
        if i in options.keys() : options[i] = kwargs[i] 
    return AttrDictSens(options)


def FIF_run(x, options=None, M = np.array([]),**kwargs):
    
    if options is None:
        options = Settings()
    
    return FIF_v2_13(x,options,M=M,**kwargs)

def FIF_v2_13(f,options,M=np.array([]),window_file=None):
   
 
    tol = 1e-12 

    if window_file is None:
        window_file = get_window_file_path()


    f = np.asarray(f)
    if len(f.shape) > 1: 
        raise Exception('Wrong dataset, the signal must be a 1D array!')
        
    N = f.size
    IMF = np.zeros([options.NIMFs, N])
    Norm1f = np.max(np.abs(f))#LA.norm(f, np.inf)
    f = f/Norm1f
    
    ###############################################################
    #                   Iterative Filtering                       #
    ###############################################################
    MM = loadmat(window_file)['MM'].flatten()

    ### Create a signal without zero regions and compute the number of extrema ###
    f_pp = np.delete(f, np.argwhere(abs(f)<=tol))
    if np.size(f_pp) < 1: 
        print('Signal too small')
        return None,None

    maxmins_pp = Maxmins(np.concatenate([f_pp, f_pp[:10]]),tol,mode='wrap')    
    maxmins_pp = maxmins_pp[0] 
    if len(maxmins_pp) < 1:
        print('No extrema detected')
        return None,None
    
    maxmins_pp = maxmins_pp[maxmins_pp<f_pp.size]

    diffMaxmins_pp = np.diff(maxmins_pp)
    
    N_pp = len(f_pp)
    k_pp = maxmins_pp.shape[0]
    countIMFs = 0
    stats_list = []
    
    ### Begin Iterating ###
    while countIMFs < options.NIMFs and k_pp >= options.ExtPoints:
        countIMFs += 1
        print('IMF', countIMFs)
        
        SD = 1
        h = f

        if 'M' not in locals() or np.size(M)<countIMFs:

            if isinstance(options.alpha,str):

                if options.alpha == 'ave': 
                    m = 2*np.round(N_pp/k_pp*options.Xi)
                elif options.alpha == 'Almost_min': 
                    m = 2*np.min( [options.Xi*np.percentile(diffMaxmins_pp,30), np.round(N_pp/k_pp*options.Xi)])
                elif options.alpha == 'Median':
                    m = 2*np.round(np.median(diffMaxmins_pp)*options.Xi)
                else:    
                    raise Exception('Value of alpha not recognized!\n')

            else:
                m = 2*np.round(options.Xi*np.percentile(diffMaxmins_pp,options.alpha))

            if countIMFs > 1:
                if m <= stats['logM'][-1]:
                    if options.verbose:
                        print('Warning mask length is decreasing at step %1d. ' % countIMFs)
                    if options.MonotoneMaskLength:
                        m = np.ceil(stats['logM'][-1] * 1.1)
                        if options.verbose:
                            print(('The old mask length is %1d whereas the new one is forced to be %1d.\n' % (
                            stats['logM'][-1], np.ceil(stats['logM'][-1]) * 1.1)))
                    else:
                        if options.verbose:
                            print('The old mask length is %1d whereas the new one is %1d.\n' % (stats['logM'][-1], m))
        else:
            m = M[countIMFs-1]
            #m = M[countIMFs-1]



        inStepN = 0
        if options.verbose:
            print('\n IMF # %1.0d   -   # Extreme points %5.0d\n' %(countIMFs,k_pp))
            print('\n  step #            SD             Mask length \n\n')

        stats = AttrDictSens({'logM': [], 'posF': [], 'valF': [], 'inStepN': [], 'diffMaxmins_pp': []})
        stats['logM'].append(int(m))

        a = get_mask_v1_1(MM, m,options.verbose,tol)#
        ExtendSig = False
        
        if N < np.size(a):
            ExtendSig = True
            Nxs = int(np.ceil(np.size(a)/N))
            N_old = N
            if np.mod(Nxs, 2) == 0:
                Nxs = Nxs + 1

            h_n = np.hstack([h]*Nxs)

            h = h_n
            N = Nxs * N

        Nza = N - np.size(a)
        if np.mod(Nza, 2) == 0:
            a = np.concatenate((np.zeros(Nza//2), a, np.zeros( Nza//2)))
            l_a = a.size
            ifftA = fft.fft(np.roll(a,l_a//2)).real
        else:
            a = np.concatenate((np.zeros( (Nza-1)//2 ), a, np.zeros( (Nza-1)//2 + 1)))
            l_a = a.size
            ifftA = fft.fft(np.roll(a,l_a//2)).real
        
        fftH = fft.fft(h)
        fft_h_new = fftH.copy()

        #compensate for the aliasing introduced in the DFT of the filter
        posF = np.where(np.diff(ifftA)>0)[0][0]
        stats.posF.append(int(posF))
        stats.valF.append(ifftA[posF])

        ifftA -= stats.valF[-1]
        ifftA[ifftA<0] = 0

        while SD>options.delta and inStepN<options.MaxInner:
            
            inStepN += options.NumSteps

            fft_h_old = (1-ifftA)**(inStepN-1) * fftH
            fft_h_new = (1-ifftA)**inStepN * fftH



            SD = LA.norm(fft_h_new-fft_h_old)**2/LA.norm(fft_h_old)**2

            ############### Generating f_n #############
            if options.verbose:
                print('    %2.0d      %1.40f          %2.0d\n' % (inStepN,SD,m))

        
        h = fft.ifft(fft_h_new)
        if ExtendSig:
            N = N_old
            h = h[int(N*(Nxs-1)/2):int(N*((Nxs-1)/2+1))]

        if inStepN >= options.MaxInner:
            print('Max # of inner steps reached')

        stats['inStepN'].append(inStepN)
        h = np.real(h)
        IMF[countIMFs-1, :] = h
        f = f-h

        #### Create a signal without zero regions and compute the number of extrema ####

        f_pp = np.delete(f, np.argwhere(abs(f)<=tol))
        if np.size(f_pp) <1:
            break
        
        if stats.logM[-1]>=20: 
            maxmins_pp = Maxmins(movmean(np.concatenate([f_pp,f_pp[:10]]),10),tol,mode='wrap')[0]
        else:    
            maxmins_pp = Maxmins(np.concatenate([f_pp,f_pp[:10]]),tol,mode='wrap')[0]
        
        if maxmins_pp is None:
            break

        maxmins_pp = np.delete(maxmins_pp, np.argwhere(maxmins_pp>=f_pp.size))
        diffMaxmins_pp = np.diff(maxmins_pp)
        N_pp = np.size(f_pp)
        k_pp = maxmins_pp.shape[0]

        stats_list.append(stats)

    # f = f.reshape(1,-1)
    # IMF = IMF[0:countIMFs, :]
    # IMF = np.concatenate((IMF, f), axis=0)
    IMF = IMF[0:countIMFs, :]
    IMF = np.vstack([IMF, f[:]])

    IMF = IMF*Norm1f # We scale back to the original values


    return IMF, stats_list



def Maxmins(x,tol,mode='wrap'):
    """
    Wrapped from: Maxmins_v3_8 in FIF_v2_13.m
    """
    @jit(nopython=True) 
    def maxmins_wrap(x,df, N,Maxs,Mins):

        h = 1
        while h<N and np.abs(df[h]/x[h]) <= tol:
            h = h + 1
   
        if h==N:
            return None, None

        cmaxs = 0
        cmins = 0
        c = 0
        N_old = N
        
        df = np.zeros(N+h)
        df[0:N] = x
        df[N:N+h] = x[1:h+1]
        for i in range(N+h-1):
            df[i] = df[i+1] - df[i]
        
        f = np.zeros(N+h-1)
        f[0:N] = x
        f[N:] = f[1:h]
        
        N = N+h
        #beginfor
        for i in range(h-1,N-2):
            if abs(df[i]*df[i+1]/f[i]**2) <= tol :
                if df[i]/abs(f[i]) < -tol:
                    last_df = -1
                    posc = i
                elif df[i]/abs(f[i]) > tol:
                    last_df = +1
                    posc = i
                elif df[i] == 0:
                    last_df = 0
                    posc = i

                c = c+1

                if df[i+1]/abs(f[i]) < -tol:
                    if last_df == 1 or last_df == 0:
                        cmaxs = cmaxs +1
                        Maxs[cmaxs] = (posc + (c-1)//2 +1)%N_old
                    c = 0
                
                if df[i+1]/abs(f[i]) > tol:
                    if last_df == -1 or last_df == 0:
                        cmins = cmins +1
                        Mins[cmins] = (posc + (c-1)//2 +1)%N_old
                    c = 0

            if df[i]*df[i+1]/f[i]**2 < -tol:
                if df[i]/abs(f[i]) < -tol and df[i+1]/abs(f[i]) > tol:
                    cmins  =cmins+1
                    Mins[cmins] = (i+1)%N_old
                    if Mins[cmins]==0:
                        Mins[cmins]=1
                    last_df=-1

                elif df[i]/abs(f[i]) > tol and df[i+1]/abs(f[i])  < -tol:
                    cmaxs = cmaxs+1
                    Maxs[cmaxs] = (i+1)%N_old
                    if Maxs[cmaxs] == 0:
                        Maxs[cmaxs]=1
            
                    last_df =+1

        if c>0:
            if cmins>0 and Mins[cmins] == 0 : Mins[cmins] = N
            if cmaxs>0 and Maxs[cmaxs] == 0 : Maxs[cmaxs] = N

        return Maxs[0:cmaxs], Mins[0:cmins]

    N = np.size(x)

    Maxs = np.zeros(N)
    Mins = np.zeros(N)
    
    df = np.diff(x)

    if mode == 'wrap':
        Maxs, Mins = maxmins_wrap(x,df,N,Maxs,Mins)
        if Maxs is None or Mins is None:
            return None,None,None

        maxmins = np.sort(np.concatenate((Maxs,Mins) ))
        
        if any(Mins ==0): Mins[Mins == 0] = 1
        if any(Maxs ==0): Maxs[Maxs == 0] = 1
        if any(maxmins ==0): maxmins[maxmins == 0] = 1

    return maxmins,Maxs,Mins


def evaluate_residual(fh,fm,j):
    r2 = fh*(1-fm)**(j-1)
    r1 = r2*fm
    return np.sum(r1**2)/np.sum(r2**2)







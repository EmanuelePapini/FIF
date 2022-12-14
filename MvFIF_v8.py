"""
 Multivariate Fast Iterative Filtering python package

 Dependencies : numpy, scipy, numba

 Authors: 
    Python version: Emanuele Papini - INAF (emanuele.papini@inaf.it) 
    Original matlab version: Antonio Cicone - university of L'Aquila (antonio.cicone@univaq.it)
"""
import os
import numpy as np
from numpy import linalg as LA
from scipy.io import loadmat
from scipy.signal import argrelextrema 
from numba import jit,njit

__version__='8.0'


#WRAPPER (version unaware. To be called by MvFIF.py) 
def FIF_run(x, *args, options=None, M = np.array([]),**kwargs):
    
    if options is None:
        return MvFIF(x,*args,**kwargs)
    else:
        return MvFIF(x, options['delta'], options['alpha'], options['NumSteps'], \
                           options['ExtPoints'], options['NIMFs'], options['MaxInner'], \
                           Xi = options['Xi'], M = M, \
                    MonotoneMaskLength = options['MonotoneMaskLength'], \
                    verbose = options['verbose'],**kwargs)


@njit
def lanorm(x,ordd):
    return LA.norm(x,ordd)


def get_mask_v1_1(y, k):
    """
    Rescale the mask y so that its length becomes 2*k+1.
    k could be an integer or not an integer.
    y is the area under the curve for each bar
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

        if abs(LA.norm(a,1)-1)>10**-14:
            print('\n\n Warning!\n\n')
            print(' Area under the mask equals %2.20f\n'%(LA.norm(a,1),))
            print(' it should be equal to 1\n We rescale it using its norm 1\n\n')
            a = a/LA.norm(a,1)
        
    return a



def MvFIF(x, delta, alpha, NumSteps, ExtPoints, NIMFs, MaxInner, Xi=1.6, M=np.array([]), MonotoneMaskLength=True, verbose=False,window_file='prefixed_double_filter.mat',fft='pyfftw',threads=None):
    """
    MultiVariate Fast Iterative Filtering python implementation (version 8)
    adapted from MvFIF_v8.m
    
    INPUT: x: array-like, shape(D,N)
        
        THE REST OF THE CRAZY INPUT IS AS USUAL :P

        the FIF decomposition is along N, and the analysis is performed on
        D channels/signals at once omogenizing the window size to be the same
        i.e., it is a concurrent decomposition
    """
    print('running MvFIF decomposition...')

    if fft =='pyfftw':
        print('using pyfftw...')
        import pyfftw
        fftw = pyfftw.interfaces.numpy_fft.fft
        ifftw = pyfftw.interfaces.numpy_fft.ifft
        fftkwargs = {'threads':threads} if threads is not None else {}
    else:
        print('using numpy.fft...')
        from numpy import fft
        fftw = fft.fft
        ifftw = fft.ifft
        fftkwargs = {}

    tol = 1e-12

    f = np.asarray(x)
    D,N = f.shape
    IMF = np.zeros([NIMFs+1, D, N])
    

    ###############################################################
    #                   Iterative Filtering                       #
    ###############################################################
    MM = loadmat(window_file)['MM'].flatten()

    ### Create a signal without zero regions and compute the number of extrema ###
    g = normc(f)
    g = np.array(dot(g[:,0:-1],g[:,1:]))
    g[g>1] = 1
    g[g<-1] = -1
    f_pp = np.arccos(g)
 
    f_pp = np.delete(f_pp, np.argwhere(abs(f_pp)<=1e-18))
    if np.size(f_pp) < 1:
        print('Signal too small')
        return None, None

    maxmins_pp = Maxmins_v3_8(f_pp,tol,mode='wrap')[0]
    if np.size(maxmins_pp) < 1:
        print('Signal too small')
        return None, None
    
    diffMaxmins_pp = np.diff(maxmins_pp)
    
    N_pp = f_pp.shape[0]
    k_pp = maxmins_pp.shape[0]
    countIMFs = 0
    stats_list = []
    
    ssend = '\r'
    ### Begin Iterating ###
    while countIMFs < NIMFs and k_pp >= ExtPoints:
        
        countIMFs += 1
        if countIMFs ==NIMFs: ssend = '\n'
        print('IMF', countIMFs,'/',NIMFs, end=ssend)
        
        SD = 1
        h = f

        if 'M' not in locals() or np.size(M)<countIMFs:

            if isinstance(alpha,str):
                if alpha == 'ave': 
                    m = 2*np.round(N_pp/k_pp*Xi)
                elif alpha == 'Almost_min':
                    m = 2*np.round(Xi*np.percentile(diffMaxmins_pp,30))
                    mp = 2*np.round(N_pp/k_pp*Xi)
                    if m >= mp : m = mp
                    del mp
                else:
                    raise Exception('Value of alpha not recognized!\n')
            else:
                m = np.round(Xi*np.percentile(diffMaxmins_pp,alpha))

            if countIMFs > 1:
                if m <= stats['logM'][-1]:
                    if verbose:
                        print('Warning mask length is decreasing at step %1d. ' % countIMFs)
                    if MonotoneMaskLength:
                        m = np.ceil(stats['logM'][-1] * 1.1)
                        if verbose:
                            print(('The old mask length is %1d whereas the new one is forced to be %1d.\n' % (
                            stats['logM'][-1], np.ceil(stats['logM'][-1]) * 1.1)))
                    else:
                        if verbose:
                            print('The old mask length is %1d whereas the new one is %1d.\n' % (stats['logM'][-1], m))
        else:
            m = M[countIMFs-1]

        inStepN = 0
        if verbose:
            print('\n IMF # %1.0d   -   # Extreme points %5.0d\n' %(countIMFs,k_pp))
            print('\n  step #            SD             Mask length \n\n')

        stats = {'logM': [], 'posF': [], 'valF': [], 'inStepN': [], 'diffMaxmins_pp': []}
        stats['logM'].append(int(m))

        a = get_mask_v1_1(MM, m)#
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
            fftA = fftw(np.roll(a,l_a//2),**fftkwargs).real
        else:
            a = np.concatenate((np.zeros( (Nza-1)//2 ), a, np.zeros( (Nza-1)//2 + 1)))
            l_a = a.size
            fftA = fftw(np.roll(a,l_a//2),**fftkwargs).real
        fftH = fftw(h,**fftkwargs)
        

        #% we compensate for the aliasing effect in the DFT of the filter
        # we look for the first minimum in fftA

        posF=np.where(np.diff(fftA)>0)[0][0]

        stats['posF'] = posF
        stats['valF'] = fftA[posF]

        fftA = fftA - fftA[posF] 
        fftA[fftA<0] = 0

        while SD > delta and inStepN < MaxInner:

            inStepN += NumSteps

            fft_h_old = (1-fftA[None,:])**(inStepN-1) * fftH
            fft_h_new = (1-fftA[None,:])**inStepN * fftH

            SD = LA.norm(fft_h_new-fft_h_old,axis=-1)**2/LA.norm(fft_h_old,axis=-1)**2
            SD = np.max(SD)
            
            if verbose:
                print('     %2.0d     %1.14f     %2.0d\n'%(inStepN,SD,m))
        
        h = ifftw(fft_h_new,**fftkwargs)
        
        if ExtendSig:
            N = N_old
            h = h[:,int(N*(Nxs-1)/2):int(N*((Nxs-1)/2+1))]
    
        if inStepN >= MaxInner:
            print('Max # of inner steps reached')

        stats['inStepN'] = inStepN
        h = np.real(h)
        IMF[countIMFs-1] = h
        f = f-h

        #### Create a signal without zero regions and compute the number of extrema ####

        g = normc(f)
        g = np.array(dot(g[:,0:-1],g[:,1:]))
        g[g>1] = 1
        g[g<-1] = -1
        f_pp = np.arccos(g)
        f_pp = np.delete(f_pp, np.argwhere(abs(f_pp)<=1e-18))
        if np.size(f_pp) < 1:
            print('Signal too small')
            return None, None
        
        if stats['logM'][-1] >=20:
            maxmins_pp=Maxmins_v3_8(movmean(f_pp,10),tol)[0] 
            # to include potential extrema at the boundaries 
            # we extend periodicaly of 10 points the signal
            # we make sure that the extrema identified belong to the signal before
            # pre-extention of 10 points
        else:
            maxmins_pp=Maxmins_v3_8(f_pp,tol)[0] 
 
        if maxmins_pp is None:
            break

        diffMaxmins_pp = np.diff(maxmins_pp)
        N_pp = np.size(f_pp)
        k_pp = maxmins_pp.shape[0]

        stats_list.append(stats)

    IMF = IMF[0:countIMFs+1]
    IMF[countIMFs] = f

    return IMF, stats_list



def Maxmins_v3_8(x,tol = 1e-15,mode='wrap'):

    @njit #(nopython=False) 
    def maxmins_wrap(x,df, N,Maxs,Mins):

        h = 1
        while h<N and np.abs(df[h-1])/x[h-1] <= tol:
            h = h + 1
   
        if h==N:
            return None, None
        
        #WARNING: h is now not the index (as in matlab)
        #, but it is the index plus one
        
        cmaxs = 0
        cmins = 0
        c = 0
        N_old = N
        df = np.concatenate((x,x[1:h+1]))
        df = np.diff(df)
        if h>1 : x = np.concatenate((x,x[1:h]))
        N = N+h

        #beginfor
        for i in range(h-1,N-2):
            dd2 = df[i]*df[i+1]/abs(x[i])**2 
            dd = df[i]/abs(x[i])
            ddp = df[i+1]/abs(x[i])
            if  dd2 <= tol and dd2 >= -tol :
                if dd < -tol:
                    last_df = -1
                    posc = i
                elif dd > tol:
                    last_df = +1
                    posc = i
                elif df[i] == 0:
                    last_df = 0
                    posc = i

                c = c+1

                if ddp < -tol:
                    if last_df == 1 or last_df == 0:
                        cmaxs = cmaxs +1
                        Maxs[cmaxs] = (posc + (c-1)//2 +1)%N_old
                    c = 0
                
                if ddp > tol:
                    if last_df == -1 or last_df == 0:
                        cmins = cmins +1
                        Mins[cmins] = (posc + (c-1)//2 +1)%N_old
                    c = 0

            if dd2 < -tol:
                if dd < -tol and ddp > tol:
                    cmins  =cmins+1
                    Mins[cmins] = (i+1)%N_old
                    if Mins[cmins]==0:
                        Mins[cmins]=1
                    last_df=-1

                elif dd > tol and ddp < -tol:
                    cmaxs = cmaxs+1
                    Maxs[cmaxs] = (i+1)%N_old
                    if Maxs[cmaxs] == 0:
                        Maxs[cmaxs]=1
            
                    last_df =+1

        #endfor
        if c>0:
            if cmins>0 and Mins[cmins] == 0 : Mins[cmins] = N
            if cmaxs>0 and Maxs[cmaxs] == 0 : Maxs[cmaxs] = N

        return Maxs[1:cmaxs+1], Mins[1:cmins+1]

    N = np.size(x)

    Maxs = np.zeros(N+1)
    Mins = np.zeros(N+1)
    
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



#######################################################################
# AUXILIARY FUNCTIONS NOT RELATED TO THE CORE FIF DECOMPOSITION
#######################################################################





def normc(Mat,norm='l2',axis = 0):
    from sklearn.preprocessing import normalize
    return normalize(Mat, norm=norm, axis=axis)

    #mods = np.sqrt(np.sum(Mat**2,axis=axis))

    #if axis == 0:
    #    for i in range(Mat.shape[0]): Mat[i,:]/=mods        
    #if axis == 1:
    #    for i in range(Mat.shape[1]): Mat[:,i]/=mods        
    #return Mat
@jit
def dot(a,b):
    return [i.dot(j) for i,j in zip(a.transpose(),b.transpose())]

@jit
def movmean(f,n,bc_type = 'None'):
    """
    Compute the walking mean of f in n steps
    
    Parameters
    ----------
        bc_type : str, optional
            boundary condition to use (default 'None')
            {'None','periodic'}
    """

    ntot = np.shape(f)[-1]

    n2 = int(n/2)

    y = np.copy(f)

    #boundary points
    if bc_type.lower() == 'none' :
        for i in range(n2):
            y[i]  = np.mean(f[0:n-n2+i])
            y[-i-1] = np.mean(f[-(n-n2+i):])
    
    if bc_type.lower() == 'periodic':

        for i in range(n2):
            y[i] = np.mean(np.roll(f, n2-i)[0:2*n2+1])
            y[-i-1] = np.mean(np.roll(f,-n2+i)[-(2*n2+1):])

    if n%2 :
        for i in range(n2,ntot-n2,1):
            y[i] = np.mean(f[i-n2:i+n2+1])
    else:
        for i in range(n2,ntot-n2,1):
            y[i] = np.mean(f[i-n2:i+n2])

    return y
    

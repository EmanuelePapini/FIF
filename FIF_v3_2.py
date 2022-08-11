"""
 Fast Iterative Filtering python package

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
from numpy import fft
import time
#import pandas as pd
import timeit
from numba import jit

__version__='3.2'

#WRAPPER (version unaware. To be called by FIF.py) 
def FIF_run(x, *args, options=None, M = np.array([]),**kwargs):
    
    if options is None:
        return FIF(x,*args,**kwargs)
    else:
        return FIF(x, options['delta'], options['alpha'], options['NumSteps'], \
            options['ExtPoints'], options['NIMFs'], options['MaxInner'], Xi = options['Xi'], \
            M = M, \
            MonotoneMaskLength = options['MonotoneMaskLength'], verbose = options['verbose'],**kwargs)



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
        b = np.interp(np.linspace(0,m,int(np.ceil(k+1))), np.linspace(0,m,m+1), f[m:2*m+1])

        a = np.concatenate((np.flipud(b[1:]), b))*dy

        if abs(LA.norm(a,1)-1)>10**-14:
            print('\n\n Warning!\n\n')
            print(' Area under the mask equals %2.20f\n'%(LA.norm(a,1),))
            print(' it should be equal to 1\n We rescale it using its norm 1\n\n')
            a = a/LA.norm(a,1)
        
    return a



def FIF(x, delta, alpha, NumSteps, ExtPoints, NIMFs, MaxInner, Xi=1.6, M=np.array([]), MonotoneMaskLength=True, verbose=False,window_file='prefixed_double_filter.mat'):
    
    f = np.asarray(x)
    N = f.size
    IMF = np.zeros([NIMFs, N])
    Norm1f = np.max(np.abs(f))#LA.norm(f, np.inf)
    f = f/Norm1f

    ###############################################################
    #                   Iterative Filtering                       #
    ###############################################################
    MM = loadmat(window_file)['MM'].flatten()

    ### Create a signal without zero regions and compute the number of extrema ###
    f_pp = np.delete(f, np.argwhere(abs(f)<=1e-18))
    maxmins_pp = Maxmins_v3_4(f_pp,mode='wrap')    
    maxmins_pp = maxmins_pp[0] 
    diffMaxmins_pp = np.diff(maxmins_pp)
    
    N_pp = len(f_pp)
    k_pp = maxmins_pp.shape[0]
    countIMFs = 0
    stats_list = []
    
    ### Begin Iterating ###
    while countIMFs < NIMFs and k_pp >= ExtPoints:
        countIMFs += 1
        print('IMF', countIMFs)
        
        SD = 1
        h = f

        if 'M' not in locals() or np.size(M)<countIMFs:

            if isinstance(alpha,str):

                if alpha == 'ave': 
                    m = 2*np.round(N_pp/k_pp*Xi)
                elif alpha == 'Almost_min': 
                    m = 2*np.min( [Xi*np.percentile(diffMaxmins_pp,30), np.round(N_pp/k_pp*Xi)])
                else:
                    raise Exception('Value of alpha not recognized!\n')

            else:
                m = 2*np.round(Xi *( np.max(diffMaxmins_pp)*alpha/100 +  np.min(diffMaxmins_pp)*(1-alpha/100)))

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
            fftA = fft.fft(np.roll(a,l_a//2)).real
        else:
            a = np.concatenate((np.zeros( (Nza-1)//2 ), a, np.zeros( (Nza-1)//2 + 1)))
            l_a = a.size
            fftA = fft.fft(np.roll(a,l_a//2)).real
        fftH = fft.fft(h)
        fft_h_new = fftH.copy()



        #determine number of iterations necessary for convergence
        compute_n_its = True
        if compute_n_its:
            fh = np.abs(fftH) ; fm = abs(fftA)
            
            r = lambda inStepN : evaluate_residual(fh,fm,inStepN) - delta

            inStepN = 1
            res = np.inf
            omfm = 1 - fm
            r2 = fh
            r1 = r2*fm
            while res > 0:
                r2 *= omfm
                r1 *= omfm
                omfm = omfm**2
                res = np.sum(r1**2) / np.sum(r2**2) - delta
                inStepN *= 2

            jl = inStepN //2; jr = inStepN
            rm = res; jm=jr
            while jr>jl+1:
                jm = np.floor((jl+jr)/2)
                rm = r(jm)
                if rm<0 :
                    jr = jm
                else:
                    jl = jm

            inStepN = jm if rm < 0 else jm + 1


        fft_h_old = fft_h_new.copy()
        fft_h_new = (1-fftA)**inStepN * fftH


        SD = LA.norm(fft_h_new-fft_h_old)**2/LA.norm(fft_h_old)**2

        ############### Generating f_n #############
        if verbose:
            print('    %2.0d      %1.40f          %2.0d\n' % (inStepN,SD,m))

        
        
        h = fft.ifft(fft_h_new)
        if ExtendSig:
            N = N_old
            h = h[int(N*(Nxs-1)/2):int(N*((Nxs-1)/2+1))]

        if inStepN >= MaxInner:
            print('Max # of inner steps reached')

        stats['inStepN'] = inStepN
        h = np.real(h)
        IMF[countIMFs-1, :] = h
        f = f-h

        #### Create a signal without zero regions and compute the number of extrema ####

        f_pp = np.delete(f, np.argwhere(abs(f)<=1e-18))
        maxmins_pp = Maxmins_v3_4(f_pp,mode='wrap')[0]
        if maxmins_pp is None:
            break

        diffMaxmins_pp = np.diff(maxmins_pp)
        N_pp = np.size(f_pp)
        k_pp = maxmins_pp.shape[0]

        stats_list.append(stats)

    IMF = IMF[0:countIMFs+1, :]
    IMF[-1,:] = f[:]

    IMF = IMF*Norm1f # We scale back to the original values


    return IMF, stats_list

def Maxmins_v3_4(x,mode='wrap'):

    @jit(nopython=True) 
    def maxmins_wrap(x,df, N,Maxs,Mins):

        h = 1
        while h<N and np.abs(df[h]) <= tol:
            h = h + 1
   
        if h==N:
            return None, None

        cmaxs = 0
        cmins = 0
        c = 0
        N_old = N
        df = np.zeros(N+h+2)
        df[0:N] = x
        df[N+1:N+h+2] = x[1:h+2]
        for i in range(N+h+1):
            df[i] = df[i+1] - df[i]
        N = N+h
        #beginfor
        for i in range(h-1,N-2):
            if df[i]*df[i+1] <= tol and df[i]*df[i+1] >= -tol :
                if df[i] < -tol:
                    last_df = -1
                    posc = i
                elif df[i] > tol:
                    last_df = +1
                    posc = i
                c = c+1

                if df[i+1] < -tol:
                    if last_df == 1:
                        cmaxs = cmaxs +1
                        Maxs[cmaxs] = (posc + (c-1)//2 +1)%N_old
                    c = 0
                
                if df[i+1] > tol:
                    if last_df == -1:
                        cmins = cmins +1
                        Mins[cmins] = (posc + (c-1)//2 +1)%N_old
                    c = 0

            if df[i]*df[+1] < -tol:
                if df[i] < -tol and df[i+1] > tol:
                    cmins  =cmins+1
                    Mins[cmins] = (i+1)%N_old
                    if Mins[cmins]==0:
                        Mins[cmins]=1
                    last_df=-1

                elif df[i] > tol and df[i+1] < -tol:
                    cmaxs = cmaxs+1
                    Maxs[cmaxs] = (i+1)%N_old
                    if Maxs[cmaxs] == 0:
                        Maxs[cmaxs]=1
            
                    last_df =+1

        #endfor
        if c>0:
            if Mins[cmins] == 0 : Mins[cmins] = N
            if Mins[cmaxs] == 0 : Mins[cmaxs] = N

        return Maxs[0:cmaxs], Mins[0:cmins]

    tol=1e-15
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

#######################################################################
# AUXILIARY FUNCTIONS NOT RELATED TO THE CORE FIF DECOMPOSITION
#######################################################################

def FIFogram_py(IMF, fs, IFint=None, winLen=None, winOverPerc=50, NIF=2000, fig=False):

    [M0, N0] = IMF.shape

    maxmins = np.empty(M0, dtype=object)
    IMF_iA = np.empty(M0, dtype=object)
    IMF_iF = np.empty(M0, dtype=object)

    min_iF = np.zeros(M0)
    max_iF = np.zeros(M0)

    for i in range(M0):
        maxmins[i] = Maxmins_v3_6(IMF[i,:])[0]
        maxmins[i] = maxmins[i].astype('int64')

        if len(maxmins[i]) < 2:
            IMF_iA[i] = np.zeros(N0)
            IMF_iF[i] = np.zeros(N0)

        else:
            temp_val = fs/(2*np.diff(maxmins[i]))
            max_iF[i] = np.max(temp_val)
            min_iF[i] = np.min(temp_val)
            if maxmins[i][0]==0 and maxmins[i][-1]==N0-1:
                IMF_iA[i] = np.interp(np.linspace(0,N0-1,N0), maxmins[i], abs(IMF[i, maxmins[i]]))
                IMF_iF[i] = np.interp(np.linspace(0,N0-1,N0), np.append(temp_val, temp_val[-1]))
            elif maxmins[i][0]!=0 and maxmins[i][-1]!=N0-1:
                IMF_iA[i] = np.interp(np.linspace(0,N0-1,N0), np.append(np.append(0, maxmins[i]), N0-1), abs(IMF[i, np.append(0, np.append(maxmins[i], N0-1))]))
                IMF_iF[i] = np.interp(np.linspace(0,N0-1,N0), np.append(np.append(0, maxmins[i]), N0-1), np.append(np.append(temp_val[0], np.append(temp_val, temp_val[-1])), temp_val[-1]))
            elif maxmins[i][0]!=1 and maxmins[i][-1]==N0-1:
                IMF_iA[i] = np.interp(np.linspace(0,N0-1,N0), np.append(1, maxmins[i]), abs(IMF[i, np.append(1, maxmins[i])]))
                IMF_iF[i] = np.interp(np.linspace(0,N0-1,N0), np.append(1, maxmins[i]), np.append(temp_val[0], np.append(temp_val, temp_val[-1])))
            else:
                IMF_iA[i] = np.interp(np.linspace(0,N0-1,N0), np.append(maxmins[i], N0-1), abs(IMF[i, np.append(maxmins[i], N0-1)]))
                IMF_iF[i] = np.interp(np.linspace(0,N0-1,N0), np.append(maxmins[i], N0-1), np.append(temp_val, np.append(temp_val[-1], temp_val[-1])))



    if not IFint:
        IFint = np.zeros(2)
        IFint[1] = np.max(max_iF)
        IFint[0] = np.min(min_iF)

    if not winLen:
        winLen = np.floor(N0/200)

    DeltaWin = np.floor(winLen*(1-winOverPerc/100))
    Nwin = int(np.floor((N0-(winLen-DeltaWin))/DeltaWin))

    IF = np.linspace(IFint[0], IFint[1], NIF)
    N_IF = NIF

    A = np.zeros([N_IF, Nwin])

    IMF_Num = np.empty([N_IF, Nwin], dtype=object)
    for i in range(N_IF):
        for j in range(Nwin):
            IMF_Num[i, j] = []

    for ii in range(M0):
        for jj in range(Nwin):
            temp_val = np.sum(IMF_iF[ii][np.arange(jj*DeltaWin, winLen+jj*DeltaWin, dtype=int)])/winLen
            if temp_val < IFint[1]+(IF[1]-IF[0]) and temp_val > IFint[0]-(IF[1]-IF[0]):

                v = np.min(abs(temp_val-IF))
                pos = np.where(abs(temp_val-IF) == v)


                IMF_Num[pos[0][0], jj].append(ii)


                A[pos, jj] = A[pos, jj] + np.sum(IMF_iA[ii][np.arange(jj*DeltaWin, winLen+jj*DeltaWin, dtype=int)])/winLen

    return A, IF, IMF_Num, IMF_iF, IMF_iA


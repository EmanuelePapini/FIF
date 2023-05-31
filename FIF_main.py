"""
 Fast Iterative Filtering python package

 Dependencies : numpy, scipy, numba

 Authors: 
    Python version: Emanuele Papini - INAF (emanuele.papini@inaf.it) 
    Original matlab version: Antonio Cicone - university of L'Aquila (antonio.cicone@univaq.it)
"""

import __main__
if hasattr(__main__,"FIFversion"):
    FIFversion = __main__.FIFversion
    if FIFversion == '2.13':
        from . import FIF_v2_13 as FIFpy
    elif FIFversion == '3.2':
        from . import FIF_v3_2 as FIFpy
    else:
        raise Exception('Wrong FIFversion selected!. Available methods are ',meths)
else:
    from . import FIF_v2_13 as FIFpy

print('Loading FIF version: '+FIFpy.__version__)

#change if you want to use a different version
from . import MvFIF_v8 as MvFIFpy

from . import fif_tools as ftools

import sys
import numpy as np
from copy import copy



__version__ = ('FIF:'+FIFpy.__version__,'MvFIF:'+MvFIFpy.__version__)

_path_=sys.modules[__name__].__file__[0:-11]
_window_file = _path_+'prefixed_double_filter.mat'

#WRAPPER 
def FIF_run(*args,**kwargs):
    
    return FIFpy.FIF_run(*args,**kwargs)




class FIF():
    """
    WARNING: This is an experimental version with minimal explanation.
    Should you need help, please contact Emanuele Papini (emanuele.papini@inaf.it) or Antonio Cicone (antonio.cicone@univaq.it)
    

    Python class of the Fast Iterative Filters (FIF) method  
    
    Calling sequence example

        #create the signal to be analyzed
        import numpy as np
        x = np.linspace(0,2*np.pi,100,endpoint=False)
        y = np.sin(2*x) + np.cos(10*x+2.3)
        
        #do the FIF analysis
        import FIF
    
        fif=FIF.FIF()

        fif.run(y)

        #plot the results
        import pylab as plt
        plt.ion()
        plt.figure()
        plt.plot(x,y,label='signal')
        [plt.plot(x,fif.data['IMC'][i,:],label = 'IMC#'+str(i)) for i in range(fif.data['IMC'].shape[0])]
        plt.legend(loc='best')

    Eventual custom settings (e.g. Xi, delta and so on) must be specified at the time of initialization
    (see __init__ below)
    Original matlab header
    % It generates the decomposition of the signal f :
    %
    %  f = IMF(1,:) + IMF(2,:) + ... + IMF(K, :)
    %
    % where the last row in the matrix IMF is the trend and the other rows
    % are actual IMFs
    %
    %                                Inputs
    %
    %   f         Signal to be decomposed
    %
    %
    %   M         Mask length values for each Inner Loop
    %
    %                               Output
    %
    %   IMF       Matrices containg in row i the i-th IMF. The last row
    %              contains the remainder-trend.
    %
    %   logM      Mask length values used for each IMF
    %
    %   See also SETTINGS_IF_V1, GETMASK_V1, MAXMINS_v3_4, PLOT_IMF_V8.
    %
    %  Ref: A. Cicone, J. Liu, H. Zhou. 'Adaptive Local Iterative Filtering for 
    %  Signal Decomposition and Instantaneous Frequency analysis'. Applied and 
    %  Computational Harmonic Analysis, Volume 41, Issue 2, September 2016, 
    %  Pages 384-411. doi:10.1016/j.acha.2016.03.001
    %  ArXiv http://arxiv.org/abs/1411.6051
    %
    %  A. Cicone. 'Nonstationary signal decomposition for dummies'. 
    %  Chapter in the book: Advances in Mathematical Methods and High 
    %  Performance Computing. Springer, 2019
    %  ArXiv https://arxiv.org/abs/1710.04844
    %
    %  A. Cicone, H. Zhou. 'Numerical Analysis for Iterative Filtering with 
    %  New Efficient Implementations Based on FFT'
    %  ArXiv http://arxiv.org/abs/1802.01359
    %
    
    Init Parameters 


    """


    def __init__(self, delta=0.001, alpha=30, NumSteps=1, ExtPoints=3, NIMFs=200, \
                       MaxInner=200, Xi=1.6, MonotoneMaskLength=True, verbose = False, window_file = None):


        self.__version__=FIFpy.__version__
        self.options={'delta' : delta, 'alpha' : alpha, 'verbose' : verbose, \
                      'NumSteps' : NumSteps, 'ExtPoints' : ExtPoints, 'NIMFs' : NIMFs, \
                      'MaxInner' : MaxInner, 'MonotoneMaskLength' : MonotoneMaskLength,\
                      'Xi' : Xi}

        if self.__version__ == '2.13':
            self.options = FIFpy.Settings(**self.options)

        self.FIFpy = FIFpy
   
        self.ancillary = {}
        self.window_file = _window_file if window_file is None else window_file
    
    def run(self, in_f, M=np.array([]), wshrink = 0, **kwargs):

        self.data = {}
        
        self.data['IMC'], self.data['stats_list'] = self.FIFpy.FIF_run(in_f, M = M, options = self.options, window_file = self.window_file, **kwargs)

        self.ancillary['wshrink'] = wshrink
        
        self.wsh = wshrink

    @property
    def input_timeseries(self):
        return np.sum(self.data['IMC'],axis=0)
    @property
    def IMC(self):
        return self.data['IMC'][:,self.wsh:-self.wsh] if self.wsh >0 else self.data['IMC'] 

    def get_inst_freq_amp(self,dt, as_output = False ):
        """
        get instantaneous frequencies and amplitudes of the IMCs
        if as_output is true, it returns the result instead of adding it to data
        """
        
        if as_output:
            return ftools.IMC_get_inst_freq_amp(self.data['IMC'],dt)
        
        self.data['IMC_inst_freq'], self.data['IMC_inst_amp'] = ftools.IMC_get_inst_freq_amp(self.data['IMC'],dt)


    def get_freq_amplitudes(self, as_output = False, use_instantaneous_freq = True,  **kwargs):
        """
        see fif_tools.IMC_get_freq_amplitudes for a list of **kwargs

        the available **kwargs should be
            dt = 1. : float 
                grid resolution (inverse of the sampling frequency) 
            resort = False : Bool
                if true, frequencies and amplitudes are sorted frequency-wise
            wshrink = 0 : int 
                only IMC[:,wshrink:-wshrink+1] will be used to compute freqs and amps.
                To use if one needs to throw away the part of the IMC that goes, e.g.,
                into the periodicization
            use_instantaneous_freq = True : bool
                use the instantaneous freq. to compute the average freq of the IMC
                
        """
        wsh = self.ancillary['wshrink']

        self.data['freqs'], self.data['amps'] = ftools.IMC_get_freq_amp(self.data['IMC'], \
                                                    use_instantaneous_freq = use_instantaneous_freq, wshrink = wsh,  **kwargs)

        self.ancillary['get_freq_amplitudes'] = kwargs
        self.ancillary['get_freq_amplitudes']['use_instantaneous_freq'] = use_instantaneous_freq
        
        if as_output: return self.data['freqs'], self.data['amps']



    def copy(self):
        return copy(self)



class MvFIF(FIF):
    """
    Python class for performing the Multivariate Fast Iterative Filtering decomposition. 
    
    (see Cicone and Pellegrino, IEEE Transactions on Signal Processing, vol. 70, pp. 1521-1531)

    This is an experimental version with minimal explanation.
    Should you need help, please contact Emanuele Papini (emanuele.papini@inaf.it) or Antonio Cicone (antonio.cicone@univaq.it)
    """

    def __init__(self, delta=0.001, alpha=30, NumSteps=1, ExtPoints=3, NIMFs=200, \
                       MaxInner=200, Xi=1.6, MonotoneMaskLength=True, verbose = False):


        self.__version__=FIFpy.__version__

        self.options={'delta' : delta, 'alpha' : alpha, 'verbose' : verbose, \
                      'NumSteps' : NumSteps, 'ExtPoints' : ExtPoints, 'NIMFs' : NIMFs, \
                      'MaxInner' : MaxInner, 'MonotoneMaskLength' : MonotoneMaskLength,\
                      'Xi' : Xi}

        self.FIFpy = MvFIFpy
   
        #contains ancillary data which keep trace of the processing done on the data
        self.ancillary = {}


    def get_inst_freq_amp(self,dt, as_output = False ):
        """
        get instantaneous frequencies and amplitudes of the IMCs
        if as_output is true, it returns the result instead of adding it to data
        """
        
        if as_output:
            return ftools.IMC_get_inst_freq_amp(np.squeeze(self.data['IMC'][:,0,:]),dt)
        
        self.data['IMC_inst_freq'], self.data['IMC_inst_amp'] = \
            ftools.IMC_get_inst_freq_amp(np.squeeze(self.data['IMC'][:,0,:]),dt)


    def get_freq_amplitudes(self, as_output = False, use_instantaneous_freq = True,  **kwargs):
        """
        see fif_tools.IMC_get_freq_amplitudes for a list of **kwargs

        the available **kwargs should be
            dt = 1. : float 
                grid resolution (inverse of the sampling frequency) 
            resort = False : Bool
                if true, frequencies and amplitudes are sorted frequency-wise
            wshrink = 0 : int 
                only IMC[:,wshrink:-wshrink+1] will be used to compute freqs and amps.
                To use if one needs to throw away the part of the IMC that goes, e.g.,
                into the periodicization
            use_instantaneous_freq = True : bool
                use the instantaneous freq. to compute the average freq of the IMC
                
        """
        wsh = self.ancillary['wshrink']

        self.data['freqs'], self.data['amps'] = ftools.IMC_get_freq_amp(np.squeeze(self.data['IMC'][:,0,:]), \
                     use_instantaneous_freq = use_instantaneous_freq, wshrink = wsh, \
                     **kwargs)

        self.ancillary['get_freq_amplitudes'] = kwargs
        self.ancillary['get_freq_amplitudes']['use_instantaneous_freq'] = use_instantaneous_freq
        
        if as_output: return self.data['freqs'], self.data['amps']


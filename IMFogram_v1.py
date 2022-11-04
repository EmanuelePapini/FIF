
import numpy as np

"""
% IMFogram_v1(IMF,fs,winOverPerc,IFint,NIF,winLen,fig)
%
% Produces the time-frequency plots of the Absolute values of the IMFs Amplitudes
% produced by (Fast) Iterative Filtering
%
%  Inputs
%
%  IMFs        = signal to be extended. Each row is a different IMF.
%                This matrix can include the trend which will not be
%                considered in the calculations
%  fs          = sampling frequency of the original signal
%  winOverPerc = Percentage value of overlapping of the sliding windows
%                Value between 0 and 100. Default value (50)
%  freq_range  = Lower and Upper bound of the frequency interval.
%                Default value = the average frequencies of the first and
%                last IMFs
%  NIF         = Number of rows of the IMFogram, i.e. number of
%                Instantaneous Frequencies values to be plotted in the
%                interval IFint. Default value (2000)
%  winLen      = Number of the sample points of the time window on which
%                computing the aewrage Frequency and Amplitude.
%                Default value = length of the signal / 200
%  verbose     = boolean which allows to specify if we want the algorithm
%                to give use statistics about the IMFs and their extrema
%                Default value set to (true)
%  scale       = frequency scale. Options
%                'linear'
%                'log' log along the frequency axis
%                'loglog' log along both the frequency and the amplitude
%                         values axes
%                'mel'
%                Default value set to ('linear')
%  fig         = handling of the figure to be generated.
%                If set to 0 no figure will be displayed.
%
%  Outputs
%
%  A       = NxM matrix containing the Amplitudes values. N = NIF and
%            represent the number of Instantaneous Frequencies values.
%            M is the number of time windows on which we devide the
%            original signal
%  IF      = Instantaneous Frequencies values for each row of A
%  IMF_Num = NxM matrix containing the IMF number(s) corresponding to each
%            row and column of A
%  IMF_iF  = Instantaneous Frequencies function associated with each IMF
%  IMF_iA  = Instantaneous Amplitudes function associated with each IMF
%  figA    = handle of the output figure to be generated
%  X       = NxM matrix containing the time
%  Y       = NxM matrix containing the frequency
%
%  Five options: None required -> IMFogram_v1 outputs the plot
%                2 required -> IMFogram_v1 outputs A and IF
%                3 required -> IMFogram_v1 outputs A, IF, the plot and its
%                handle
%                4 required -> IMFogram_v1 outputs A, IF, IMF_Num, the plot
%                and its handle
%                8 required -> IMFogram_v1 outputs X, Y, A, IF, IMF_Num, IMF_iF,
%                IMF_iA, the plot and its handle
%
%   See also IF2mel, hz2mel, mel2hz

% we use overlapping windows
"""


def IMFogram_v1(IMF, fs, winOverPerc = 50, IF_range = None, NIF = 2000, winLen = None, \
                verbose = False, scale = 'linear', plot = False, fig = None, ax = None,**contourfkwargs):
    """
    write something
    """
    
    M0, N0 = np.shape(IMF) #M0 = nimf, N0 = nt
    
    #arrays of inst. freqs and amplitudes
    IMF_iA = np.zeros((M0,N0))
    IMF_iF = np.zeros((M0,N0))
    
    min_iF = np.zeros(M0) 
    max_iF = np.zeros(M0) 
   
    zci = lambda v: np.where(np.diff(np.sign(v)) != 0)[0]
    
    
    for i in range(M0):
        
        zerocrossings = zci(IMF[i])
        
        if np.size(zerocrossings) < 2:
            IMF_iA[i]=0
            IMF_iF[i]=0
        else:
            temp_val = fs/(2*np.diff(zerocrossings))
            max_iF[i] = temp_val.max()
            min_iF[i] = temp_val.min()
            
            temp_IMF_maxval=np.zeros(zerocrossings.size - 1);
            
            #find maxima between zerocrossings
            temp_IMF_maxval_pos = [np.abs(IMF[i][zerocrossings[j]:zerocrossings[j+1]+1]).argmax() for j in range(zerocrossings.size-1)]
            temp_IMF_maxval_pos = [it + jt for it,jt in zip(temp_IMF_maxval_pos,zerocrossings[:-1])]
            temp_IMF_maxval = np.abs(IMF[i][temp_IMF_maxval_pos])
            temp_IMF_maxval_pos = np.array(temp_IMF_maxval_pos)
            # we prepare to comute the iAmplitude curve
            # we remove all repeated entries from temp_IMF_maxval_pos and the
            # corresponding entries in temp_IMF_maxval
            IMF_maxval_pos, IMF_maxval_pos_pos= np.unique(np.concatenate([[0],temp_IMF_maxval_pos,[N0-1]]), return_index = True)
            IMF_maxval = np.concatenate([[temp_IMF_maxval[0]],temp_IMF_maxval,[temp_IMF_maxval[-1]]])[IMF_maxval_pos_pos]
            IMF_iA[i]=np.interp(np.arange(float(N0)),IMF_maxval_pos,IMF_maxval)
            
            if zerocrossings[0] == 0 and zerocrossings[-1] == N0-1:
                IMF_iF[i] = np.interp(np.arange(float(N0)), zerocrossings, np.concatenate([temp_val, [temp_val[-1]]]))
            elif zerocrossings[0]!=0 and zerocrossings[-1]!=N0-1:
                dummy = np.concatenate([[0], zerocrossings, [N0-1]])
                IMF_iF[i] = np.interp(np.arange(float(N0)), dummy, np.concatenate([[temp_val[0]], temp_val, [temp_val[-1]]*2 ]))
            elif zerocrossings[0]!=0 and zerocrossings[-1]==N0-1:
                dummy = np.concatenate([[0], zerocrossings])
                IMF_iF[i] = np.interp(np.arange(float(N0)), dummy, np.concatenate([[temp_val[0]], temp_val, [temp_val[-1]]]))
            elif zerocrossings[0]==0 and zerocrossings[-1]!=N0-1:
                dummy = np.concatenate([zerocrossings,[N0-1]])
                IMF_iF[i] = np.interp(np.arange(float(N0)), dummy, np.concatenate([temp_val, [temp_val[-1]]*2]))

    if IF_range is None:
        # we compute the highest average frequency
        # and the lowest average frequency
        IF_range = [np.min(min_iF), np.max(max_iF)]
    
    if winLen is None:
        winLen=int(N0/200) # we define the smallest time window on which we compute the average amplitudes and frequencies
    
    DeltaWin = int(winLen*(1 - winOverPerc/100)) # Nonoverlapping Window length
    
    if DeltaWin==0:
        raise ValueError(' DeltaWin = 0! Try increasing "winLen" or decreasing "winOverPerc"')
    
    Nwin=int((N0-(winLen-DeltaWin))/DeltaWin)

    if scale == 'mel':
        IF = IF2mel(IF_range[0], IF_range[1], NIF, 0);
    elif scale == 'linear':
        IF=np.linspace(IF_range[0],IF_range[1],NIF) # Instantaneous Frequency values used to produce the IMFogram
    elif scale == 'log' or scale == 'loglog':
        IF=np.logspace(np.log10(IF_range[0]),np.log10(IF_range[1]),NIF)  # Instantaneous Frequency values used to produce the IMFogram
    else:
        raise ValueError('selected scale "'+scale+'"is not a valid option!' )
    
    N_IF=np.size(IF)
    
    A=np.zeros([N_IF,Nwin]) # we prepare the Matrix used to plot the IMFogram
    
    if verbose: IMF_Num = [[None for jj in range(Nwin)] for ii in range(N_IF)]
    for ii in range(M0): # we scan all IMFs containing more than 1 extrema
        for jj in range(Nwin): # we study the IMFogram over windows in time
            temp_val=np.sum(IMF_iF[ii][jj*DeltaWin:winLen+jj*DeltaWin])/winLen
            if temp_val<(IF_range[1]+IF[1]-IF[0]) and temp_val>(IF_range[0]-IF[1]+IF[0]):
                pos = np.argmin(np.abs(temp_val-IF)) # with mink we can spread the outcome to nearby frequencies to make more readable the plot # ceil(5*N_IF/1000) #originally: ceil(1.5*N_IF/100)
                v = np.abs((temp_val-IF)[pos])
                if verbose:
                    if IMF_Num[pos][jj] is not None:
                        IMF_Num[pos][jj].append(ii)  
                    else:
                        IMF_Num[pos][jj] = [ii]
                A[pos,jj] +=np.sum(IMF_iA[ii][jj*DeltaWin:winLen+jj*DeltaWin])/winLen # we consider the average amplitude to mimic what is done in the spectrogram where we compute the amplitude of the stationary sin or cos

    X,Y = np.meshgrid(np.arange(1,Nwin+1)*DeltaWin/fs,IF)

    out = {'X':X,'Y':Y,'A':A,'IF':IF,'IMF_iF':IMF_iF,'IMF_iA':IMF_iA}
    if verbose: out['IMF_Num'] = IMF_Num
    if not plot:
        return out        
    
    import pylab as plt
    if fig is None:
        fig=plt.figure()
    if ax is None:
        ax=fig.add_subplot()

    ax.contourf(X,Y,A,**contourfkwargs)
    ax.set_title('IMFogram')
    plt.ion()
    plt.show()
    out['fig'] = fig
    out['ax'] = ax
    return out

def IF2mel(min_iF, max_iF, nfilts, htkmel):

    minmel = hz2mel(min_iF, htkmel)
    maxmel = hz2mel(max_iF, htkmel)
    return mel2hz(minmel+np.arange(nfilts)/(nfilts-1)*(maxmel-minmel), htkmel)

def mel2hz(z, htk = 0):
    """
    %   f = mel2hz(z, htk)
    %   Convert 'mel scale' frequencies into Hz
    %   Optional htk = 1 means use the HTK formula
    %   else use the formula from Slaney's mfcc.m
    % 2005-04-19 dpwe@ee.columbia.edu
    """

    if htk == 1:
        return 700*(10.^(z/2595)-1)
        
    f_0 = 0 # 133.33333;
    f_sp = 200/3 # 66.66667;
    brkfrq = 1000
    brkpt  = (brkfrq - f_0)/f_sp  # starting mel value for log region
    logstep = np.exp(np.log(6.4)/27); # the magic 1.0711703 which is the ratio needed to get from 1000 Hz to 6400 Hz in 27 steps, and is *almost* the ratio between 1000 Hz and the preceding linear filter center at 933.33333 Hz (actually 1000/933.33333 = 1.07142857142857 and  exp(log(6.4)/27) = 1.07117028749447)
    
    linpts = z < brkpt
    
    f = np.zeros(np.shape(z))
    
    # fill in parts separately
    f[linpts] = f_0 + f_sp*z[linpts]
    f[~linpts] = brkfrq*np.exp(np.log(logstep)*(z[~linpts]-brkpt))
    return f

def hz2mel(f,htk = 0):
    """
    %  z = hz2mel(f,htk)
    %  Convert frequencies f (in Hz) to mel 'scale'.
    %  Optional htk = 1 uses the mel axis defined in the HTKBook
    %  otherwise use Slaney's formula
    % 2005-04-19 dpwe@ee.columbia.edu
    """

    if htk == 1:
        return 2595 * np.log10(1+f/700)
    # Mel fn to match Slaney's Auditory Toolbox mfcc.m
    
    f_0 = 0 # 133.33333
    f_sp = 200/3 # 66.66667
    brkfrq = 1000
    brkpt  = (brkfrq - f_0)/f_sp  # starting mel value for log region
    logstep = np.exp(np.log(6.4)/27) # the magic 1.0711703 which is the ratio needed to get from 1000 Hz to 6400 Hz in 27 steps, and is *almost* the ratio between 1000 Hz and the preceding linear filter center at 933.33333 Hz (actually 1000/933.33333 = 1.07142857142857 and  exp(log(6.4)/27) = 1.07117028749447)
    
    linpts = f < brkfrq
    
    z = np.zeros(np.shape(f)) 
    
    # fill in parts separately
    z[linpts] = (f[linpts] - f_0)/f_sp
    z[~linpts] = brkpt+(np.log(f[~linpts]/brkfrq))/np.log[logstep] 
    return z

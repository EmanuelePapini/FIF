# FAST ITERATIVE FILTERING

This repository contains the python package for the Fast Iterative Filtering (FIF) and the Multivariate Fast Iterative Filtering (MvFIF) algorithm.

FIF is an adaptive method for decomposing a 1D signal into a set of Intrinsic Mode Components (IMC) plus a trend. These components are simple oscillating functions whose average frequency is well behaved and form a complete and nearly orthogonal basis of the original signal.

In the package, IMFogram_v1.py contains the methods to calculate the IMFogram (see https://ui.adsabs.harvard.edu/abs/2020arXiv201114209B/abstract)

### Dependencies ###
The package has been written and tested in python3.

Dependencies: scipy, numpy, numba (plus other standard libraries that should be already installed)

### Install ###

Simply download the repository in the desired folder to start using it.
If you have a PYTHONPATH already set, you can put the FIF folder directly there so that you can import the package from everywhere.

example: assuming FIF is located in the PYTHONPATH or in the local path from where python3 is been executed 

```
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

```

### IMPORTANT REMARKS ###

Currently, there are different versions of FIF available (2.13 and 3.2). The system automatically uses version 2.13, which is more recent than v3.2. As such, v3.2 is considered deprecated and should not be used. However, it is possible to select that version by doing the following:

```
FIFversion = '3.2'
import FIF
```

A message specifying which version of FIF has been loaded will be always displayed.


### Contacts ###

The python version of the FIF and the MvFIF algorithm have been written by Emanuele Papini - INAF (emanuele.papini@inaf.it).

The original code and algorithm conceptualization are authored by Antonio Cicone - University of L'Aquila (antonio.cicone@univaq.it).

Please feel free to contact us would you need any help in getting FIF up and running.

### Links ###
 http://people.disim.univaq.it/~antonio.cicone/Software.html

### References ###
1) A. Cicone, H. Zhou. [Numerical Analysis for Iterative Filtering with New Efficient Implementations Based on FFT.](https://arxiv.org/abs/1802.01359) Numerische Mathematik, 147 (1), pages 1-28, 2021. doi: 10.1007/s00211-020-01165-5
2) A. Cicone and E. Pellegrino. [Multivariate Fast Iterative Filtering for the decomposition of nonstationary signals.](https://arxiv.org/abs/1902.04860) IEEE Transactions on Signal Processing, Volume 70, pages 1521-1531, 2022. doi: 10.1109/TSP.2022.3157482


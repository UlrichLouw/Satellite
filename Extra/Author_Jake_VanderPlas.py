# Author: Jake VanderPlas
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
#   To report a bug or issue, use the following forum:
#    https://groups.google.com/forum/#!forum/astroml-general
import matplotlib.pyplot as plt
import numpy as np
a = 5. # shape

s = np.random.weibull(a, 1000)

x = np.arange(1,100.)/50.


#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
#------------------------------------------------------------
# Define the distribution parameters to be plotted

#count, bins, ignored = plt.hist(np.random.weibull(5.,1000))

x = np.arange(1,14.)

B = 0.4287

n = 20624500/365

scale = weib(x, n, B).max()

print(np.random.weibull(, 1000))

plt.plot(x, weib(x, n, B)/scale)

plt.show()
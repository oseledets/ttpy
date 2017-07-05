# we need numpy, tt
from __future__ import print_function
try:
    xrange
except NameError:
    xrange = range

import numpy as np
import tt, tt.cross
from tt.cross.rectcross import cross
# fascilitating reshape
def reshape(x, shape):
    return np.reshape(x, shape, order = 'F')
# our TT-als module
from als import ttSparseALS
# import visualization tools
import matplotlib.pyplot as plt
from matplotlib import gridspec


def demo_completion():
    d = 3
    n=20
    crossR=18
    shape = np.array([n]*d)
    def func(X):
        return 1./(1+(X - n/2)**2).sum(axis = 1)**0.5

    # tt-approximation built via cross approximation
    x0 = tt.rand(np.array([n]*d), r = crossR)
    tta = cross(func, x0)
    print("TT-cross ranks: ", tta.r)

    R = 10
    gamma = 0.25
    P = int(np.floor(gamma*d*n*(R**2)))
    Pb = 100

    # random choice 
    indices = np.random.choice(n, [P, d])
    indicesB = np.random.choice(n, [Pb, d])
    # making list of tupled stings [indices]
    indices = [tuple(indices[k, :]) for k in xrange(indices.shape[0])]
    indicesB = [tuple(indicesB[k, :]) for k in xrange(indicesB.shape[0])]

    # set naturally filters input to be unique
    indices = set(indices)
    indicesB = set(indicesB)
    # convert it into list
    indices = list(indices)
    indicesB = list(indicesB)
    # return into numpy.array form
    indices = np.array(indices)
    indicesB = np.array(indicesB)

    print("Unique sample points: %d/%d (%d)" % (indices.shape[0], P, n**d))

    vals = func(indices)
    cooP = {'values': vals, 'indices': indices}
    cooPb = {'values': func(indicesB), 'indices': indicesB}

    maxR = 5
    x0 = tt.rand(shape, r=1)
    x0 = x0 * (1./ x0.norm())
    x0 = x0.round(0.)


    # verbose
    vb = True

    X1, f = ttSparseALS(
                        cooP,
                        shape,
                        x0=None,
                        ttRank=maxR,
                        maxnsweeps=50,
                        verbose=vb,
                        tol=1e-8,
                        alpha = 1e-3
    )
                       

    # Restore original, initial and approximation into full-format (do not try it in higher dimensions!)
    xf1 = X1.full() # approximation ALS
    a = tta.full() # original
    b = np.zeros([n]*d) # initial
    for p in xrange(indices.shape[0]):
        b[tuple(indices[p,:])] += vals[p]

    # Visualize slices of original, completed and initial tensors. Colormap is standartized.
    plt.clf()
    M = [a, xf1, b]
    titles = ['Original', 'Completed (ALS)', 'Initial']
    nRow = n
    nCol = 3
    fig = plt.figure(figsize=(5*nCol, nRow*5))
    gs = gridspec.GridSpec(nRow, nCol, wspace=0., hspace=1e-2, right=1-0.5/nCol)#top=1 - 0.5/nRow, 
        #bottom=0.5/nRow, left=0.5/nCol, right=1 - 0.5/nCol)

    for k in xrange(nRow):
        vmin = [x[k, :, :].min() for x in M]
        vmax = [x[k, :, :].max() for x in M]
        vmin = min( vmin)
        vmax = max( vmax)
        for l in xrange(nCol):
            ax = plt.subplot(gs[k, l])
            im = ax.imshow(M[l][k, :, :].T, vmin=vmin, vmax=vmax, interpolation='none')
            ax.set_axis_off()
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if k == 0:
                ax.set_title(titles[l], fontsize=20)
            if l == (nCol-1):
                box = ax.get_position()
                ax.set_position([box.x0*1.05, box.y0, box.width, box.height])
                axColor = plt.axes([box.x0*1.05 + box.width * 1.05, box.y0, 0.01, box.height])
                fig.colorbar(im, cax = axColor)
                
        
    #fig.subplots_adjust(right = 0.5)
    #cbar_ax = fig.add_axes([0.55, 0.45, 0.005, 0.11])
    #fig.colorbar(im, cax=cbar_ax)
    #fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig('demo_completion_gridplot.pdf', dpi=300)
    #fig.show()
    # plot functional curves

    plt.clf()
    fig = plt.figure()
    plt.semilogy(f['fit'], label='ALS')
    plt.xlabel('It.num.')
    plt.ylabel('ln( Fit )')
    plt.grid(True)
    plt.legend()
    plt.title('Funval ALS')
    plt.savefig('demo_completion_fitplot.pdf', dpi=300)
    
if __name__ == '__main__':
    demo_completion()

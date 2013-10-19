"""
Show how to modify the coordinate formatter to report the image "z"
value of the nearest pixel given x and y
"""

#Docs example: http://matplotlib.org/examples/pylab_examples/multi_image.html

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.cm import colors

#----------------------------------------------------------------------
def byte_to_float_ratio(nv, nm, bx, nz):
# no consideration of cache.
    bi = 4
    #print nv,nm,nz
    #rworst = (2.*nv*nm)/(bx*(nv+nm+nm*nv/float(nz))+bi)
    # in worst case, every vector element brings in an entire cache line (256 bits = 32 floats),
    # so nv=4 vectors (1/8th cache line), requires a transfer of 32 bytes = 8 floats
    # 16 vectors = 2 cache lines (, so the formula becomes
    # Assume infinite cache size, and cache line size = 1 float = 4 bytes. So there is no bandwidth waste
    cl = 4 # bytes
    flops = 2.*nv*nm
    # count in flops
    x_read = bx*nv
    a_read = bx*nm
    icol_read = bi
    y_write = bx*nm*nv/float(nz)
    ratio = (flops)/(x_read+a_read+icol_read+y_write)
    #ratio = (2.*nv*nm)/(bx*nv+bx*nm+bx*nm*nv/float(nz))+bi)
    # assumes cl=1
    cl = 64.  # size of most caches in bytes
    x_read = cl*np.ceil((bx*nv)/float(cl))  # assumes small cache, x must be reread
    rworst = flops/(x_read+a_read+icol_read+y_write)
    #rworst = (2.*nv*nm)/((cl*np.ceil((bx*nv)/cl)+nm+nm*nv/float(nz))+bi)
    x_read = bx*nv/float(nz)   # each element of x is read once
    rbest = flops/(x_read+a_read+icol_read+y_write)
    print "xx: ", [nv,nm,rworst,rbest]
    return(ratio)
#----------------------------------------------------------------------
def ratio_cache(nv, nm, bx, nz):
# byte to float taking cache effect into account
# per row:
# flops divided by number bytes sent/received
# nv : number vectors
# nm : number matrices
# bx = 4 or 8 (single, double precision)
# nz = nb nonzeros per row
# y = A*x
# y : nv*nm*bx   writes
# x : nv*bx      loads
# A : nm*nz*bx   loads
# col_id : nz*bi loads
    bi = 4
    #rworst = (2.*nv*nm)/(bx*(nv+nm+nm*nv/float(nz))+bi)
    # in worst case, every vector element brings in an entire cache line (256 bits = 32 floats),
    # so nv=4 vectors (1/8th cache line), requires a transfer of 32 bytes = 8 floats
    # 16 vectors = 2 cache lines (, so the formula becomes
    """
    cl = 4 # bytes
    # count in flops
    x_read = bx*nv
    a_read = bx*nm
    icol_read = bi
    y_write = bx*nm*nv/float(nz)
    ratio = (2.*nv*nm)/(x_read+a_read+icol_read+y_write)
    #ratio = (2.*nv*nm)/(bx*nv+bx*nm+bx*nm*nv/float(nz))+bi)
    # assumes cl=1
    """

    cl = 64  # size of most caches

    # for each row of A

    flops = 2.*nv*nm*nz

    a_read = bx*nm*nz
    icol_read = bi*nz
    y_write = bx*nm*nv

    x_read = nz *  cl*np.ceil((bx*nv)/float(cl))  # assumes small cache, x must be reread
    rworst = flops/(x_read+a_read+icol_read+y_write)

    x_read = bx*nv   # each element of x is read once
    rbest = flops/(x_read+a_read+icol_read+y_write)
    print "yy: ", [nv,nm,rworst,rbest]

    return([rworst,rbest])
#----------------------------------
#----------------------------------------------------------------------
def matrix_plot(X, title, filename):
    #X = 10*np.random.rand(5,3)

    fig, ax = plt.subplots()
    #ax.xaxis.set_ticks([1,2,4,6,8,10,12,16])
    #ax.xaxis.set_ticks_position([1,2,3,4,5,6,7,8])
    ax.xaxis.set_ticklabels([0,1,2,4,6,8,10,12,16])
    ax.yaxis.set_ticklabels([0,1,2,4,6,8,10,12,16])
    #ax.legend()
    #ax.set_title(title)
    ax.set_xlabel("nb matrices", fontsize=18)
    ax.set_ylabel("nb vectors", fontsize=18)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    im = ax.imshow(X, cmap=cm.jet, interpolation='nearest', aspect='equal',origin='lower')
    #ax.xlabel("nb matrices")
    norm = colors.Normalize(vmin=0, vmax=5)
    #fig.colorbar(im)
    im.set_norm(norm)

    # add text
    #print "ax= ", dir(ax)
    #print "fig= ", dir(fig)
    #print "plt= ", dir(plt)

    for i in range(8):
        for j in range(8):
            if (X[j,i] < 1.3 or X[j,i] > 4.5):
                plt.text(i-.3,j-.1,"%0.2f" % X[j,i], color='white')
            else:
                plt.text(i-.3,j-.1,"%0.2f" % X[j,i])


    numrows, numcols = X.shape
    #print X.shape

    def format_coord(x, y):
        row = x+1
        col = y+10
        #col = int(x+0.5)
        #row = int(y+0.5)
        if col>=0 and col<numcols and row>=0 and row<numrows:
            z = X[row,col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f'%(x, y)

    #ax.format_coord = format_coord
    #plt.show()
    plt.tight_layout()
    plt.savefig.format = "png"
    plt.savefig(filename)

#----------------------------------------------------------------------
nv = [1,2,3,4,5,6,7,8]
nv = [1,2,3,4,5,6,7,8]
nv = [1,2,4,6,8,10,12,16]
nm = [1,2,4,6,8,10,12,16]
bx = 4  # single precision
nz = [32, 64, 96]
vc = [[v] for v in nv for m in nm ]
vc = np.reshape(np.array(vc),[8,8])
#print vc
ratios_32 = [[byte_to_float_ratio(v, m, bx, 32)] for v in nv for m in nm ]
ratios_64 = [[byte_to_float_ratio(v, m, bx, 64)] for v in nv for m in nm ]
ratios_96 = [[byte_to_float_ratio(v, m, bx, 96)] for v in nv for m in nm ]

ratios_32 = np.reshape(np.array(ratios_32),[8,8])
ratios_64 = np.reshape(np.array(ratios_64),[8,8])
ratios_96 = np.reshape(np.array(ratios_96),[8,8])
#print ratios_32
#print ratios_96

#print byte_to_float_ratio(4,1,4,1)
#print byte_to_float_ratio(1,4,4,1)

print ratios_32
matrix_plot(ratios_32, "flops to bytes \n(no cache effects)", "flops_to_bytes_no_cache.pdf")
#----------------------------------------------------------------------
xratios_32 = [[ratio_cache(v, m, bx, 32)] for v in nv for m in nm ]
xratios_32 = np.reshape(np.array(xratios_32),[8,8,2])
#print xratios_32
xratios_32_worst = xratios_32[:,:,0]
xratios_32_best = xratios_32[:,:,1]
print xratios_32_worst
print xratios_32_best
#print np.shape(xratios_32)

matrix_plot(vc, "vc", "vc.pdf")
matrix_plot(xratios_32_worst, "flops to byte (worst)", "flops_to_bytes_worst.pdf")
matrix_plot(xratios_32_best, "flops to byte (best)", "flops_to_bytes_best.pdf")


##----------------------------------------------------------------------

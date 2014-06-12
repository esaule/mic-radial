import sys
import numpy as np
import matplotlib.pyplot as plt

#----------------------------------
def byte_to_float_ratio(nv, nm, bx, nz):
# no consideration of cache.
    bi = 4
    #print nv,nm,nz
    #rworst = (2.*nv*nm)/(bx*(nv+nm+nm*nv/float(nz))+bi)
    # in worst case, every vector element brings in an entire cache line (256 bits = 32 floats),
    # so nv=4 vectors (1/8th cache line), requires a transfer of 32 bytes = 8 floats
    # 16 vectors = 2 cache lines (, so the formula becomes
    ratio = (2.*nv*nm)/(bx*(nv+nm+nm*nv/float(nz))+bi)
    return(ratio)
#----------------------------------
#----------------------------------
def ratio(nv, nm, bx, nz):
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
    #print "yy: ", [nv,nm,rworst,rbest]
    return([rworst,rbest])


"""
    cl = 64.
    #rworst = (2.*nv*nm)/(bx*(nv+nm+nm*nv/float(nz))+bi)
    # in worst case, every vector element brings in an entire cache line (256 bits = 32 floats),
    # so nv=4 vectors (1/8th cache line), requires a transfer of 32 bytes = 8 floats
    # 16 vectors = 2 cache lines (, so the formula becomes
    if nv == 1:
        #rworst = (2.*nv*nm)/(bx*(16.+nm+nm*nv/float(nz))+bi)
        rworst = (2.*nv*nm)/(cl*np.ceil((bx*nv)/cl)+bx*(nm+nm*nv/float(nz))+bi)
    elif nv == 4:
        #rworst = (2.*nv*nm)/(bx*(16.+nm+nm*nv/float(nz))+bi)
        rworst = (2.*nv*nm)/(cl*np.ceil((bx*nv)/cl)+bx*(nm+nm*nv/float(nz))+bi)
        #rworst = (2.*nv*nm)/(bx*(16.*nv+nm+nm*nv/float(nz))+bi) # error
    elif nv == 16:
        #rworst= (2.*nv*nm)/(bx*(nv+nm+nm*nv/float(nz))+bi)
        rworst = (2.*nv*nm)/(cl*np.ceil((bx*nv)/cl)+bx*(nm+nm*nv/float(nz))+bi)
    else:
        print "nv case not considered"

    rbest = (2.*nv*nm)/(bx*(nm+(1+nm)*nv/float(nz))+bi)
"""
#----------------------------------
nz = 32
single = 4
double = 8
ulti_bandwidth = 190
max_bandwidth = 150
bandwidths = [150,190]
nv = [1,4]
nm = [1,4]
labels = [(v+','+m) for v in ["nv=1","nv=4"] for m in ["nm=1","nm=4"]]
labels.extend(["nv=16,nm=1"])
print labels
# v loop is slowest
# m loop is fastest
ratios_s = np.array([ratio(v,m,single, nz) for v in nv for m in nm])
ratios_s = np.reshape(np.append(ratios_s, ratio(16,1,single, nz)),[5,2])
ratios_s_ulti = ratios_s*ulti_bandwidth
ratios_s_max  = ratios_s*max_bandwidth

ratios_d =  np.array([ratio(v,m,double, nz) for v in nv for m in nm])
ratios_d = np.reshape(np.append(ratios_d, ratio(16,1,double, nz)),[5,2])
ratios_d_ulti = ratios_d*ulti_bandwidth
ratios_d_max  = ratios_d*max_bandwidth

ratios_s_worst = np.transpose(ratios_s_max)[0]
ratios_s_best = np.transpose(ratios_s_max)[1]

ratios_d_worst = np.transpose(ratios_d_max)[0]
ratios_d_best = np.transpose(ratios_d_max)[1]

print "ratio_d_worst shape: ", np.shape(ratios_d_worst)
print "ratio_d_best shape: ",  np.shape(ratios_d_best)
print "shape: ", np.shape(ratios_d_max)

#### BEGINNING OF CHART

ngroups = len(labels)

fig, ax = plt.subplots()
index = np.arange(ngroups)
#print "index= ", index
bar_width = 0.35/2
#print np.shape(index), np.shape(ratios_d_worst), np.shape(ratios_d_best)
#print np.shape(index), np.shape(ratios_s_worst), np.shape(ratios_s_best)

rects1 = plt.bar(index, ratios_s_worst, bar_width, alpha=1, color='b', label='single precision worst case')
rects1 = plt.bar(index+bar_width, ratios_s_best, bar_width, alpha=.5, color='b', label='single precision best case')
#print ratios_s_best
print("worst, best")
print(ratios_s_worst)
print(ratios_s_best)

rects2 = plt.bar(index+2*bar_width, ratios_d_worst, bar_width, alpha=1, color='r', label='double precision worst case')
rects2 = plt.bar(index+3*bar_width, ratios_d_best, bar_width, alpha=.5, color='r', label='double precision best case')

#plt.xlabel('Vector/Matrix sizes', fontsize=16)
plt.ylabel('Peak Performance (Gflop/s)', fontsize=18)
#plt.title('Best possible performance (max application bandwidth: 150 Gbytes/sec')
plt.xticks(index + bar_width + bar_width, labels, fontsize=14)
plt.legend(loc=2,fontsize=18)  # upper left
plt.grid(True)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

plt.tight_layout()
plt.savefig.format = "pdf"
plt.savefig("gflops_peak.pdf")
#plt.show()
#### END OF CHART

#======================================================================
speedup_s_worst = ratios_s_worst / ratios_s_worst[0]
speedup_s_best  = ratios_s_best  / ratios_s_best[0]
speedup_d_worst = ratios_d_worst / ratios_d_worst[0]
speedup_d_best  = ratios_d_best  / ratios_d_best[0]

fig, ax = plt.subplots()
index = np.arange(ngroups)

rects1 = plt.bar(index, speedup_s_worst, bar_width, alpha=1, color='b', label='single precision worst case')
rects1 = plt.bar(index+bar_width, speedup_s_best, bar_width, alpha=.5, color='b', label='single precision best case')

rects2 = plt.bar(index+2*bar_width, speedup_d_worst, bar_width, alpha=1, color='r', label='double precision worst case')
rects2 = plt.bar(index+3*bar_width, speedup_d_best, bar_width, alpha=.5, color='r', label='double precision best case')

plt.xlabel('Vector/Matrix sizes', fontsize=14)
plt.ylabel('Speedup', fontsize=14)
#plt.title('Speedup with respect to 1/1 case')
plt.xticks(index + bar_width, labels)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.legend(loc=2, fontsize=18)  # upper left
plt.grid(True)

plt.tight_layout()
plt.savefig.format = "pdf"
plt.savefig("speedup_wrt_base.pdf")

nv = [1,2,4,8,16]
nm = [1,2,4,8,16]
bx = 4  # single precision
nz = [32, 64, 96]
ratios_32 = [[v,m,byte_to_float_ratio(v, m, bx, 32)] for v in nv for m in nm ]
ratios_64 = [[v,m,byte_to_float_ratio(v, m, bx, 64)] for v in nv for m in nm ]
ratios_96 = [[v,m,byte_to_float_ratio(v, m, bx, 96)] for v in nv for m in nm ]

ratios_32 = np.array(ratios_32)
ratios_64 = np.array(ratios_64)
ratios_96 = np.array(ratios_96)
print ratios_32


labels = [(v+m) for v in ["v1","v4"] for m in ["m1","m4"]]


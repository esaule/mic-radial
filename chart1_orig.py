import sys
import numpy as np
import matplotlib.pyplot as plt

#----------------------------------
def byte_to_float_ratio(nv, nm, bx, nz):
# no consideration of cache.
    bi = 4
    #rworst = (2.*nv*nm)/(bx*(nv+nm+nm*nv/float(nz))+bi)
    # in worst case, every vector element brings in an entire cache line (256 bits = 32 floats),
    # so nv=4 vectors (1/8th cache line), requires a transfer of 32 bytes = 8 floats
    # 16 vectors = 2 cache lines (, so the formula becomes
        rworst = (2.*nv*nm)/(bx*(nv+nm+nm*nv/float(nz))+bi)
    else:
        print "nv case not considered"

    rbest = (2.*nv*nm)/(bx*(nm+(1+nm)*nv/float(nz))+bi)
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
    #rworst = (2.*nv*nm)/(bx*(nv+nm+nm*nv/float(nz))+bi)
    # in worst case, every vector element brings in an entire cache line (256 bits = 32 floats),
    # so nv=4 vectors (1/8th cache line), requires a transfer of 32 bytes = 8 floats
    # 16 vectors = 2 cache lines (, so the formula becomes
    if nv == 1:
        rworst = (2.*nv*nm)/(bx*(16.+nm+nm*nv/float(nz))+bi)
    elif nv == 4:
        rworst = (2.*nv*nm)/(bx*(16.+nm+nm*nv/float(nz))+bi)
    elif nv == 16:
        rworst= (2.*nv*nm)/(bx*(nv+nm+nm*nv/float(nz))+bi)
    else:
        print "nv case not considered"

    rbest = (2.*nv*nm)/(bx*(nm+(1+nm)*nv/float(nz))+bi)
    return([rworst,rbest])
#----------------------------------
nz = 32
single = 4
double = 8
ulti_bandwidth = 190
max_bandwidth = 150
bandwidths = [150,190]
nv = [1,4,16]
nm = [1,4]
labels = [(v+m) for v in ["v1","v4"] for m in ["m1","m4"]]
# v loop is slowest
# m loop is fastest
ratios_s =  np.array([ratio(v,m,single, nz) for v in nv for m in nm])
ratios_s_ulti = ratios_s*ulti_bandwidth
ratios_s_max  = ratios_s*max_bandwidth

ratios_d =  np.array([ratio(v,m,double, nz) for v in nv for m in nm])
ratios_d_ulti = ratios_d*ulti_bandwidth
ratios_d_max  = ratios_d*max_bandwidth


ratios_s_worst = np.transpose(ratios_s_max)[0]
ratios_s_best = np.transpose(ratios_s_max)[1]

ratios_d_worst = np.transpose(ratios_d_max)[0]
ratios_d_best = np.transpose(ratios_d_max)[1]

#### BEGINNING OF CHART

ngroups = len(labels)

fig, ax = plt.subplots()
index = np.arange(ngroups)
bar_width = 0.35/2

rects1 = plt.bar(index, ratios_s_worst, bar_width, alpha=1, color='b', label='single worst case')
rects1 = plt.bar(index+bar_width, ratios_s_best, bar_width, alpha=.5, color='b', label='single best case')

rects2 = plt.bar(index+2*bar_width, ratios_d_worst, bar_width, alpha=1, color='r', label='double worst case')
rects2 = plt.bar(index+3*bar_width, ratios_d_best, bar_width, alpha=.5, color='r', label='double best case')

plt.xlabel('Vector/Matrix sizes')
plt.ylabel('Peak Gflops')
plt.title('Best possible performance (max application bandwidth: 150 Gbytes/sec')
plt.xticks(index + bar_width, labels)
plt.legend(loc=2)  # upper left
plt.grid(True)

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

rects1 = plt.bar(index, speedup_s_worst, bar_width, alpha=1, color='b', label='speedup worst single case')
rects1 = plt.bar(index+bar_width, speedup_s_best, bar_width, alpha=.5, color='b', label='speedup best single case')

rects2 = plt.bar(index+2*bar_width, speedup_d_worst, bar_width, alpha=1, color='r', label='double worst double case')
rects2 = plt.bar(index+3*bar_width, speedup_d_best, bar_width, alpha=.5, color='r', label='double best double case')

plt.xlabel('Vector/Matrix sizes')
plt.ylabel('Speedup')
plt.title('Speedup with respect to 1/1 case')
plt.xticks(index + bar_width, labels)
plt.legend(loc=2)  # upper left
plt.grid(True)

plt.tight_layout()
plt.savefig.format = "pdf"
plt.savefig("speedup_wrt_base.pdf")

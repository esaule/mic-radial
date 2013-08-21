import sys
import numpy as np
import matplotlib.pyplot as plt

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
    r = 2.*nz*nv*nm/(bx*(nv+nm*nz+nm*nv)+nz*bi)
    return(r)
#----------------------------------
nz = 32
single = 4
double = 8
ulti_bandwidth = 190
max_bandwidth = 150
bandwidths = [150,190]
nv = [1,4]
nm = [1,4]
labels = [(v+m) for v in ["v1","v4"] for m in ["m1","m4"]]
# v loop is slowest
# m loop is fastest
ratios_s =  np.array([ratio(v,m,single, nz) for v in nv for m in nm])
ratios_s_ulti = ratios_s*ulti_bandwidth
ratios_s_max  = ratios_s*max_bandwidth
#print("single precision")
#print ratios_s
#print ratios_s_ulti
#print ratios_s_max

ratios_d =  np.array([ratio(v,m,double, nz) for v in nv for m in nm])
ratios_d_ulti = ratios_d*ulti_bandwidth
ratios_d_max  = ratios_d*max_bandwidth
#print("\ndouble precision")
#print ratios_d
#print ratios_d_ulti
#print ratios_d_max



#### BEGINNING OF CHART

ngroups = len(labels)

fig, ax = plt.subplots()
index = np.arange(ngroups)
bar_width = 0.35
opacity = .5

rects1 = plt.bar(index, ratios_s_ulti, bar_width, alpha=opacity,
           color='b', label='single ulti')
rects2 = plt.bar(index+bar_width, ratios_d_ulti, bar_width, alpha=opacity, color='r', label='double ulti')

opacity = 1
rects1 = plt.bar(index, ratios_s_max, bar_width, alpha=opacity,
           color='b', label='single max')
rects2 = plt.bar(index+bar_width, ratios_d_max, bar_width, alpha=opacity, color='r', label='double max')


plt.xlabel('Vector/Matrix sizes')
plt.ylabel('Peak Gflops')
plt.title('Best possible performance')
plt.xticks(index + bar_width, labels)
plt.legend(loc=2)  # upper left
plt.grid(True)

plt.tight_layout()
plt.savefig.format = "png"
plt.savefig("gordon.png")
#plt.show()
#### END OF CHART

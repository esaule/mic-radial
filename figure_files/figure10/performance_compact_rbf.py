
#ell_kd-tree_rcm_sym_1_x_weights_direct__no_hv_stsize_32_3d_32x_32y_32z.out:method_8a_multi, threads: 240, Max Gflops: 162.098709, min time: 0.207000 (ms)

#Create secondary file with
#nx ny nz rcm/norcm gflops

str = "#ell_kd-tree_rcm_sym_1_x_weights_direct__no_hv_stsize_32_3d_32x_32y_32z.out:method_8a_multi, threads: 240, Max Gflops: 162.098709, min time: 0.  207000 (ms)"

import matplotlib.pyplot as plt
import numpy as np
import myutil
import csv

def readData(filename):
    x1 = []
    y1 = []
    y2 = []
    with open(filename, "rb") as infile:
        reader = csv.reader(infile)
        for i, line in enumerate(reader):
            x1.extend([int(line[0])])
            y1.extend([float(line[1])])
            y2.extend([float(line[2])])
    print [x1,y1,y2]
    return [x1,y1,y2]

# in the file: x, norcm, rcm
[nx,n2,n3] = readData("none1.data")
[cx,c2,c3] = readData("compact.data")
[sx,s2,s3] = readData("supercompact.data")
[rx,r2,r3] = readData("random.data")


#-------------------------
"""
plt.clf()
plt.plot(sx,s2,'rs-',label='supercompact, no rcm', ms=10, lw=2)
plt.plot(sx,s3,'r^-',label='supercompact, rcm', ms=10, lw=2)
plt.plot(rx,r2,'bs-',label='random, no rcm', ms=10, lw=2)
plt.plot(rx,r3,'b^-',label='random, rcm', ms=10, lw=2)
plt.ylim([50,220])
plt.ylabel('Performance (Gflop/s)', fontsize=18)
plt.xlabel('number rows / 1000', fontsize=18)
plt.legend(loc=7, fontsize=18)  # upper left
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
myutil.horizontal_lines()
plt.savefig.format = "pdf"
plt.savefig("random_supercompact.pdf")
"""
#-------------------------
plt.clf()
plt.plot(cx,c2,'rs-',label='compact, no rcm', ms=10, lw=2)
plt.plot(cx,c3,'r^-',label='compact, rcm', ms=10, lw=2)
plt.plot(nx,n2,'bs-',label='rbf, no rcm', ms=10, lw=2)
plt.plot(nx,n3,'b^-',label='rbf, rcm', ms=10, lw=2)
plt.ylim([50,220])
plt.ylabel('Performance (Gflop/s)', fontsize=18)
plt.xlabel('number rows / 1000', fontsize=18)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.legend(bbox_to_anchor=(0,0.2,1,0.8), loc=7, fontsize=18)  # upper left
myutil.horizontal_lines()
plt.savefig.format = "pdf"
plt.savefig("rbf_compact.pdf")

exit()
#-------------------------

#-----------------------------------------------
#with open("o", "rb") as infile, open("test.csv", "wb") as outfile:
#writer = csv.writer(outfile, quoting=False)

rows_rcm = []
rows_norcm = []
gflop_rcm = []
gflop_norcm = []
dict_rcm = {}
dict_norcm = {}
with open("raw1.dat", "rb") as infile:
    reader = csv.reader(infile)
    for i, line in enumerate(reader):
        in_file,max_flops,nx,ny,nz,rcm = myutil.parse_line(line)
        nb_rows = nx*ny*nz/1000
        print in_file,max_flops,nx,ny,nz, nb_rows/1000, rcm
        if rcm == 0:
            rows_norcm.append(nb_rows)
            dict_norcm[nb_rows] = max_flops
        else:
            rows_rcm.append(nb_rows)
            dict_rcm[nb_rows] = max_flops

sorted_rows = np.sort(rows_norcm)
print "sorted rows: ", np.shape(sorted_rows)
for r in sorted_rows:
    gflop_norcm.append(dict_norcm[r])
    gflop_rcm.append(dict_rcm[r])

gflop_norcm = np.array(gflop_norcm)
gflop_rcm = np.array(gflop_rcm)
plt.plot(sorted_rows,gflop_norcm,'rv-',label='no rcm', ms=6, lw=2)
plt.plot(sorted_rows,gflop_rcm,'b^-', label='rcm', ms=6, lw=2)
myutil.horizontal_lines()
#plt.title('Gflops, 240 threads, Supercompact Matrix')
plt.ylim([50,220])
plt.ylabel('Gflops')
plt.xlabel('number rows / 1000')
plt.legend(loc=3, fontsize=12)  # upper left
plt.savefig.format = "pdf"
outfile='plot.pdf'
plt.savefig(outfile)
outfile='supercompact_max_perf.pdf'
plt.savefig(outfile)


with open("supercompact.data", "wb") as csvfile:
    print "sorted_rows", len(sorted_rows)
    writer = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_NONE)
    for i,r in enumerate(range(len(sorted_rows))):
        writer.writerow([sorted_rows[i], gflop_norcm[i], gflop_rcm[i]])
#-----------------------------------------------

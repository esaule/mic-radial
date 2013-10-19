import sys
import numpy as np
import matplotlib.pyplot as plt
import csv

def parse(infile):
    rows = []
    nb_rows = []
    all_nb_rows = []
    all_bw = []
    with open(infile, 'r') as csvfile:
        csvfile.seek(0)
        reader = csv.reader(csvfile, delimiter=' ')
        for i,row in enumerate(reader):
# if bandwidth in row, continue
            try:
                if row[5] == 'bandwidth=':
                    rows.append(row)
                else:
                    continue
            except:
                pass
    #print rows[0]
    #print rows[1]
    #print rows[2]

    for r in rows:
        #print r
        #print r[1], r[6]
        bandwidth = float(r[6])
        nb_rows = int(r[1])
        #print r
        print bandwidth, nb_rows
        all_nb_rows.append(nb_rows)
        all_bw.append(bandwidth)

    #print rows
    #print bw
    return [all_nb_rows, all_bw]

#----------------------------------------------------------------------
plt.clf()
[rows, bw] = parse("bench=read_cpp_col=compact_rows=00.out")
p1, = plt.plot(rows, bw, '-b^', label='read, cpp', ms=6, lw=2)

[rows, bw] = parse("bench=write_col=compact_rows=00.out")
p2, = plt.plot(rows, bw, '-rs', label='write', ms=6, lw=2)

[rows, bw] = parse("bench=read_col=compact_rows=00.out")
p3, = plt.plot(rows, bw, '-bs', label='read', ms=6, lw=2)

[rows, bw] = parse("bench=read_write_col=compact_rows=00.out")
p4, = plt.plot(rows, bw, '-gs', label='read/write', ms=6, lw=2)

[rows, bw] = parse("bench=read_write_cpp_col=compact_rows=00.out")
p5, = plt.plot(rows, bw, '-g^', label='read/write, cpp', ms=6, lw=2)

[rows, bw] = parse("bench=write_cpp_col=compact_rows=00.out")
p6, = plt.plot(rows, bw, '-r^', label='write, cpp', ms=6, lw=2)

plt.xlabel("Vector size (in millions)", fontsize=18)
plt.ylabel("Bandwidth (GB/s)", fontsize=18)
plt.rc("xtick", labelsize=16)
plt.rc("ytick", labelsize=16)
l1 = plt.legend([p1,p2,p3], ['read, cpp', 'write', 'read'], bbox_to_anchor=(0.,0.,.5,.5), loc=8, fontsize=16)
l2 = plt.legend([p4,p5,p6], ['read/write', 'read/write, cpp', 'write, cpp'], bbox_to_anchor=(.5,0.,.5,.5), loc=8, fontsize=16)
plt.gca().add_artist(l1) # add l1 back, which is deleted

[rows, bw] = parse("bench=gather_col=compact_rows=00.out")
plt.ylim([0,200])
plt.xlim([0,150])

plt.tight_layout()
plt.savefig.format = "pdf"
plt.savefig("bandwidth_read_write.pdf")

#-----------------------------------------
plt.clf()
lines = ['-','-','-','-']
cols = ['b','r','m','c','k','g']
sym = ['o','^','s','v','h','x']

[rows, bw] = parse("bench=gather_col=compact_rows=00.out")
p1, = plt.plot(rows, bw, '-bs', label='gather, compact', ms=6, lw=2)

[rows, bw] = parse("bench=unpack_col=compact_rows=00.out")
p2, = plt.plot(rows, bw, '-rs', label='unpack, compact', ms=6, lw=2)

[rows, bw] = parse("bench=gather_cpp_col=compact_rows=00.out")
p3, = plt.plot(rows, bw, '-b^', label='gather, cpp, compact', ms=6, lw=2)

[rows, bw] = parse("bench=gather_col=random_rows=00.out")
p4, = plt.plot(rows, bw, '-bo', label='gather, random', ms=6, lw=2)

plt.xlabel("Vector size (in millions)", fontsize=18)
plt.ylabel("Bandwidth (GB/s)", fontsize=18)
plt.rc("xtick", labelsize=16)
plt.rc("ytick", labelsize=16)
#plt.legend(bbox_to_anchor=(0,-.5,1,1), loc=9, fontsize=12)
l1 = plt.legend([p1,p2], ['gather, compact', 'unpack, compact'], loc=1, bbox_to_anchor=(0.0,.0,.5,1), fontsize=18)
l2 = plt.legend([p3,p4], ['gather, cpp, compact', 'gather, random'], loc=7, bbox_to_anchor=(0.5,.0,.5,1), fontsize=18)
plt.gca().add_artist(l1)
plt.ylim([0,200])
plt.xlim([0,150])

#plt.show()
print rows
print bw

plt.tight_layout()
plt.savefig.format = "pdf"
plt.savefig("bandwidth_gather_unpack.pdf")



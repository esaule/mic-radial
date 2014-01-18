
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
import myutil

#----------------------------------

# 3 columns: filenmame, rows, bandwidth
def plot_file(infile,outfile):
    dict = {}
    #dict['64_norcm'] = {}
    dict['96_norcm'] = {}
    #dict['64_rcm'] = {}
    dict['96_rcm'] = {}
    with open(infile, 'r') as csvfile:
        csvfile.seek(0)
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            if row == []: break
            if row[0][0] == '#': continue
            rfile = row[0]
            method = row[1]
            #print method
            r1 = float(row[2])
            r2 = float(row[3])

            try:
                #if method != "method_8a_multi_cpp":
                    dict[rfile][method].append([r1,r2])
            except:
                try:
                    #print(rfile)
                    dict[rfile][method] = []
                    dict[rfile][method].append([r1,r2])
                except:
                    pass

    for f in dict.keys():
        for m in dict[f].keys():
            dict[f][m] = zip(*dict[f][m])  # * means transpose


#### BEGINNING OF CHART

    def plot_file(xlist, ylist, col, sym, leg):
        lines = ['-','-','-','-']
        count = 0
        x = xlist
        y = ylist
        p = plt.plot(x,y, col+sym+'-', label=leg, ms=6, lw=2)
        lines.append(p)
        count += 1


    cols = ['b','r','m','c','k','g']
    syms = ['o','^','s','v','h','x']
    fi = 0
    #for f in dict.keys():
        #print "** key= ", f

    """
    for f in dict.keys():
        print "** key= ", f
        col = cols[fi]; fi+=1; mi = 0
        for m in dict[f].keys():
            print "  ++ key= ", m
            sym = syms[mi]; mi+=1
            leg = f + "_" + m
            plot_file(dict[f][m][0], dict[f][m][1], col, sym, leg)
    """

    d = dict['96_rcm']['method_8a_multi']
    plot_file(d[0], d[1], 'r', 's', "rcm, vector")
    d = dict['96_norcm']['method_8a_multi']
    plot_file(d[0], d[1], 'r', '^', "no rcm, vector")

    d = dict['96_rcm']['method_8a_multi_cpp']
    plot_file(d[0], d[1], 'g', 's', "rcm, cpp")
    d = dict['96_norcm']['method_8a_multi_cpp']
    plot_file(d[0], d[1], 'g', '^', "no rcm, cpp ")

    d = dict['96_rcm']['method_8a_base']
    plot_file(d[0], d[1], 'b', 's', "rcm, base")
    d = dict['96_norcm']['method_8a_base']
    plot_file(d[0], d[1], 'b', '^', "no rcm, base")

    plt.xlabel('Nb threads', fontsize=18)
    plt.ylabel('Performance (Gflop/s)', fontsize=18)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    #plt.title('Performance RBF-FD derivative MIC, 96^3')
    plt.legend(loc=2, fontsize=18)  # upper left
    plt.grid(True)
    #plt.ylim(0,100)
    plt.xlim([0,250])

    myutil.horizontal_line(35,  xmin=205, xmax=250, col='b', lw=4)
    myutil.horizontal_line(135, xmin=205, xmax=250, col='g', lw=4)
    plt.text(130,32.5,'peak: 35 Gflop/s', fontsize=16)
    plt.text(130,132.5,'peak: 135 Gflop/s', fontsize=16)

    plt.tight_layout()
    plt.savefig.format = "pdf"
    plt.savefig(outfile)
    plt.savefig("mic_performance_nb_threads.pdf")
#plt.show()
#### END OF CHART
#----------------------------------------------------------------------

filename = "raw2"
plot_file(filename+".dat",filename+"x.pdf")


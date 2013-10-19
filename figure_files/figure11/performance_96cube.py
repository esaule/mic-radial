
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv

#----------------------------------

# 3 columns: filenmame, rows, bandwidth
def plot_file(infile,outfile):
    dict = {}
    dict['64_norcm'] = {}
    dict['96_norcm'] = {}
    dict['64_rcm'] = {}
    dict['96_rcm'] = {}
    with open(infile, 'r') as csvfile:
        csvfile.seek(0)
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            if row == []: break
            if row[0][0] == '#': continue
            rfile = row[0]
            method = row[1]
            r1 = float(row[2])
            r2 = float(row[3])

            try:
                dict[rfile][method].append([r1,r2])
            except:
                print(rfile)
                dict[rfile][method] = []
                dict[rfile][method].append([r1,r2])

    for f in dict.keys():
        for m in dict[f].keys():
            dict[f][m] = zip(*dict[f][m])  # * means transpose


#### BEGINNING OF CHART

    def plot_file(xlist, ylist, col, sym, leg):
        lines = ['-','-','-','-']
        count = 0
        x = xlist
        y = ylist
        p = plt.plot(x,y, col+sym+'-', label=leg)
        lines.append(p)
        count += 1


    cols = ['b','r','m','c','k','g']
    syms = ['o','^','s','v','h','x']
    fi = 0
    for f in dict.keys():
        col = cols[fi]; fi+=1; mi = 0
        for m in dict[f].keys():
            sym = syms[mi]; mi+=1
            leg = f + "_" + m
            plot_file(dict[f][m][0], dict[f][m][1], col, sym, leg)

    plt.xlabel('Nb threads')
    plt.ylabel('Gflops')
    #plt.title('Performance RBF-FD derivative MIC, 96^3')
    plt.legend(loc=2, fontsize=10)  # upper left
    plt.grid(True)
    #plt.ylim(0,100)

    plt.tight_layout()
    plt.savefig.format = "pdf"
    plt.savefig(outfile)
#plt.show()
#### END OF CHART
#----------------------------------------------------------------------

filename = "raw2"
plot_file(filename+".dat",filename+".pdf")


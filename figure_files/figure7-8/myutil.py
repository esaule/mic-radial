
import matplotlib.pyplot as plt

#-----------------------------------------------
def parse_line(line):
    in_file,method = line[0].split(':')
    rcm = len(in_file.split('rcm'))
    rcm = rcm-1    # 0 if no RCM, 1 if rcm
    threads = line[1].split(',')[0].split(':')[1]
    max_flops = line[2].split(',')[0].split(':')[1]
    numbs = in_file.split('3d_')[1].split('_')
    nx = int( numbs[0].split('x')[0] )
    ny = int( numbs[1].split('y')[0] )
    nz = int( numbs[2].split('z')[0].split('z')[0] )
    return [in_file,max_flops,nx,ny,nz,rcm]
#-----------------------------------------------
def horizontal_lines():
    # worst and best cases single precision
    horizontal_line(55.8, 0, 1600)
    horizontal_line(213.3, 0, 1600)

def horizontal_line(ylev, xmin=0, xmax=1):
    x = [xmin, xmax]
    y = [ylev, ylev]
    plt.plot(x, y, 'k-',lw=2)

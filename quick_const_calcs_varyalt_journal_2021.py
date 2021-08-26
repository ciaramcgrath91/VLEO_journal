import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy import interpolate


# number of satellites
def number_of_sats(h, Ps, res):
    incl = np.arccos((-2*(h+Re)**(7/2) * rate_ss )/(3 * Re**2 * J2 *np.sqrt(mu)))
    beta = abs(np.arctan((np.sqrt(np.sin(incl)**2 - np.sin(lat)**2))/ (np.cos(incl) - rot_rate * np.cos(lat)**2) ))
    num_sats = np.ceil( h**2 * (np.sin(beta)/(yr2coverage * secs_py * res**2)) *(Edet/(Ps * A * Q * rho * tau**2)) * ((2*np.pi**2 * eqcirc * Re) / (1 - cloud_perc)))
    return num_sats

def plot_graph(x_data, y_data, x_label, y_label, y_lim, scale, ax1_col, legend=None):
    ax = plt.figure().add_subplot(111)
    grey = (.1, .1, .1)

    # cols = [
    #     "darkorange",
    #     "green",
    #     "cornflowerblue",
    # ]

    col_mono = "black"
    cols = [col_mono for x in range(10)]

    linestyles = ['-', '--', '-.', ':']

    for y in range(len(y_data)):
        ax.plot(
            x_data,
            y_data[y],
            color=cols[y],
            linestyle=linestyles[y],
            label=legend[y]
        )

    ax.set_ylim(y_lim[0], y_lim[1])
    ax.set_yscale(scale)
    ax.set_xlabel(x_label, color=(.1, .1, .1))
    ax.set_ylabel(y_label, color=grey)

    ax.legend(loc='lower right', shadow=False, facecolor='white')
    ax.grid(which="both", color=(.9, .9, .9))
    ax.yaxis.set_major_formatter(ScalarFormatter())

    ax.set_facecolor((1., 1., 1.))
    return ax



# define variables/constants
mu =  3.986004418 * 10**14    # standard gravitational parameter, m^3/s^2
Re = 6371000               # mean Earth radius, m
J2 = 1082.7 * (10**-6)        # coefficient of the Earth's gravitational zonal harmonic of the 2nd degree, -
rate_ss = 1.991063853e-7        # rate of nodal drift needed for sun-sync orbit (rad/sec)
rot_rate = 0.0000729212   # rotation rate of central body



#h = 405000 # 705000                 # constellation altitude, m
#h0 = 405000                  # reference altitude, m
P_pay = 150 # [150, 200, 250, 300]
L_e = 0.08
#Ps_list = [i*L_e for i in P_pay]
Ps = P_pay * L_e
#Ps = 12 # 80                     # power per satellite, W          # matches UoE results with power of 80 W: graph 1 (a)
#Eshot =  0.01 # 0.125 # 0.01      # energy per shot, J
#res = 20 # 20 # 30                     # ground resolution, m
res_list = [5, 10, 20, 30]
eqcirc = 40075000           # earth equatorial circumference              
secs_py = 3.154 * 10**7     # seconds in a year
#flattening = 0.00335281    # flattening value
yr2coverage = 1 #5 #1            # number of years in which to obtain full coverage
#D = 0.8                     # telescope diameter
#A = 0.5
A = 0.24
Q = 0.45                    # % quantum efficiency
rho = 0.4 #0.57                        # surface reflectance
tau = 0.8                         # % atmospheric transmittance
Edet = 0.562*10**-15 #0.281*10**-15             # energy detected at receiver

h_list = range(200000, 410000, 5)
cloud_perc = 0.5
lat = 0




# call loop to calculate minimum sats needed
num_sats_list = []
for i in range(0,len(res_list)):
    res = res_list[i]
    num_sats_list.append([])
    for h in h_list:
        num_sats = number_of_sats(h, Ps, res)
        num_sats_list[i].append(num_sats)

    #plt.plot([i/1000 for i in h_list], num_sats_list[i], label = str(res) + "m resolution")


ax0 = plot_graph(
        [i/1000 for i in h_list],
        [j for j in num_sats_list],
        'Altitude (km)',
        'Number of spacecraft required',
        (10, 5000),
        "log",
        (0.1, 0.4, 0.85),
        [str(res) + "m resolution" for res in res_list],
    )

plt.show()
#plt.savefig("Figures/num_sats.eps")      


#plt.xlabel('Altitude, km')
#plt.ylabel('Number of spacecraft required')
#plt.xlim(200, 500)
#axes = plt.gca()
#axes.set_yscale('log')
#plt.ylim(1, 5000)
#plt.ylim(0, 1000)
#plt.legend()
#plt.legend(loc = "lower right", fancybox=True, framealpha=0.5)
#plt.show()
#plt.savefig("Figures/num_sats_plot_log.jpg")





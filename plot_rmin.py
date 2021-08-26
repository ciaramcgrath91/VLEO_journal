import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def calc_min_res(h, P_pay):
    Eshot =  (2 * np.pi * Edet * h**2)/(Q * A * rho * (tau**2))                # output laser energy per shot, J
    res_min =  (Eshot/(P_pay * L_e)) * ((Re * np.sqrt(mu)) / ((Re + h)**(3/2)))
    return res_min

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




# define constants
mu =  3.986004418 * 10**14    # standard gravitational parameter, m^3/s^2
Re = 6371000               # mean Earth radius, m

# Laser parameters
P_pay = [150, 200, 250, 300]
L_e = 0.08
Edet  = 0.562 *10**-15
Q = 0.45
A = 0.24 # 0.5
rho = 0.4
tau = 0.8


# Altitude
h = [i*1000 for i in range(200,401,10)]

res_min = []
for j in range(0, len(P_pay)):
    res_min.append([])
    P = P_pay[j]
    for i in h:
        res_min[j].append(calc_min_res(i, P))
    
    #plt.plot([i/1000 for i in h], res_min[j], label = str(P) + "W")

#plt.xlabel('Altitude, km')
#plt.ylabel('Minimum possible resolution, m')
#plt.xlim(200, 500)
#plt.ylim(0, 10)
#plt.legend()
#plt.show()
#plt.savefig("Figures/min_res_plot.jpg")

ax0 = plot_graph(
        [i/1000 for i in h],
        [j for j in res_min],
        'Altitude (km)',
        'Minimum possible resolution, m',
        (0, 10),
        "linear",
        (0.1, 0.4, 0.85),
        [str(i) + "W" for i in P_pay],
    )

plt.show()
#plt.savefig("Figures/res_v_alt.eps")          
            


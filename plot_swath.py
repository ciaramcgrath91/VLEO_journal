
from matplotlib.ticker import ScalarFormatter
import numpy as np
import matplotlib.pyplot as plt

def swathfromalt2(h, res):
    """
    calculate LIDAR swath from altitude and satellite parameters etc
    returns swath in metres

    h = altitude, m
    Re = mean Earth radius, m
    mu = standard graviational paramter, m3/s2
    h0 = reference altitude used to calculate pulse power, m
    Ps = power per platform, W
    Eshot = energy per laser shot, J
    res = image resolution, m
    """

    #swath = (Ps * h0**2 * res**2 * (Re + h)**(3/2)) / (Eshot * h**2 * Re * np.sqrt(mu))
    swath = (P * A * Q * rho * tau**2 * res**2 * (Re + h)**(3/2)) / (Edet * 2 * np.pi * h**2 * Re * np.sqrt(mu))

    return swath  

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

    ax.legend(loc='upper right', shadow=False, facecolor='white')
    ax.grid(which="both", color=(.9, .9, .9))
    ax.yaxis.set_major_formatter(ScalarFormatter())

    ax.set_facecolor((1., 1., 1.))
    return ax


# define constants
mu =  3.986004418 * 10**14    # standard gravitational parameter, m^3/s^2
Re = 6371000               # mean Earth radius, m

# Laser parameters
res_list = [5, 10, 20, 30]
P_pay = 150 #[150, 200, 250, 300]
L_e = 0.08
P = P_pay * L_e
Edet  = 0.562 *10**-15
Q = 0.45
A = 0.24 #0.5
rho = 0.4
tau = 0.8


# Altitude
h = [i*1000 for i in range(200,401,10)]

swath = []
for j in range(0, len(res_list)):
    swath.append([])
    res = res_list[j]
    for i in h:
        swath[j].append(swathfromalt2(i, res))
    
    #plt.plot([i/1000 for i in h], swath[j], label = str(res) + "m resolution")

#plt.xlabel('Altitude, km')
#plt.ylabel('Swath, m')
#plt.xlim(200, 400)
#plt.ylim(0, 300)
#plt.legend()
#plt.show()
#plt.savefig("Figures/min_res_plot.jpg")

ax0 = plot_graph(
        [i/1000 for i in h],
        [j for j in swath],
        'Altitude (km)',
        'Swath, m',
        (0, 300),
        "linear",
        (0.1, 0.4, 0.85),
        [str(res) + "m resolution" for res in res_list],
    )

plt.show()
#plt.savefig("Figures/swath_v_alt.eps")            



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def calc_launch_cost(h, Isp):
    incl = np.arccos((-2*(h+Re)**(7/2) * rate_ss )/(3 * Re**2 * J2 *np.sqrt(mu)))
    beta = abs(np.arctan((np.sqrt(np.sin(incl)**2 - np.sin(lat)**2))/ (np.cos(incl) - rot_rate * np.cos(lat)**2) ))
    swath = (Ps * A * Q * rho * tau**2 * res**2 * (Re + h)**(3/2)) / (Edet * 2 * np.pi * h**2 * Re * np.sqrt(mu))
    num_sats = np.ceil( h**2 * (np.sin(beta)/(t * res**2)) *(Edet/(Ps * A * Q * rho * tau**2)) * ((2*np.pi**2 * eqcirc * Re) / (1 - cloud_perc)))
    prop_mass = (1000**gamma * L * C_D * mu * cross_sec_area * Lambda) / (2 * g0 * Isp * h**gamma * (Re + h)) 
    F_D = 0.5*Lambda *(h/1000)**(-gamma) * mu/(Re + h) * cross_sec_area * C_D
    num_thrusters = np.ceil(F_D/F_max)
    launch_mass = num_sats *(mass_dry + num_thrusters*thruster_mass + prop_mass)
    launch_cost = launch_mass * cost_per_kilo
    return launch_cost, num_sats, swath, prop_mass, num_thrusters

def plot_graph(x_data, y_data, x_label, y_label, y_lim, scale, ax1_col, legend=None):
    ax = plt.gca()
    grey = (.1, .1, .1)

    # cols = [
    #     "darkorange",
    #     "green",
    #     "cornflowerblue",
    # ]

    col_mono = ax1_col #"black"
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

    #ax.legend(loc='lower right', shadow=False, facecolor='white')
    ax.grid(which="both", color=(.9, .9, .9))
    #ax.yaxis.set_major_formatter(ScalarFormatter())
    sf = ScalarFormatter()
    sf.set_scientific(True)
    ax.yaxis.set_major_formatter(sf)
    ax.yaxis.set_minor_formatter(sf)

    ax.set_facecolor((1., 1., 1.))
    return ax




# define constants
mu =  3.986004418 * 10**14    # standard gravitational parameter, m^3/s^2
Re = 6371000               # mean Earth radius, m
secs_py = 3.154 * 10**7     # seconds in a year
g0 = 9.81
J2 = 1082.7 * (10**-6)        # coefficient of the Earth's gravitational zonal harmonic of the 2nd degree, -
rot_rate = 0.0000729212   # rotation rate of central body
rate_ss = 1.991063853e-7        # rate of nodal drift needed for sun-sync orbit (rad/sec)
eqcirc = 40075000           # earth equatorial circumference              



# Laser parameters
P_pay = 150
L_e = 0.08
Ps = P_pay * L_e
Edet  = 0.562 *10**-15
Q = 0.45
A = 0.24 # 0.5
rho = 0.4
tau = 0.8

#res_list = [5, 10, 20, 30]


# study parameters
yr2coverage = 1
t = yr2coverage * secs_py
L = 5* secs_py
cross_sec_area = 0.4 #0.5 #m
C_D = 2.2
Lambda = 10**7
gamma = 7.201
#Isp = 4000  # secs
Isp_list = [1000, 2000, 3000, 4000]
mass_dry = 112.5 #150 # kg
F_max = 1 / 1000
thruster_mass = 2.6 # kg
cloud_perc = 0.5
lat = 0


cost_per_kilo = 10000

# Altitude
h_list = [i for i in range(200000,400001,1000)]


res = 20
# call loop to calculate minimum sats needed
launch_cost_list = []
num_sats_list = []
swath_list = []
prop_mass_list = []
num_thrusters_list = []
for i in range(0,len(Isp_list)):
    Isp = Isp_list[i]
    launch_cost_list.append([])
    num_sats_list.append([])
    swath_list.append([])
    prop_mass_list.append([])
    num_thrusters_list.append([])
    for h in h_list:
        launch_cost, num_sats, swath, prop_mass, num_thrusters = calc_launch_cost(h, Isp)
        launch_cost_list[i].append(launch_cost/1000000) # get in millions $
        num_sats_list[i].append(num_sats)
        swath_list[i].append(swath)
        prop_mass_list[i].append(prop_mass)
        num_thrusters_list[i].append(num_thrusters)

    #plt.plot([k/1000 for k in h_list], [j/1000000 for j in launch_cost_list[i]], label = str(res) + "m resolution")

ax = plt.figure(figsize=(10, 5)).add_subplot(111)
ax0 = plot_graph(
        [i/1000 for i in h_list],
        [j for j in launch_cost_list],
        'Altitude (km)',
        'Launch Cost, M$',
        (30, 270),
        "log",
        "black", #(0.1, 0.4, 0.85),
        [str(i) + "s" for i in Isp_list],
    )
ax.legend(loc='lower right', shadow=False, facecolor='white')

# print solutions
print("For a resolution of " + str(res) + "m")
for i in range(0,len(launch_cost_list)):
    cost = launch_cost_list[i][launch_cost_list[i].index(min(launch_cost_list[i]))]
    altitude = h_list[launch_cost_list[i].index(min(launch_cost_list[i]))]
    sats = num_sats_list[i][launch_cost_list[i].index(min(launch_cost_list[i]))]
    swath_size = swath_list[i][launch_cost_list[i].index(min(launch_cost_list[i]))]
    prop_mass_kg = prop_mass_list[i][launch_cost_list[i].index(min(launch_cost_list[i]))]
    thrusters = num_thrusters_list[i][launch_cost_list[i].index(min(launch_cost_list[i]))]
    print("Cost $" + str(cost) + "M. Altitude " + str(altitude/1000) + "km. Num sats " + str(sats) + ". Swath: " + str(swath_size) + "m.")
    print("Propellant mass " + str(prop_mass_kg) + " kg. No. thrusters " + str(thrusters))
    print(" ")



res = 30
# call loop to calculate minimum sats needed
launch_cost_list = []
num_sats_list = []
swath_list = []
prop_mass_list = []
num_thrusters_list = []
for i in range(0,len(Isp_list)):
    Isp = Isp_list[i]
    launch_cost_list.append([])
    num_sats_list.append([])
    swath_list.append([])
    prop_mass_list.append([])
    num_thrusters_list.append([])
    for h in h_list:
        launch_cost, num_sats, swath, prop_mass, num_thrusters = calc_launch_cost(h, Isp)
        launch_cost_list[i].append(launch_cost/1000000) # get in millions $
        num_sats_list[i].append(num_sats)
        swath_list[i].append(swath)
        prop_mass_list[i].append(prop_mass)
        num_thrusters_list[i].append(num_thrusters)

ax1 = plot_graph(
        [i/1000 for i in h_list],
        [j for j in launch_cost_list],
        'Altitude (km)',
        'Launch Cost, M$',
        (30, 270),
        "log",
        (0.1, 0.4, 0.85),
        [str(i) + "s" for i in Isp_list],
    )
lines = ax.get_lines()
legend1 = plt.legend([lines[i] for i in [0,1,2,3]], ["1000s Isp", "2000s Isp", "3000s Isp", "4000s Isp"], bbox_to_anchor=[1, 0.275], loc='right')
legend2 = plt.legend([lines[i] for i in [0,4]], ["20m resolution", "30m resolution"], bbox_to_anchor=[1, 0.075], loc='right')
ax.add_artist(legend1)
ax.add_artist(legend2)


plt.show()    
#plt.savefig("Figures/launch_cost_isp2.eps")

#print results
print("For a resolution of " + str(res) + "m")
for i in range(0,len(launch_cost_list)):
    cost = launch_cost_list[i][launch_cost_list[i].index(min(launch_cost_list[i]))]
    altitude = h_list[launch_cost_list[i].index(min(launch_cost_list[i]))]
    sats = num_sats_list[i][launch_cost_list[i].index(min(launch_cost_list[i]))]
    swath_size = swath_list[i][launch_cost_list[i].index(min(launch_cost_list[i]))]
    prop_mass_kg = prop_mass_list[i][launch_cost_list[i].index(min(launch_cost_list[i]))]
    thrusters = num_thrusters_list[i][launch_cost_list[i].index(min(launch_cost_list[i]))]
    print("Cost $" + str(cost) + "M. Altitude " + str(altitude/1000) + "km. Num sats " + str(sats) + ". Swath: " + str(swath_size) + "m.")
    print("Propellant mass " + str(prop_mass_kg) + " kg. No. thrusters " + str(thrusters))
    print(" ")




            


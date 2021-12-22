import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def calc_launch_cost(h, Isp, F_max, thruster_mass, fuel_tank): # , support_mass):
    incl = np.arccos((-2*(h+Re)**(7/2) * rate_ss )/(3 * Re**2 * J2 *np.sqrt(mu)))
    beta = abs(np.arctan((np.sqrt(np.sin(incl)**2 - np.sin(lat)**2))/ (np.cos(incl) - rot_rate * np.cos(lat)**2) ))
    swath = (Ps * A * Q * rho * tau**2 * res**2 * (Re + h)**(3/2)) / (Edet * 2 * np.pi * h**2 * Re * np.sqrt(mu))
    num_sats = np.ceil( h**2 * (np.sin(beta)/(t * res**2)) *(Edet/(Ps * A * Q * rho * tau**2)) * ((2*np.pi**2 * eqcirc * Re) / (1 - cloud_perc)))
    prop_mass = (1000**gamma * L * C_D * mu * cross_sec_area * Lambda) / (2 * g0 * Isp * h**gamma * (Re + h)) 
    F_D = 0.5*Lambda *(h/1000)**(-gamma) * mu/(Re + h) * cross_sec_area * C_D
    if fuel_tank:
        tank_mass = 0
    else:
        tank_mass = 0.5 * prop_mass
    num_thrusters = max(np.ceil(F_D/F_max), np.ceil(L*F_D/(g0*Isp*max_prop_per_thruster)))

    orbit_period = 2*np.pi * np.sqrt(((h+Re)**3)/mu)
    time_in_eclipse_hrs = percentage_time_in_eclipse * orbit_period/(60*60)
    time_in_sun_hrs = (orbit_period/(60*60) - time_in_eclipse_hrs)
    prop_power = F_D * P_thrust_unit/F_max
    battery_mass = prop_power * time_in_eclipse_hrs / (specific_energy * battery_efficiency * DoD)
    solar_array_mass = (prop_power/sa_specific_power) * ((1/sa_eff1) + time_in_eclipse_hrs/(time_in_sun_hrs*sa_eff2))
    eps_mass = solar_array_mass + battery_mass

    
    sat_mass = mass_dry + num_thrusters*thruster_mass + prop_mass + tank_mass + eps_mass
    launch_mass = num_sats * sat_mass
    launch_cost = launch_mass * cost_per_kilo
    return launch_cost, num_sats, swath, prop_mass, num_thrusters, sat_mass

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

    linestyles = ['--', '-.', ':','-']

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


# EPS Parameters
DoD = 0.8
specific_energy = 120 # Whr/kg
battery_efficiency = 0.75
percentage_time_in_eclipse = 0.38
sa_specific_power = 10  # W/kg
sa_eff1 = 0.8
sa_eff2 = 0.6


# study parameters
yr2coverage = 1
t = yr2coverage * secs_py
L = 5* secs_py
cross_sec_area = 0.4 #0.5 #m
C_D = 2.2
Lambda = 10**7
gamma = 7.201
cloud_perc = 0.5
lat = 0
cost_per_kilo = 10000
mass_dry = 150 # kg

# Propulsion system specs
prop_systems_names = ["T5-GIT", "MiXI-ARCH", "ENP-R3", "BIT-3"]       # to loop through list
prop_systems_parameters = {
                    "T5-GIT": {'Isp': 3500, 'Thrust': 20*10**-3, 'Thruster mass': 27.2, 'Max prop mass': np.inf, "Power demand": 585, "Integrated Tank": False},
                    "MiXI-ARCH": {'Isp': 3200, 'Thrust': 3*10**-3, 'Thruster mass': 0.41, 'Max prop mass': np.inf, "Power demand": 86, "Integrated Tank": False},
                    "ENP-R3": {'Isp': 4000, 'Thrust': 0.9*10**-3, 'Thruster mass': 2.6, 'Max prop mass': 1.3, "Power demand": 100, "Integrated Tank": True},
                    "BIT-3" : {'Isp': 2150, 'Thrust': 1.1*10**-3, 'Thruster mass': 1.4, 'Max prop mass': 1.5, "Power demand": 75, "Integrated Tank": True} 
                    }



# Altitude
h_list = [i for i in range(200000,400001,1000)]


res = 20
# call loop to calculate minimum sats needed
launch_cost_list = []
num_sats_list = []
swath_list = []
prop_mass_list = []
num_thrusters_list = []
sat_mass_list = []
for i in range(0,len(prop_systems_names)):
    thruster = prop_systems_names[i]
    Isp = prop_systems_parameters[thruster]["Isp"]
    F_max = prop_systems_parameters[thruster]["Thrust"]
    P_thrust_unit = prop_systems_parameters[thruster]["Power demand"]
    thruster_mass = prop_systems_parameters[thruster]["Thruster mass"]
    fuel_tank = prop_systems_parameters[thruster]["Integrated Tank"] 
    max_prop_per_thruster = prop_systems_parameters[thruster]["Max prop mass"] 
    launch_cost_list.append([])
    num_sats_list.append([])
    swath_list.append([])
    prop_mass_list.append([])
    num_thrusters_list.append([])
    sat_mass_list.append([])
    for h in h_list:
        launch_cost, num_sats, swath, prop_mass, num_thrusters, sat_mass = calc_launch_cost(h, Isp, F_max, thruster_mass, fuel_tank)#, support_mass)
        if sat_mass < 250:
            launch_cost_list[i].append(launch_cost/1000000) # get in millions $
        else:
            launch_cost_list[i].append(None) # get in millions $
        num_sats_list[i].append(num_sats)
        swath_list[i].append(swath)
        prop_mass_list[i].append(prop_mass)
        num_thrusters_list[i].append(num_thrusters)
        sat_mass_list[i].append(sat_mass)


ax = plt.figure(figsize=(10, 5)).add_subplot(111)
ax0 = plot_graph(
        [i/1000 for i in h_list],
        [j for j in launch_cost_list],
        'Altitude (km)',
        'Launch Cost, M$',
        (30, 400),
        "log",
        "black", #(0.1, 0.4, 0.85),
        [i for i in prop_systems_names],
    )
ax.legend(loc='lower right', shadow=False, facecolor='white')

# print solutions
print("For a resolution of " + str(res) + "m")
for i in range(0,len(launch_cost_list)):
    cost = launch_cost_list[i][launch_cost_list[i].index(min(x for x in launch_cost_list[i] if x is not None))]
    altitude = h_list[launch_cost_list[i].index(min(x for x in launch_cost_list[i] if x is not None))]
    sats = num_sats_list[i][launch_cost_list[i].index(min(x for x in launch_cost_list[i] if x is not None))]
    swath_size = swath_list[i][launch_cost_list[i].index(min(x for x in launch_cost_list[i] if x is not None))]
    prop_mass_kg = prop_mass_list[i][launch_cost_list[i].index(min(x for x in launch_cost_list[i] if x is not None))]
    thrusters = num_thrusters_list[i][launch_cost_list[i].index(min(x for x in launch_cost_list[i] if x is not None))]
    satellite_mass = sat_mass_list[i][launch_cost_list[i].index(min(x for x in launch_cost_list[i] if x is not None))]
    print("Cost $" + str(cost) + "M. Altitude " + str(altitude/1000) + "km. Num sats " + str(sats) + ". Swath: " + str(swath_size) + "m.")
    print("Propellant mass " + str(prop_mass_kg) + " kg. No. thrusters " + str(thrusters) + ". S/c mass: " + str(satellite_mass) + "kg.")
    print(" ")



res = 30
# call loop to calculate minimum sats needed
launch_cost_list = []
num_sats_list = []
swath_list = []
prop_mass_list = []
num_thrusters_list = []
sat_mass_list = []
for i in range(0,len(prop_systems_names)):
    thruster = prop_systems_names[i]
    Isp = prop_systems_parameters[thruster]["Isp"]
    F_max = prop_systems_parameters[thruster]["Thrust"]
    P_thrust_unit = prop_systems_parameters[thruster]["Power demand"]
    thruster_mass = prop_systems_parameters[thruster]["Thruster mass"] #+ prop_systems_parameters[thruster]["Support mass"] 
    fuel_tank = prop_systems_parameters[thruster]["Integrated Tank"]
    #support_mass = prop_systems_parameters[thruster]["Support mass"]
    max_prop_per_thruster = prop_systems_parameters[thruster]["Max prop mass"] 
    launch_cost_list.append([])
    num_sats_list.append([])
    swath_list.append([])
    prop_mass_list.append([])
    num_thrusters_list.append([])
    sat_mass_list.append([])
    for h in h_list:
        launch_cost, num_sats, swath, prop_mass, num_thrusters, sat_mass = calc_launch_cost(h, Isp, F_max, thruster_mass, fuel_tank)#, support_mass)
        if sat_mass < 250:
            launch_cost_list[i].append(launch_cost/1000000) # get in millions $
        else:
            launch_cost_list[i].append(None) # get in millions $
        num_sats_list[i].append(num_sats)
        swath_list[i].append(swath)
        prop_mass_list[i].append(prop_mass)
        num_thrusters_list[i].append(num_thrusters)
        sat_mass_list[i].append(sat_mass)

ax1 = plot_graph(
        [i/1000 for i in h_list],
        [j for j in launch_cost_list],
        'Altitude (km)',
        'Launch Cost, M$',
        (30, 400),
        "log",
        (0.1, 0.4, 0.85),
        [i for i in prop_systems_names],
    )

lines = ax.get_lines()
plt.grid(True, which="both")
legend1 = plt.legend([lines[i] for i in [0,1,2,3]], prop_systems_names, bbox_to_anchor=[1, 0.275], loc='right')
legend2 = plt.legend([lines[i] for i in [0,4]], ["20m resolution", "30m resolution"], bbox_to_anchor=[1, 0.075], loc='right')
ax.add_artist(legend1)
ax.add_artist(legend2)


plt.show()    
#plt.savefig("launch_cost_prop_systems.eps")
#print results
print("For a resolution of " + str(res) + "m")
for i in range(0,len(launch_cost_list)):
    cost = launch_cost_list[i][launch_cost_list[i].index(min(x for x in launch_cost_list[i] if x is not None))]
    altitude = h_list[launch_cost_list[i].index(min(x for x in launch_cost_list[i] if x is not None))]
    sats = num_sats_list[i][launch_cost_list[i].index(min(x for x in launch_cost_list[i] if x is not None))]
    swath_size = swath_list[i][launch_cost_list[i].index(min(x for x in launch_cost_list[i] if x is not None))]
    prop_mass_kg = prop_mass_list[i][launch_cost_list[i].index(min(x for x in launch_cost_list[i] if x is not None))]
    thrusters = num_thrusters_list[i][launch_cost_list[i].index(min(x for x in launch_cost_list[i] if x is not None))]
    satellite_mass = sat_mass_list[i][launch_cost_list[i].index(min(x for x in launch_cost_list[i] if x is not None))]
    print("Cost $" + str(cost) + "M. Altitude " + str(altitude/1000) + "km. Num sats " + str(sats) + ". Swath: " + str(swath_size) + "m.")
    print("Propellant mass " + str(prop_mass_kg) + " kg. No. thrusters " + str(thrusters) + ". S/c mass: " + str(satellite_mass) + "kg.")
    print(" ")

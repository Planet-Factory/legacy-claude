# CLimate Analysis using Digital Estimations (CLAuDE)

import numpy as np 
import matplotlib.pyplot as plt
import time
import sys
import pickle
import claude_low_level_library as low_level
import claude_top_level_library as top_level
# from twitch import prime_sub

"""
CONTROL
"""

# define length of DAY (used for calculating Coriolis as well) (s)
DAY = 60 * 60 * 24

# how many degrees between latitude and longitude gridpoints
RESOLUTION = 3

# define the planet's radius (m)
PLANET_RADIUS = 6.4E6

# TOA radiation from star (W m^-2)
INSOLATION = 1370

# define surface GRAVITY for planet (m s^-2)
GRAVITY = 9.81

# tilt of rotational axis w.r.t. solar plane
AXIAL_TILT = 23.5

# length of YEAR (s)
YEAR = 365 * DAY

PRESSURE_LEVELS = (np.array([
    1000,
    950,
    900,
    800,
    700,
    600,
    500,
    400,
    350,
    300,
    250,
    200,
    150,
    100,
    75,
    50,
    25,
    10,
    5,
    2,
    1
])) * 100

NLEVELS = len(PRESSURE_LEVELS)

# timestep for initial period where the model only calculates radiative effects
DT_SPINUP = 60 * 17.2

# timestep for the main sequence where the
# model calculates velocities/advection
DT_MAIN = 60 * 9.2

# how long the model should only calculate radiative effects
SPINUP_LENGTH = 0 * DAY

"""
SMOOTHING
"""

# you probably won't need this, but there is the option to smooth out fields
# using FFTs (NB this slows down computation considerably and can
# introduce nonphysical errors)
SMOOTHING = False
SMOOTHING_PARAM_T = 1.0
SMOOTHING_PARAM_U = 0.9
SMOOTHING_PARAM_V = 0.9
SMOOTHING_PARAM_W = 0.3
SMOOTHING_PARAM_ADD = 0.3

"""
SAVE / LOADING
"""

# SAVE current state to file?
SAVE = True
SAVE_FILE = "save_file.p"

# LOAD initial state from file?
LOAD = True

INITIAL_SETUP = True

SETUP_GRIDS = True

# write to file after this many timesteps have passed
SAVE_FREQ = 100

# how many timesteps between plots (set this low if you want realtime plots,
# set this high to improve performance)
PLOT_FREQ = 5

"""
DISPLAY
"""

# display TOP down view of a POLE? showing polar plane data and regular
# gridded data
ABOVE = False

# which POLE to display - 'n' for north, 's' for south
POLE = "N"

# which vertical level to display over the POLE
ABOVE_LEVEL = 16

# display plots of output?
PLOT = True

# display raw fields for DIAGNOSTIC purposes
DIAGNOSTIC = False

# display plots of output on vertical levels?
LEVEL_PLOTS = False

# how many levels you want to see plots of (evenly distributed through column)
NPLOTS = 3

# TOP pressure level to display (i.e. trim off sponge layer)
TOP = 17

# print times taken to calculate specific processes each timestep
VERBOSE = False

"""
POLE LATITUDE LIMIT
"""

# how far north polar plane data is calculated from the south POLE
# (do not set this beyond 45!) [mirrored to north POLE as well]
POLE_LOWER_LAT_LIMIT = -75

# how far south regular gridded data is calculated
# (do not set beyond about 80) [also mirrored to north POLE]
POLE_HIGHER_LAT_LIMIT = -85

"""
STUFF :}
"""

# define coordinate arrays
LAT = np.arange(-90, 91, RESOLUTION)
LON = np.arange(0, 360, RESOLUTION)
NLAT = len(LAT)
NLON = len(LON)
LON_PLOT, LAT_PLOT = np.meshgrid(LON, LAT)
HEIGHTS_PLOT, LAT_Z_PLOT = np.meshgrid(LAT, PRESSURE_LEVELS[:TOP] / 100)
temperature_world = np.zeros((NLAT, NLON))

##########################

if not LOAD:
    # initialise arrays for various physical fields
    temperature_world += 290
    potential_temperature = np.zeros((NLAT, NLON, NLEVELS))
    u = np.zeros_like(potential_temperature)
    v = np.zeros_like(potential_temperature)
    w = np.zeros_like(potential_temperature)
    atmosp_addition = np.zeros_like(potential_temperature)

    # read temperature and density in from standard atmosphere
    with open("standard_atmosphere.txt", "r") as f:
        standard_temp = []
        standard_pressure = []

        standard_temp_append = standard_temp.append
        standard_pressure_append = standard_pressure.append

        # These var names hurt my soul.
        for x in f:
            h, t, r, p = x.split()
            standard_temp_append(float(t))
            standard_pressure_append(float(p))

    # density_profile = np.interp(
    # x=heights/1E3,xp=standard_height,fp=standard_density)
    temp_profile = np.interp(
        x=PRESSURE_LEVELS[::-1],
        xp=standard_pressure[::-1],
        fp=standard_temp[::-1]
    )[::-1]
    for k in range(NLEVELS):
        potential_temperature[:, :, k] = temp_profile[k]

    potential_temperature = low_level.t_to_theta(
        potential_temperature,
        PRESSURE_LEVELS
    )
    geopotential = np.zeros_like(potential_temperature)

if INITIAL_SETUP:
    sigma = np.zeros_like(PRESSURE_LEVELS)
    kappa = 287 / 1000
    # pride
    for index in range(len(sigma)):
        sigma[index] = 1E3 * (
            PRESSURE_LEVELS[index] / PRESSURE_LEVELS[0]
        ) ** kappa

    heat_capacity_earth = np.zeros_like(temperature_world) + 1E6

    # heat_capacity_earth[15:36,30:60] = 1E7
    # heat_capacity_earth[30:40,80:90] = 1E7

    albedo_variance = 0.001
    albedo = np.random.uniform(
        -albedo_variance,
        albedo_variance, (NLAT, NLON)
    ) + 0.2
    albedo = np.zeros((NLAT, NLON)) + 0.2

    specific_gas = 287
    thermal_diffusivity_roc = 1.5E-6

    # define planet size and various geometric constants
    circumference = 2 * np.pi * PLANET_RADIUS
    circle = np.pi * PLANET_RADIUS ** 2
    sphere = 4 * np.pi * PLANET_RADIUS ** 2

    # define how far apart the gridpoints are: note that we use central
    # difference derivatives, and so these distances are actually twice the
    # distance between gridboxes
    dy = circumference / NLAT
    dx = np.zeros(NLAT)
    coriolis = np.zeros(NLAT)  # also define the coriolis parameter here
    angular_speed = 2 * np.pi / DAY
    for index in range(NLAT):
        dx[index] = dy * np.cos(LAT[index] * np.pi / 180)
        coriolis[index] = angular_speed * np.sin(LAT[index] * np.pi / 180)

if SETUP_GRIDS:
    grid_pad = 2

    pole_low_index_S = np.where(LAT > POLE_LOWER_LAT_LIMIT)[0][0]
    pole_high_index_S = np.where(LAT > POLE_HIGHER_LAT_LIMIT)[0][0]

    # initialise grid
    polar_grid_resolution = dx[pole_low_index_S]
    size_of_grid = PLANET_RADIUS * np.cos(
        LAT[pole_low_index_S + grid_pad] * np.pi / 180.0
    )

    def get_grid():
        return np.arange(-size_of_grid, size_of_grid, polar_grid_resolution)

    """
    south POLE
    """
    grid_x_values_S = get_grid()
    grid_y_values_S = get_grid()
    grid_xx_S, grid_yy_S = np.meshgrid(grid_x_values_S, grid_y_values_S)

    grid_side_length = len(grid_x_values_S)

    grid_lat_coords_S = (
        -np.arccos(
            ((grid_xx_S ** 2 + grid_yy_S ** 2) ** 0.5) / PLANET_RADIUS
        ) * 180.0 / np.pi
    ).flatten()
    grid_lon_coords_S = (
        180.0 - np.arctan2(grid_yy_S, grid_xx_S) * 180.0 / np.pi
    ).flatten()

    polar_x_coords_S = []
    polar_y_coords_S = []
    for index in range(pole_low_index_S):
        for j in range(NLON):
            var = PLANET_RADIUS * np.cos(LAT[index] * np.pi / 180.0)
            polar_x_coords_S.append(
                var * np.sin(LON[j] * np.pi / 180.0)
            )
            polar_y_coords_S.append(
                -var * np.cos(LON[j] * np.pi / 180.0)
            )

    """
    north POLE
    """
    pole_low_index_N = np.where(LAT < -POLE_LOWER_LAT_LIMIT)[0][-1]
    pole_high_index_N = np.where(LAT < -POLE_HIGHER_LAT_LIMIT)[0][-1]

    grid_x_values_N = get_grid()
    grid_y_values_N = get_grid()
    grid_xx_N, grid_yy_N = np.meshgrid(grid_x_values_N, grid_y_values_N)

    grid_lat_coords_N = (
        np.arccos((grid_xx_N ** 2 + grid_yy_N ** 2) ** 0.5 / PLANET_RADIUS)
        * 180.0 / np.pi
    ).flatten()
    grid_lon_coords_N = (
        180.0 - np.arctan2(grid_yy_N, grid_xx_N) * 180.0 / np.pi
    ).flatten()

    polar_x_coords_N = []
    polar_y_coords_N = []
    for index in np.arange(pole_low_index_N, NLAT):
        for j in range(NLON):
            var = PLANET_RADIUS*np.cos(LAT[index]*np.pi/180.0)
            polar_x_coords_N.append(var * np.sin(LON[j] * np.pi / 180.0))
            polar_y_coords_N.append(-var * np.cos(LON[j] * np.pi / 180.0))

    indices = (
        pole_low_index_N,
        pole_high_index_N,
        pole_low_index_S,
        pole_high_index_S
    )
    grids = (
        grid_xx_N.shape[0],
        grid_xx_S.shape[0]
    )

    # create Coriolis data on north and south planes
    data = np.zeros((NLAT-pole_low_index_N + grid_pad, NLON))
    for index in np.arange(pole_low_index_N - grid_pad, NLAT):
        data[index - pole_low_index_N, :] = coriolis[index]

    coriolis_plane_N = low_level.beam_me_up_2D(
        LAT[(pole_low_index_N-grid_pad):],
        LON,
        data,
        grids[0],
        grid_lat_coords_N,
        grid_lon_coords_N
    )

    data = np.zeros((pole_low_index_S + grid_pad, NLON))
    for index in range(pole_low_index_S+grid_pad):
        data[index, :] = coriolis[index]

    coriolis_plane_S = low_level.beam_me_up_2D(
        LAT[:(pole_low_index_S+grid_pad)],
        LON,
        data,
        grids[1],
        grid_lat_coords_S,
        grid_lon_coords_S
    )

    x_dot_N = np.zeros((grids[0], grids[0], NLEVELS))
    y_dot_N = np.zeros((grids[0], grids[0], NLEVELS))
    x_dot_S = np.zeros((grids[1], grids[1], NLEVELS))
    y_dot_S = np.zeros((grids[1], grids[1], NLEVELS))

    coords = (grid_lat_coords_N, grid_lon_coords_N, grid_x_values_N,
              grid_y_values_N, polar_x_coords_N, polar_y_coords_N,
              grid_lat_coords_S, grid_lon_coords_S, grid_x_values_S,
              grid_y_values_S, polar_x_coords_S, polar_y_coords_S)

"""
LINE BREAK
"""

# INITIATE TIME
t = 0.0

# NOTE
# how potential_temperature is defined could result in it being out of bounds.

if LOAD:
    # LOAD in previous SAVE file
    (potential_temperature, temperature_world, u, v,
     w, x_dot_N, y_dot_N, x_dot_S, y_dot_S, t, albedo, tracer
     ) = pickle.load(open(SAVE_FILE, "rb"))

sample_level = 5
tracer = np.zeros_like(potential_temperature)

last_plot = t - 0.1
last_save = t - 0.1

if PLOT:
    if not DIAGNOSTIC:
        # set up PLOT
        f, ax = plt.subplots(2, figsize=(9, 9))
        f.canvas.set_window_title('CLAuDE')
        ax[0].contourf(LON_PLOT, LAT_PLOT, temperature_world, cmap="seismic")
        ax[0].streamplot(
            LON_PLOT,
            LAT_PLOT,
            u[:, :, 0],
            v[:, :, 0],
            color="white",
            density=1
        )
        test = ax[1].contourf(
            HEIGHTS_PLOT,
            LAT_Z_PLOT,
            np.transpose(
                np.mean(
                    low_level.theta_to_t(
                        potential_temperature, PRESSURE_LEVELS
                    ), axis=1
                )
            )[:TOP, :],
            cmap="seismic",
            levels=15
        )
        ax[1].contour(
            HEIGHTS_PLOT,
            LAT_Z_PLOT,
            np.transpose(
                np.mean(
                    u,
                    axis=1
                )
            )[:TOP, :],
            colors="white",
            levels=20,
            linewidths=1,
            alpha=0.8
        )
        ax[1].quiver(
            HEIGHTS_PLOT,
            LAT_Z_PLOT,
            np.transpose(
                np.mean(
                    v,
                    axis=1
                )
            )[:TOP, :],
            np.transpose(
                np.mean(
                    10 * w,
                    axis=1
                )
            )[:TOP, :],
            color="black"
        )
        plt.subplots_adjust(left=0.1, right=0.75)
        ax[0].set_title("Surface temperature")
        ax[0].set_xlim(LON.min(), LON.max())
        ax[1].set_title("Atmosphere temperature")
        ax[1].set_xlim(LAT.min(), LAT.max())
        ax[1].set_ylim((
            PRESSURE_LEVELS.max() / 100,
            PRESSURE_LEVELS[:TOP].min() / 100
        ))
        ax[1].set_yscale("log")
        ax[1].set_ylabel("Pressure (hPa)")
        ax[1].set_xlabel("Latitude")
        cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
        f.colorbar(test, cax=cbar_ax)
        cbar_ax.set_title("Temperature (K)")
    else:
        # set up PLOT
        f, ax = plt.subplots(2, 2, figsize=(9, 9))
        f.canvas.set_window_title("CLAuDE")
        ax[0, 0].contourf(
            HEIGHTS_PLOT,
            LAT_Z_PLOT,
            np.transpose(np.mean(u, axis=1))[:TOP, :],
            cmap="seismic"
        )

        ax[0, 0].set_title("u")
        ax[0, 1].contourf(
            HEIGHTS_PLOT,
            LAT_Z_PLOT,
            np.transpose(np.mean(v, axis=1))[:TOP, :],
            cmap="seismic"
        )
        ax[0, 1].set_title("v")
        ax[1, 0].contourf(
            HEIGHTS_PLOT,
            LAT_Z_PLOT,
            np.transpose(
                np.mean(w, axis=1)
            )[:TOP, :],
            cmap="seismic"
        )
        ax[1, 0].set_title("w")
        ax[1, 1].contourf(
            HEIGHTS_PLOT,
            LAT_Z_PLOT,
            np.transpose(
                np.mean(atmosp_addition, axis=1)
            )[:TOP, :],
            cmap="seismic"
        )
        ax[1, 1].set_title("atmosp_addition")

        for axis in ax.ravel():
            axis.set_ylim((
                PRESSURE_LEVELS.max() / 100, PRESSURE_LEVELS[:TOP].min() / 100
            ))
            axis.set_yscale("log")

    f.suptitle("Time {} days".format(round(t / DAY, 2)))

    if LEVEL_PLOTS:
        level_divisions = int(np.floor(NLEVELS/NPLOTS))
        level_plots_levels = range(NLEVELS)[::level_divisions][::-1]

        g, bx = plt.subplots(NPLOTS, figsize=(9, 8), sharex=True)
        g.canvas.set_window_title('CLAuDE pressure levels')
        for k, z in zip(range(NPLOTS), level_plots_levels):
            z += 1
            bx[k].contourf(
                LON_PLOT,
                LAT_PLOT,
                potential_temperature[:, :, z],
                cmap="seismic"
            )
            bx[k].set_title(str(PRESSURE_LEVELS[z] / 100) + " hPa")
            bx[k].set_ylabel("Latitude")

        bx[-1].set_xlabel("Longitude")

    plt.ion()
    plt.show()
    plt.pause(2)

    if not DIAGNOSTIC:
        ax[0].cla()
        ax[1].cla()

        if LEVEL_PLOTS:
            for k in range(NPLOTS):
                bx[k].cla()
    else:
        ax[0, 0].cla()
        ax[0, 1].cla()
        ax[1, 0].cla()
        ax[1, 1].cla()

if ABOVE:
    g, gx = plt.subplots(1, 3, figsize=(15, 5))
    plt.ion()
    plt.show()


def plotting_routine():
    quiver_padding = int(12 / RESOLUTION)

    if PLOT:
        if VERBOSE:
            before_plot = time.time()

        # update PLOT
        if not DIAGNOSTIC:
            # ax[0].contourf(LON_PLOT, LAT_PLOT, temperature_world,
            # cmap='seismic',levels=15)

            # field = np.copy(w)[:,:,sample_level]
            field = np.copy(atmosp_addition)[:, :, sample_level]
            ax[0].contourf(
                LON_PLOT,
                LAT_PLOT,
                field,
                cmap="seismic",
                levels=15
            )
            ax[0].contour(
                LON_PLOT,
                LAT_PLOT,
                tracer[:, :, sample_level],
                alpha=0.5,
                antialiased=True,
                levels=np.arange(0.01, 1.01, 0.01)
            )

            if velocity:
                ax[0].quiver(
                    LON_PLOT[::quiver_padding, ::quiver_padding],
                    LAT_PLOT[::quiver_padding, ::quiver_padding],
                    u[::quiver_padding, ::quiver_padding, sample_level],
                    v[::quiver_padding, ::quiver_padding, sample_level],
                    color="white"
                )

            # ax[0].set_title('$\it{Ground} \quad \it{temperature}$')

            ax[0].set_xlim((LON.min(), LON.max()))
            ax[0].set_ylim((LAT.min(), LAT.max()))
            ax[0].set_ylabel("Latitude")
            ax[0].axhline(y=0, color="black", alpha=0.3)
            ax[0].set_xlabel("Longitude")

            test = ax[1].contourf(HEIGHTS_PLOT, LAT_Z_PLOT, np.transpose(
                np.mean(
                    low_level.theta_to_t(
                        potential_temperature,
                        PRESSURE_LEVELS
                    ),
                    axis=1
                ))[:TOP, :],
                cmap="seismic",
                levels=15
            )

            # test = ax[1].contourf(HEIGHTS_PLOT, LAT_Z_PLOT, np.transpose(np.
            # mean(atmosp_addition,axis=1))[:TOP,:], cmap='seismic',levels=15)
            # test = ax[1].contourf(HEIGHTS_PLOT, LAT_Z_PLOT, np.transpose(np.
            # mean(potential_temperature,axis=1)), cmap='seismic',levels=15)
            ax[1].contour(
                HEIGHTS_PLOT,
                LAT_Z_PLOT,
                np.transpose(
                    np.mean(tracer, axis=1)
                )[:TOP, :],
                alpha=0.5,
                antialiased=True,
                levels=np.arange(0.001, 1.01, 0.01)
                )

            if velocity:
                ax[1].contour(
                    HEIGHTS_PLOT,
                    LAT_Z_PLOT,
                    np.transpose(
                        np.mean(u, axis=1)
                    )[:TOP, :],
                    colors="white",
                    levels=20,
                    linewidths=1,
                    alpha=0.8
                )
                ax[1].quiver(
                    HEIGHTS_PLOT,
                    LAT_Z_PLOT,
                    np.transpose(
                        np.mean(v, axis=1)
                    )[:TOP, :],
                    np.transpose(np.mean(5 * w, axis=1))[:TOP, :],
                    color="black"
                )

            ax[1].set_title("$\it{Atmospheric} \quad \it{temperature}$")
            ax[1].set_xlim((-90, 90))
            ax[1].set_ylim((
                PRESSURE_LEVELS.max() / 100,
                PRESSURE_LEVELS[:TOP].min() / 100)
            )
            ax[1].set_ylabel("Pressure (hPa)")
            ax[1].set_xlabel("Latitude")
            ax[1].set_yscale("log")
            f.colorbar(test, cax=cbar_ax)
            cbar_ax.set_title('Temperature (K)')
        else:
            ax[0, 0].contourf(
                HEIGHTS_PLOT,
                LAT_Z_PLOT,
                np.transpose(
                    np.mean(u, axis=1)
                )[:TOP, :],
                cmap="seismic"
            )
            ax[0, 0].set_title("u")
            ax[0, 1].contourf(
                HEIGHTS_PLOT,
                LAT_Z_PLOT,
                np.transpose(
                    np.mean(v, axis=1)
                )[:TOP, :],
                cmap="seismic"
            )
            ax[0, 1].set_title("v")
            ax[1, 0].contourf(
                HEIGHTS_PLOT,
                LAT_Z_PLOT,
                np.transpose(
                    np.mean(w, axis=1)
                )[:TOP, :],
                cmap="seismic"
            )
            ax[1, 0].set_title("w")
            ax[1, 1].contourf(
                HEIGHTS_PLOT,
                LAT_Z_PLOT,
                np.transpose(
                    np.mean(atmosp_addition, axis=1)
                )[:TOP, :],
                cmap="seismic"
            )
            ax[1, 1].set_title("atmosp_addition")

            for axis in ax.ravel():
                axis.set_ylim((
                    PRESSURE_LEVELS.max() / 100,
                    PRESSURE_LEVELS[:TOP].min() / 100
                ))
                axis.set_yscale("log")

        f.suptitle("Time {} days".format(round(t / DAY, 2)))

        if LEVEL_PLOTS:
            for k, z in zip(range(NPLOTS), level_plots_levels):	
                z += 1
                bx[k].contourf(
                    LON_PLOT,
                    LAT_PLOT,
                    potential_temperature[:, :, z],
                    cmap="seismic",
                    levels=15
                )
                bx[k].quiver(
                    LON_PLOT[::quiver_padding, ::quiver_padding],
                    LAT_PLOT[::quiver_padding, ::quiver_padding],
                    u[::quiver_padding, ::quiver_padding, z],
                    v[::quiver_padding, ::quiver_padding, z],
                    color="white"
                )
                bx[k].set_title(str(round(PRESSURE_LEVELS[z] / 100)) + " hPa")
                bx[k].set_ylabel("Latitude")
                bx[k].set_xlim((LON.min(), LON.max()))
                bx[k].set_ylim((LAT.min(), LAT.max()))

            bx[-1].set_xlabel("Longitude")

    if ABOVE and velocity:
        gx[0].set_title("Original data")
        gx[1].set_title("Polar plane")
        gx[2].set_title("Reprojected data")

        g.suptitle("Time {} days".format(round(t / DAY, 2)))

        gx[0].set_title("temperature")

        if POLE.lower() == 's':
            gx[0].contourf(
                LON,
                LAT[:pole_low_index_S],
                potential_temperature[:pole_low_index_S, :, ABOVE_LEVEL]
            )

            gx[1].set_title("polar_plane_advect")
            polar_temps = low_level.beam_me_up(
                LAT[:pole_low_index_S],
                LON,
                potential_temperature[:pole_low_index_S, :, :],
                grids[1],
                grid_lat_coords_S,
                grid_lon_coords_S
            )
            output = low_level.beam_me_up(
                LAT[:pole_low_index_S],
                LON,
                south_reprojected_addition,
                grids[1],
                grid_lat_coords_S,
                grid_lon_coords_S
            )

            gx[1].contourf(
                grid_x_values_S / 1E3,
                grid_y_values_S / 1E3,
                output[:, :, ABOVE_LEVEL]
            )
            gx[1].contour(
                grid_x_values_S / 1E3,
                grid_y_values_S / 1E3,
                polar_temps[:, :, ABOVE_LEVEL],
                colors="white",
                levels=20,
                linewidths=1,
                alpha=0.8
            )
            gx[1].quiver(
                grid_x_values_S / 1E3, grid_y_values_S / 1E3,
                x_dot_S[:, :, ABOVE_LEVEL],
                y_dot_S[:, :, ABOVE_LEVEL]
            )

            gx[1].add_patch(
                plt.Circle(
                    (0, 0),
                    PLANET_RADIUS * np.cos(
                        LAT[pole_low_index_S] * np.pi / 180.0
                    ) / 1E3,
                    color="r",
                    fill=False
                )
            )
            gx[1].add_patch(plt.Circle((0,0),PLANET_RADIUS*np.cos(LAT[pole_high_index_S]*np.pi/180.0)/1E3,color='r',fill=False))

            gx[2].set_title("south_addition_smoothed")
            gx[2].contourf(LON,LAT[:pole_low_index_S],south_addition_smoothed[:pole_low_index_S,:,ABOVE_LEVEL])
            # gx[2].contourf(LON,LAT[:pole_low_index_S],u[:pole_low_index_S,:,ABOVE_LEVEL])
            gx[2].quiver(LON[::5],LAT[:pole_low_index_S],u[:pole_low_index_S,::5,ABOVE_LEVEL],v[:pole_low_index_S,::5,ABOVE_LEVEL])
        else:
            gx[0].contourf(LON,LAT[pole_low_index_N:],potential_temperature[pole_low_index_N:,:,ABOVE_LEVEL])

            gx[1].set_title('polar_plane_advect')
            polar_temps = low_level.beam_me_up(LAT[pole_low_index_N:],LON,np.flip(potential_temperature[pole_low_index_N:,:,:],axis=1),grids[0],grid_lat_coords_N,grid_lon_coords_N)
            output = low_level.beam_me_up(LAT[pole_low_index_N:],LON,north_reprojected_addition,grids[0],grid_lat_coords_N,grid_lon_coords_N)
            gx[1].contourf(grid_x_values_N/1E3,grid_y_values_N/1E3,output[:,:,ABOVE_LEVEL])
            gx[1].contour(grid_x_values_N/1E3,grid_y_values_N/1E3,polar_temps[:,:,ABOVE_LEVEL],colors='white',levels=20,linewidths=1,alpha=0.8)
            gx[1].quiver(grid_x_values_N/1E3,grid_y_values_N/1E3,x_dot_N[:,:,ABOVE_LEVEL],y_dot_N[:,:,ABOVE_LEVEL])
            
            gx[1].add_patch(plt.Circle((0,0),PLANET_RADIUS*np.cos(LAT[pole_low_index_N]*np.pi/180.0)/1E3,color='r',fill=False))
            gx[1].add_patch(plt.Circle((0,0),PLANET_RADIUS*np.cos(LAT[pole_high_index_N]*np.pi/180.0)/1E3,color='r',fill=False))

            gx[2].set_title("south_addition_smoothed")
            # gx[2].contourf(LON,LAT[pole_low_index_N:],north_addition_smoothed[:,:,ABOVE_LEVEL])
            gx[2].contourf(LON,LAT[pole_low_index_N:],u[pole_low_index_N:,:,ABOVE_LEVEL])
            gx[2].quiver(LON[::5],LAT[pole_low_index_N:],u[pole_low_index_N:,::5,ABOVE_LEVEL],v[pole_low_index_N:,::5,ABOVE_LEVEL])

    # clear plots
    if PLOT or ABOVE:
        plt.pause(0.001)

    if PLOT:
        if not DIAGNOSTIC:
            ax[0].cla()
            ax[1].cla()
            cbar_ax.cla()  
        else:
            ax[0,0].cla()
            ax[0,1].cla()
            ax[1,0].cla()
            ax[1,1].cla()

        if LEVEL_PLOTS:
            for k in range(NPLOTS):
                bx[k].cla()

        if VERBOSE:
            time_taken = float(round(time.time() - before_plot, 3))
            print('Plotting: ',str(time_taken),'s')

    if ABOVE:
        gx[0].cla()
        gx[1].cla()
        gx[2].cla()

while True:

	initial_time = time.time()

	if t < SPINUP_LENGTH:
		dt = DT_SPINUP
		velocity = False
	else:
		dt = DT_MAIN
		velocity = True

	# print current time in simulation to command line
	print("+++ t = " + str(round(t/DAY,2)) + " days +++")
	print('T: ',round(temperature_world.max()-273.15,1),' - ',round(temperature_world.min()-273.15,1),' C')
	print('U: ',round(u.max(),2),' - ',round(u.min(),2),' V: ',round(v.max(),2),' - ',round(v.min(),2),' W: ',round(w.max(),2),' - ',round(w.min(),4))

	tracer[40,50,sample_level] = 1
	tracer[20,50,sample_level] = 1

	if VERBOSE: before_radiation = time.time()
	temperature_world, potential_temperature = top_level.radiation_calculation(temperature_world, potential_temperature, PRESSURE_LEVELS, heat_capacity_earth, albedo, INSOLATION, LAT, LON, t, dt, DAY, YEAR, AXIAL_TILT)
	if SMOOTHING: potential_temperature = top_level.smoothing_3D(potential_temperature,SMOOTHING_PARAM_T)
	if VERBOSE:
		time_taken = float(round(time.time() - before_radiation,3))
		print('Radiation: ',str(time_taken),'s')

	diffusion = top_level.laplacian_2d(temperature_world,dx,dy)
	diffusion[0,:] = np.mean(diffusion[1,:],axis=0)
	diffusion[-1,:] = np.mean(diffusion[-2,:],axis=0)
	temperature_world -= dt*1E-5*diffusion

	# update geopotential field
	geopotential = np.zeros_like(potential_temperature)
	for k in np.arange(1,NLEVELS):	geopotential[:,:,k] = geopotential[:,:,k-1] - potential_temperature[:,:,k]*(sigma[k]-sigma[k-1])

	if velocity:

		if VERBOSE:	before_velocity = time.time()
		
		u_add,v_add = top_level.velocity_calculation(u,v,w,PRESSURE_LEVELS,geopotential,potential_temperature,coriolis,GRAVITY,dx,dy,dt)

		if VERBOSE:	
			time_taken = float(round(time.time() - before_velocity,3))
			print('Velocity: ',str(time_taken),'s')

		if VERBOSE:	before_projection = time.time()
		
		grid_velocities = (x_dot_N,y_dot_N,x_dot_S,y_dot_S)
	
		u_add,v_add,north_reprojected_addition,south_reprojected_addition,x_dot_N,y_dot_N,x_dot_S,y_dot_S = top_level.polar_planes(u,v,u_add,v_add,potential_temperature,geopotential,grid_velocities,indices,grids,coords,coriolis_plane_N,coriolis_plane_S,grid_side_length,PRESSURE_LEVELS,LAT,LON,dt,polar_grid_resolution,GRAVITY)
		
		u += u_add
		v += v_add

		if SMOOTHING: u = top_level.smoothing_3D(u,SMOOTHING_PARAM_U)
		if SMOOTHING: v = top_level.smoothing_3D(v,SMOOTHING_PARAM_V)

		x_dot_N,y_dot_N,x_dot_S,y_dot_S = top_level.update_plane_velocities(LAT,LON,pole_low_index_N,pole_low_index_S,np.flip(u[pole_low_index_N:,:,:],axis=1),np.flip(v[pole_low_index_N:,:,:],axis=1),grids,grid_lat_coords_N,grid_lon_coords_N,u[:pole_low_index_S,:,:],v[:pole_low_index_S,:,:],grid_lat_coords_S,grid_lon_coords_S)
		
		if VERBOSE:	
			time_taken = float(round(time.time() - before_projection,3))
			print('Projection: ',str(time_taken),'s')

		### allow for thermal advection in the atmosphere
		if VERBOSE:	before_advection = time.time()



		if VERBOSE: before_w = time.time()
		# using updated u,v fields calculated w
		# https://www.sjsu.edu/faculty/watkins/omega.htm
		w = top_level.w_calculation(u,v,w,PRESSURE_LEVELS,geopotential,potential_temperature,coriolis,GRAVITY,dx,dy,dt)
		if SMOOTHING: w = top_level.smoothing_3D(w,SMOOTHING_PARAM_W,0.25)

		theta_N = low_level.beam_me_up(LAT[pole_low_index_N:],LON,potential_temperature[pole_low_index_N:,:,:],grids[0],grid_lat_coords_N,grid_lon_coords_N)
		w_N = top_level.w_plane(x_dot_N,y_dot_N,theta_N,PRESSURE_LEVELS,polar_grid_resolution,GRAVITY)
		w_N = np.flip(low_level.beam_me_down(LON,w_N,pole_low_index_N, grid_x_values_N, grid_y_values_N,polar_x_coords_N, polar_y_coords_N),axis=1)
		w[pole_low_index_N:,:,:] = low_level.combine_data(pole_low_index_N,pole_high_index_N,w[pole_low_index_N:,:,:],w_N,LAT)
		
		w_S = top_level.w_plane(x_dot_S,y_dot_S,low_level.beam_me_up(LAT[:pole_low_index_S],LON,potential_temperature[:pole_low_index_S,:,:],grids[1],grid_lat_coords_S,grid_lon_coords_S),PRESSURE_LEVELS,polar_grid_resolution,GRAVITY)
		w_S = low_level.beam_me_down(LON,w_S,pole_low_index_S, grid_x_values_S, grid_y_values_S,polar_x_coords_S, polar_y_coords_S)
		w[:pole_low_index_S,:,:] = low_level.combine_data(pole_low_index_S,pole_high_index_S,w[:pole_low_index_S,:,:],w_S,LAT)

		# for k in np.arange(1,NLEVELS-1):
		# 	north_reprojected_addition[:,:,k] += 0.5*(w_N[:,:,k] + abs(w_N[:,:,k]))*(potential_temperature[pole_low_index_N:,:,k] - potential_temperature[pole_low_index_N:,:,k-1])/(PRESSURE_LEVELS[k] - PRESSURE_LEVELS[k-1])
		# 	north_reprojected_addition[:,:,k] += 0.5*(w_N[:,:,k] - abs(w_N[:,:,k]))*(potential_temperature[pole_low_index_N:,:,k+1] - potential_temperature[pole_low_index_N:,:,k])/(PRESSURE_LEVELS[k+1] - PRESSURE_LEVELS[k])

		# 	south_reprojected_addition[:,:,k] += 0.5*(w_S[:,:,k] + abs(w_S[:,:,k]))*(potential_temperature[:pole_low_index_S,:,k] - potential_temperature[:pole_low_index_S,:,k-1])/(PRESSURE_LEVELS[k] - PRESSURE_LEVELS[k-1])
		# 	south_reprojected_addition[:,:,k] += 0.5*(w_S[:,:,k] - abs(w_S[:,:,k]))*(potential_temperature[:pole_low_index_S,:,k+1] - potential_temperature[:pole_low_index_S,:,k])/(PRESSURE_LEVELS[k+1] - PRESSURE_LEVELS[k])

		w[:,:,18:] *= 0

		if VERBOSE:	
			time_taken = float(round(time.time() - before_w,3))
			print('Calculate w: ',str(time_taken),'s')

		#################################

		atmosp_addition = top_level.divergence_with_scalar(potential_temperature,u,v,w,dx,dy,PRESSURE_LEVELS)

		# combine addition calculated on polar grid with that calculated on the cartestian grid
		north_addition_smoothed = low_level.combine_data(pole_low_index_N,pole_high_index_N,atmosp_addition[pole_low_index_N:,:,:],north_reprojected_addition,LAT)
		south_addition_smoothed = low_level.combine_data(pole_low_index_S,pole_high_index_S,atmosp_addition[:pole_low_index_S,:,:],south_reprojected_addition,LAT)
		
		# add the blended/combined addition to global temperature addition array
		atmosp_addition[:pole_low_index_S,:,:] = south_addition_smoothed
		atmosp_addition[pole_low_index_N:,:,:] = north_addition_smoothed

		if SMOOTHING: atmosp_addition = top_level.smoothing_3D(atmosp_addition,SMOOTHING_PARAM_ADD)

		atmosp_addition[:,:,17] *= 0.5
		atmosp_addition[:,:,18:] *= 0

		potential_temperature -= dt*atmosp_addition

		###################################################################

		tracer_addition = top_level.divergence_with_scalar(tracer,u,v,w,dx,dy,PRESSURE_LEVELS)
		tracer_addition[:4,:,:] *= 0
		tracer_addition[-4:,:,:] *= 0

		for k in np.arange(1,NLEVELS-1):

			tracer_addition[:,:,k] += 0.5*(w[:,:,k] - abs(w[:,:,k]))*(tracer[:,:,k] - tracer[:,:,k-1])/(PRESSURE_LEVELS[k] - PRESSURE_LEVELS[k-1])
			tracer_addition[:,:,k] += 0.5*(w[:,:,k] + abs(w[:,:,k]))*(tracer[:,:,k+1] - tracer[:,:,k])/(PRESSURE_LEVELS[k] - PRESSURE_LEVELS[k-1])

		tracer -= dt*tracer_addition

		diffusion = top_level.laplacian_3d(potential_temperature,dx,dy,PRESSURE_LEVELS)
		diffusion[0,:,:] = np.mean(diffusion[1,:,:],axis=0)
		diffusion[-1,:,:] = np.mean(diffusion[-2,:,:],axis=0)
		potential_temperature -= dt*1E-4*diffusion

		###################################################################

		if VERBOSE:	
			time_taken = float(round(time.time() - before_advection,3))
			print('Advection: ',str(time_taken),'s')

	if t-last_plot >= PLOT_FREQ*dt:
		plotting_routine()
		last_plot = t

	if SAVE:
		if t-last_save >= SAVE_FREQ*dt:
			pickle.dump((potential_temperature,temperature_world,u,v,w,x_dot_N,y_dot_N,x_dot_S,y_dot_S,t,albedo,tracer), open(SAVE_FILE,"wb"))
			last_save = t

	if np.isnan(u.max()):
		sys.exit()

	# advance time by one timestep
	t += dt

	time_taken = float(round(time.time() - initial_time,3))

	print('Time: ',str(time_taken),'s')
	# print('777777777777777777')
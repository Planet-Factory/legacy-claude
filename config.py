import numpy as np


class Config:
    """Configure Claude!
    """

    # Length of DAY (used for calculating Coriolis as well) (s)
    DAY = 60 * 60 * 24
    # Length of YEAR (s)
    YEAR = 365 * DAY
    # Degrees between latitude and longitude gridpoints
    RESOLUTION = 3
    # The planet's radius (m)
    PLANET_RADIUS = 6.4E6
    # TOA radiation from star (W m^-2)
    INSOLATION = 1370
    # Surface GRAVITY for planet (m s^-2)
    GRAVITY = 9.81
    # Tilt of rotational axis w.r.t. solar plane
    AXIAL_TILT = 23.5
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
    # timestep for initial period where the model only
    # calculates radiative effects
    DT_SPINUP = 60 * 17.2
    # timestep for the main sequence where the
    # model calculates velocities/advection
    DT_MAIN = 60 * 9.2
    # how long the model should only calculate radiative effects
    SPINUP_LENGTH = 0 * DAY

    """
    SMOOTHING
    """

    # you probably won't need this, but there is the option to smooth out
    # fields using FFTs (NB this slows down computation considerably and can
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
    # how many timesteps between plots (set this low if you want realtime
    # plots, set this high to improve performance)
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
    # how many levels you want to see plots of
    # (evenly distributed through column)
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

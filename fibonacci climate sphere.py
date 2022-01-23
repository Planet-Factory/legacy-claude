from dataclasses import dataclass
import math, os, sys, math
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, spatial
from matplotlib.patches import Rectangle


# Constants -------------------------------------------------------------------

GOLDEN_RATIO_RAD = math.pi * (3. - math.sqrt(5.))

PLANET_RADIUS = 6.4E6
PLANET_ALBEDO = 0.3
HEAT_CAPACITY = 1E5

DAY_IN_SECONDS = 60 * 60 * 24
YEAR_IN_SECONDS = 365.25 * DAY_IN_SECONDS
TIMESTEP = 18 * 60

SOLAR_CONSTANT = 1370

PLOTTING_RESOLUTION = 75

# -----------------------------------------------------------------------------
class Planet:
    """This should be a single instance object to contain planetary properties and operations."""

    def __init__(self, num_locations: int) -> None:
        """Generate the fibonacci sphere and constructs the atmosphere for the planet."""
        self.num_points = num_locations
        self._generate_fibonacci_sphere()
        self._construct_atmosphere()
        pass

    def _generate_fibonacci_sphere(self) -> None:
        """PRIVATE.

        Generates a Fibonacci sphere where the number of points is equivalent
        to num_locations. The function provides the properties self.locations,
        self.latitudes and self.longitudes. self.locations is a list of tuples
        of latitudes and longitudes.
        """
        self.locations = []
        self.latitudes = []
        self.longitudes = []

        for i in range(self.num_points):

            y = 1 - (i / float(self.num_points - 1)) * 2  # y goes from 1 to -1
            radius_at_y = math.sqrt(1 - y * y)

            golden_ang_delta = GOLDEN_RATIO_RAD * i  # golden angle increment

            x = math.cos(golden_ang_delta) * radius_at_y
            z = math.sin(golden_ang_delta) * radius_at_y

            theta = math.acos(z)
            varphi = np.arctan2(radius_at_y, x) + np.pi

            threshold = 0.0*np.pi

            if theta > threshold and theta < np.pi-threshold:
                # points.append((x, y, z))
                self.locations.append((varphi,theta))
                self.latitudes.append(theta)
                self.longitudes.append(varphi)

        self.pixels = [Pixel(vartheta, phi) for vartheta, phi in self.locations]
        print("Fibonacci sphere generated")

        return

    def _construct_atmosphere(self) -> None:
        """PRIVATE.

        Initialises a list of Pixel objects for each location.
        """
        self.atmosphere = [Pixel(vartheta, phi) for vartheta, phi in self.locations]
        print("Atmosphere constructed")

        return

    # PUBLIC
    def get_temperatures(self) -> list:
        """Get a list of temperatures for all locations."""
        self.temperatures = [pixel.temperature for pixel in self.atmosphere]
        
        return self.temperatures

    def get_zonal_velocities(self) -> list:
        """Get a list of zonal velocities for all locations."""
        self.zonal_velocities = [pixel.zonal_velocity for pixel in self.atmosphere]
        
        return self.zonal_velocities

    def get_merdional_velocities(self) -> list:
        """Get a list of meridonal velocities for all locations."""
        self.meridional_velocities = [pixel.meridional_velocity for pixel in self.atmosphere]
        
        return self.meridional_velocities

    def update(self, sun_lon: float):
        """Use to update the planet via an iteration. Currently updates temperate, velocity and advects."""
        for location in self.atmosphere.locations:
            location.update_temp(sun_lon)
            location.update_velocity()
            location.advect()
        
        return
@dataclass
class Pixel:
    """Contains all location-specific properties and operations."""

    latitude: float
    longitude: float
    temperature: float
    zonal_velocity: float
    meridional_velocity: float
    coreolis_force: float
    
    def __init__(self, longitude: float, latitude: float) -> None:
        """Set initial property values for: latitude, longitude, temperature, zonal velocity, meridional velocity and coreolis force."""
        self.latitude = latitude
        self.longitude = longitude 
        self.temperature = 270 + 20 * np.sin(self.latitude)

        self.zonal_velocity = 0
        self.meridional_velocity = 0
        self.coreolis_force = 1E-5 * np.cos(self.latitude)
        
        return

    def update_temp(self, sun_lon: float) -> None:
        """Perform an operation to update the temperate based on calculation <REF DOCS>."""
        self.temperature += TIMESTEP * (
            SOLAR_CONSTANT * (1-PLANET_ALBEDO) * max(0,np.sin(self.latitude)) * max(0,np.sin(self.longitude-sun_lon)) 
            - (5.67E-8) * (self.temperature ** 4)
            ) / HEAT_CAPACITY
       
        return

    def update_velocity(self) -> None:
        """Perform an operation to update the zonal and meridional velocities based on calculation <REF DOCS>."""
        self.zonal_velocity -= TIMESTEP * ( 
            self.zonal_velocity * self._field_d_lon(self.zonal_velocity) 
            + self.meridional_velocity * self._field_d_lat(self.zonal_velocity) 
            + self.coreolis_force * self.meridional_velocity 
            + self._field_d_lon(self.temperature) 
            )
        self.meridional_velocity -= TIMESTEP * (
            self.zonal_velocity * self._field_d_lon(self.meridional_velocity) 
            + self.meridional_velocity * self._field_d_lat(self.meridional_velocity) 
            - self.coreolis_force * self.zonal_velocity 
            + self._field_d_lat(self.temperature) 
            )
        
        return

    def advect(self) -> None:
        """Perform an operation to calculate advection effects based on calculation <REF DOCS>."""
        self.temperature -= TIMESTEP*( 
            self.temperature * self._field_d_lon(self.zonal_velocity) 
            + self.zonal_velocity * self._field_d_lon(self.temperature)
            + self.temperature * self._field_d_lat(self.meridional_velocity) 
            + self.meridional_velocity * self._field_d_lat(self.temperature) 
            )
        
        return

    # PRIVATE
    def _field_d_lat(self, interpolated_field: interpolate.SmoothSphereBivariateSpline) -> np.ndarray:
        """PRIVATE.

        Do something stupid with an interpolated field for a proxy some shit idk.
        """
        return interpolated_field(self.latitude, self.longitude, dphi=1)[0] / PLANET_RADIUS
    
    def _field_d_lon(self, interpolated_field: interpolate.SmoothSphereBivariateSpline) -> np.ndarray:
        """PRIVATE.

        Do something stupid with an interpolated field for a proxy some shit idk.
        """
        return interpolated_field(self.latitude, self.longitude, dphi=1)[0] / (PLANET_RADIUS * np.sin(self.latitude))

class Plotter:
    """Public class to use as an object-oriented way to plot on each iteration and initialise the plot."""

    def __init__(self, num_points: int) -> None:
        """Initialise the plot and sets x and y limits, and generates the numpy grids used in plotting."""
        plt.ion()
        plt.xlim((0,2*np.pi))
        plt.ylim((0,np.pi))
        plt.title(str(num_points)+' points')

        self.lons_grid = np.linspace(0, 2*np.pi, 2*PLOTTING_RESOLUTION)
        self.lats_grid = np.linspace(0, np.pi, PLOTTING_RESOLUTION)
        self.lons_grid_gridded, self.lats_grid_gridded = np.meshgrid(self.lons_grid, self.lats_grid)
        
        return

    def _interpolate(self, planet: Planet) -> np.ndarray:
        """Private method to interpolate the temperature and velocity fields using Smooth Sphere Bivariate Spline."""
        interpolated_temperatures = interpolate.SmoothSphereBivariateSpline(planet.latitudes, planet.longitudes, planet.get_temperatures(), s=4)
        interpolated_zonal_velocities = interpolate.SmoothSphereBivariateSpline(planet.latitudes, planet.longitudes, planet.get_zonal_velocities(), s=4)
        interpolated_merdional_velocities = interpolate.SmoothSphereBivariateSpline(planet.latitudes, planet.longitudes, planet.get_merdional_velocities(), s=4)

        interpolated_temperatures = interpolated_temperatures(self.lats_grid, self.lons_grid)
        interpolated_zonal_velocities = interpolated_zonal_velocities(self.lats_grid, self.lons_grid)
        interpolated_merdional_velocities = interpolated_merdional_velocities(self.lats_grid, self.lons_grid)

        return interpolated_temperatures, interpolated_zonal_velocities, interpolated_merdional_velocities

    def plot(self, planet: Planet) -> None:
        """Call after updating planet to plot the most up-to-date data."""
        plt.cla()

        temperatures, zonal_velocities, merdional_velocities = self._interpolate(planet)
        
        quiver_resample = 4
        plt.pcolormesh(self.lons_grid_gridded, self.lats_grid_gridded, temperatures)
        plt.gca().add_patch(Rectangle((0,0), 2*np.pi, np.pi, linewidth=1, edgecolor='w', facecolor='none'))
        plt.quiver(
            self.lons_grid_gridded[::quiver_resample,::quiver_resample],
            self.lats_grid_gridded[::quiver_resample,::quiver_resample], 
            zonal_velocities[::quiver_resample,::quiver_resample], 
            merdional_velocities[::quiver_resample,::quiver_resample]
        )
        plt.scatter(planet.longitudes, planet.latitudes, s=0.5, color='black')
        
        plt.pause(0.01)

        print('T: ',round(temperatures.max()-273.15,1),' - ',round(temperatures.min()-273.15,1),' C')
        print('zonal_velocity: ',round(zonal_velocities.max(),2),' - ',round(zonal_velocities.min(),2),' meridional_velocity: ',round(merdional_velocities.max(),2),' - ',round(merdional_velocities.min(),2))
        
        if np.isinf(temperatures.max()):
            sys.exit()
        if np.isnan(zonal_velocities.max()):
            sys.exit()

        return

# Main ------------------------------------------------------------------------

if __name__ == "__main__":
    # NOTE THIS DOES NOT FUNCTION PROPERLY - MORE IS AN EXAMPLE OF STYLE
    sun_lon = 0
    time = 0
    
    planet = Planet(2500)

    plotter = Plotter()
    plotter.plot(planet)
    
    while True:
        print('TIME: ',str(time/DAY_IN_SECONDS).zfill(2), "days")

        planet.update(sun_lon)
        plotter.plot(planet)

        sun_lon += TIMESTEP * 2 * np.pi / DAY_IN_SECONDS
        time += TIMESTEP

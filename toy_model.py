# toy model for use on stream
# Please give me your Twitch prime sub!

# CLimate Analysis using Digital Estimations (CLAuDE)

import numpy as np 
import matplotlib.pyplot as plt
import time, sys

# define temporal parameters, including the length of time between calculation of fields and the length of a day on the planet (used for calculating Coriolis as well)
day = 60*60*24
dt = 60*6														###### <----- TIMESTEP

# power incident on (lat,lon) at time t
def solar(insolation, lat, lon, t):
	sun_longitude = -t % day
	sun_longitude *= 360/day
	value = insolation*np.cos(lat*np.pi/180)*np.cos((lon-sun_longitude)*np.pi/180)
	if value < 0:	return 0
	else:	return value

t = 0

# how many degrees between latitude and longitude gridpoints
resolution = 3

# define coordinate arrays
lat = np.arange(-90,91,resolution)
lon = np.arange(0,360,resolution)
nlat = len(lat)
nlon = len(lon)
lon_plot, lat_plot = np.meshgrid(lon, lat)

# initialise arrays for various physical fields
temperature_planet = np.zeros((nlat,nlon)) + 270
temperature_atmosp = np.zeros((nlat,nlon)) + 270
albedo = np.zeros((nlat,nlon)) + 0.5
heat_capacity_earth = np.zeros((nlat,nlon)) + 1E5
air_pressure = np.zeros((nlat,nlon))
u = np.zeros((nlat,nlon))
v = np.zeros((nlat,nlon))
air_density = np.zeros_like(air_pressure) + 1.3

# custom oceans with lower albedo and higher heat capacity
ocean = False
if ocean:
	albedo[5:55,9:20] = 0.2
	albedo[23:50,45:70] = 0.2
	albedo[2:30,85:110] = 0.2
	heat_capacity_earth[5:55,9:20] = 1E6
	heat_capacity_earth[23:50,45:70] = 1E6
	heat_capacity_earth[2:30,85:110] = 1E6

# define physical constants
epsilon = 0.75
heat_capacity_atmos = 1E3
specific_gas = 287
thermal_diffusivity_air = 20E-6
thermal_diffusivity_roc = 1.5E-6
insolation = 1370
sigma = 5.67E-8

# define planet size and various geometric constants
planet_radius = 6.4E6
circumference = 2*np.pi*planet_radius
circle = np.pi*planet_radius**2
sphere = 4*np.pi*planet_radius**2

# define how far apart the gridpoints are: note that we use central difference derivatives, and so these distances are actually twice the distance between gridboxes
dy = circumference/nlat
dx = np.zeros(nlat)
coriolis = np.zeros(nlat)	# also define the coriolis parameter here
angular_speed = 2*np.pi/day
for i in range(nlat):
	dx[i] = dy*np.cos(lat[i]*np.pi/180)
	coriolis[i] = angular_speed*np.sin(lat[i]*np.pi/180)

# define various useful differential functions:
# gradient of scalar field a in the local x direction at point i,j
def scalar_gradient_x(a,i,j):
	return (a[i,(j+1)%nlon]-a[i,(j-1)%nlon])/dx[i]
# gradient of scalar field a in the local y direction at point i,j
def scalar_gradient_y(a,i,j):
	if i == 0 or i == nlat-1:
		return 0
	else:
		return (a[i+1,j]-a[i-1,j])/dy
# laplacian of scalar field a in the local x direction
def laplacian(a):
	output = np.zeros_like(a)
	for i in np.arange(1,len(a[:,0])-1):
		for j in range(len(a[0,:])):
			output[i,j] = (scalar_gradient_x(a,i,(j+1)%nlon) - scalar_gradient_x(a,i,(j-1)%nlon)/dx[i]) + (scalar_gradient_y(a,i+1,j) - scalar_gradient_y(a,i-1,j))/dy
	return output
# divergence of (a*u) where a is a scalar field and u is the atmospheric velocity field
def divergence_with_scalar(a):
	output = np.zeros_like(a)
	for i in range(len(a[:,0])):
		for j in range(len(a[0,:])):
			output[i,j] = scalar_gradient_x(a*u,i,j) + scalar_gradient_y(a*v,i,j)
	return output	

#####

# set up plot
f, ax = plt.subplots(2,figsize=(9,9))
f.canvas.set_window_title('CLAuDE')
ax[0].contourf(lon_plot, lat_plot, temperature_planet, cmap='seismic')
ax[1].contourf(lon_plot, lat_plot, temperature_atmosp, cmap='seismic')
plt.subplots_adjust(left=0.1, right=0.75)
ax[0].set_title('Ground temperature')
ax[1].set_title('Atmosphere temperature')
# allow for live updating as calculations take place
plt.ion()
plt.show()

# if you want to include advection set this to be True
advection = True

while True:

	# print current time in simulation to command line
	print("t = " + str(round(t/day,2)) + " days", end='\r')
	# print(u.max(),air_density.max(),air_density.min())

	# calculate change in temperature of ground and atmosphere due to radiative imbalance
	for i in range(nlat):
		for j in range(nlon):
			temperature_planet[i,j] += dt*(albedo[i,j]*solar(insolation,lat[i],lon[j],t) + epsilon*sigma*temperature_atmosp[i,j]**4 - sigma*temperature_planet[i,j]**4)/heat_capacity_earth[i,j]
			temperature_atmosp[i,j] += dt*(epsilon*sigma*temperature_planet[i,j]**4 - 2*epsilon*sigma*temperature_atmosp[i,j]**4)/heat_capacity_atmos

	if advection:
		boundary = 10
		# allow for thermal advection in the atmosphere, and heat diffusion in the atmosphere and the ground
		atmosp_addition = dt*divergence_with_scalar(temperature_atmosp)
		temperature_atmosp[boundary:-boundary,:] -= atmosp_addition[boundary:-boundary,:]

		# as density is now variable, allow for mass advection
		density_addition = dt*divergence_with_scalar(air_density)
		air_density[boundary:-boundary,:boundary] -= density_addition[boundary:-boundary,:boundary]

	temperature_atmosp += dt*(thermal_diffusivity_air*laplacian(temperature_atmosp))
	temperature_planet += dt*(thermal_diffusivity_roc*laplacian(temperature_planet))
	
	# update air pressure
	air_pressure = air_density*specific_gas*temperature_atmosp

	u_temp = np.zeros_like(u)
	v_temp = np.zeros_like(v)

	# calculate acceleration of atmosphere using primitive equations on beta-plane
	for i in np.arange(1,nlat-1):
		for j in range(nlon):
			u_temp[i,j] += dt*(  - scalar_gradient_x(air_pressure,i,j)/air_density[i,j] + coriolis[i]*v[i,j]  - u[i,j]*scalar_gradient_x(u,i,j) - v[i,j]*scalar_gradient_y(u,i,j))
			v_temp[i,j] += dt*(  - scalar_gradient_y(air_pressure,i,j)/air_density[i,j] - coriolis[i]*u[i,j]  - u[i,j]*scalar_gradient_x(v,i,j) - v[i,j]*scalar_gradient_y(v,i,j))

	u += u_temp
	v += v_temp

	# update plot
	test = ax[0].contourf(lon_plot, lat_plot, temperature_planet, cmap='seismic')
	ax[1].contourf(lon_plot, lat_plot, temperature_atmosp, cmap='seismic')
	ax[1].quiver(lon_plot[::3, ::3],lat_plot[::3, ::3],u[::3, ::3],v[::3, ::3])
	ax[0].set_title('$\it{Ground} \quad \it{temperature}$')
	ax[1].set_title('$\it{Atmospheric} \quad \it{temperature}$')
	for i in ax: 
		i.set_ylabel('Latitude')
		i.axhline(y=0,color='black',alpha=0.3)
	ax[-1].set_xlabel('Longitude')
	cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
	f.colorbar(test, cax=cbar_ax)
	cbar_ax.set_title('Temperature (K)')
	f.suptitle( 'Time ' + str(round(24*t/day,2)) + ' hours' )
	plt.pause(0.01)
	ax[0].cla()
	ax[1].cla()

	# advance time by one timestep
	t += dt
# toy model for use on stream
# Please give me your Twitch prime sub!

# CLimate Analysis using Digital Estimations (CLAuDE)

import numpy as np 
import matplotlib.pyplot as plt
import time, sys, pickle

### CONTROL
day = 60*60*24					# define length of day (used for calculating Coriolis as well) (s)
dt = 60*9						# <----- TIMESTEP (s)
resolution = 3					# how many degrees between latitude and longitude gridpoints
planet_radius = 6.4E6			# define the planet's radius (m)
insolation = 1370				# TOA radiation from star (W m^-2)

advection = True 				# if you want to include advection set this to be True (currently this breaks the model!)
advection_boundary = 7			# how many gridpoints away from poles to apply advection

save = False
load = True

###########################

# define coordinate arrays
lat = np.arange(-90,91,resolution)
lon = np.arange(0,360,resolution)
nlat = len(lat)
nlon = len(lon)
lon_plot, lat_plot = np.meshgrid(lon, lat)

# initialise arrays for various physical fields
temperature_planet = np.zeros((nlat,nlon)) + 270
temperature_atmosp = np.zeros_like(temperature_planet) + 270
air_pressure = np.zeros_like(temperature_planet)
u = np.zeros_like(temperature_planet)
v = np.zeros_like(temperature_planet)
air_density = np.zeros_like(temperature_planet) + 1.3

albedo = np.zeros_like(temperature_planet) + 0.5
heat_capacity_earth = np.zeros_like(temperature_planet) + 1E7

albedo_variance = 0.001
for i in range(nlat):
	for j in range(nlon):
		albedo[i,j] += np.random.uniform(-albedo_variance,albedo_variance)

# if including an ocean, uncomment the below
# albedo[5:55,9:20] = 0.2
# albedo[23:50,45:70] = 0.2
# albedo[2:30,85:110] = 0.2
# heat_capacity_earth[5:55,9:20] = 1E6
# heat_capacity_earth[23:50,45:70] = 1E6
# heat_capacity_earth[2:30,85:110] = 1E6

# define physical constants
epsilon = 0.75
heat_capacity_atmos = 1E7

specific_gas = 287
thermal_diffusivity_air = 20E-6
thermal_diffusivity_roc = 1.5E-6
sigma = 5.67E-8

# define planet size and various geometric constants
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

###################### FUNCTIONS ######################

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

# power incident on (lat,lon) at time t
def solar(insolation, lat, lon, t):
	sun_longitude = -t % day
	sun_longitude *= 360/day
	value = insolation*np.cos(lat*np.pi/180)*np.cos((lon-sun_longitude)*np.pi/180)
	if value < 0:	return 0
	else:	return value

#################### SHOW TIME ####################

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

# INITIATE TIME
t = 0

if load:
	# load in previous save file
	temperature_atmosp,temperature_planet,u,v,t,air_density,albedo = pickle.load(open("save_file.p","rb"))

while True:

	initial_time = time.time()

	if t < 7*day:
		dt = 60*47
		velocity = False
	else:
		dt = 60*9
		velocity = True

	# print current time in simulation to command line
	print("t = " + str(round(t/day,2)) + " days", end='\r')
	print('U:',u.max(),u.min(),'V: ',v.max(),v.min())

	# calculate change in temperature of ground and atmosphere due to radiative imbalance
	for i in range(nlat):
		for j in range(nlon):
			temperature_planet[i,j] += dt*((1-albedo[i,j])*solar(insolation,lat[i],lon[j],t) + epsilon*sigma*temperature_atmosp[i,j]**4 - sigma*temperature_planet[i,j]**4)/heat_capacity_earth[i,j]
			temperature_atmosp[i,j] += dt*(epsilon*sigma*temperature_planet[i,j]**4 - 2*epsilon*sigma*temperature_atmosp[i,j]**4)/heat_capacity_atmos

	# update air pressure
	air_pressure = air_density*specific_gas*temperature_atmosp

	if velocity:

		# introduce temporary arrays to update velocity in the atmosphere
		u_temp = np.zeros_like(u)
		v_temp = np.zeros_like(v)

		# calculate acceleration of atmosphere using primitive equations on beta-plane
		for i in np.arange(1,nlat-1):
			for j in range(nlon):
				u_temp[i,j] += dt*( -u[i,j]*scalar_gradient_x(u,i,j) - v[i,j]*scalar_gradient_y(u,i,j) + coriolis[i]*v[i,j] - scalar_gradient_x(air_pressure,i,j)/air_density[i,j] )
				v_temp[i,j] += dt*( -u[i,j]*scalar_gradient_x(v,i,j) - v[i,j]*scalar_gradient_y(v,i,j) - coriolis[i]*u[i,j] - scalar_gradient_y(air_pressure,i,j)/air_density[i,j] )

		u += u_temp
		v += v_temp

		u *= 0.99
		v *= 0.99

		if advection:
			# allow for thermal advection in the atmosphere, and heat diffusion in the atmosphere and the ground
			atmosp_addition = dt*(thermal_diffusivity_air*laplacian(temperature_atmosp) + divergence_with_scalar(temperature_atmosp))
			temperature_atmosp[advection_boundary:-advection_boundary,:] -= atmosp_addition[advection_boundary:-advection_boundary,:]
			temperature_atmosp[advection_boundary-1,:] -= 0.5*atmosp_addition[advection_boundary-1,:]
			temperature_atmosp[-advection_boundary,:] -= 0.5*atmosp_addition[-advection_boundary,:]

			# as density is now variable, allow for mass advection
			density_addition = dt*divergence_with_scalar(air_density)
			air_density[advection_boundary:-advection_boundary,:] -= density_addition[advection_boundary:-advection_boundary,:]
			air_density[(advection_boundary-1),:] -= 0.5*density_addition[advection_boundary-1,:]
			air_density[-advection_boundary,:] -= 0.5*density_addition[-advection_boundary,:]

			temperature_planet += dt*(thermal_diffusivity_roc*laplacian(temperature_planet))
	
	# update plot
	test = ax[0].contourf(lon_plot, lat_plot, temperature_planet, cmap='seismic')
	ax[0].set_title('$\it{Ground} \quad \it{temperature}$')

	ax[1].contourf(lon_plot, lat_plot, temperature_atmosp, cmap='seismic')
	ax[1].streamplot(lon_plot,lat_plot,u,v,density=0.75,color='black')
	ax[1].set_title('$\it{Atmospheric} \quad \it{temperature}$')
	for i in ax: 
		i.set_xlim((lon.min(),lon.max()))
		i.set_ylim((lat.min(),lat.max()))
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

	time_taken = float(round(time.time() - initial_time,3))

	print('Time: ',str(time_taken),'s')

	if save:
		pickle.dump((temperature_atmosp,temperature_planet,u,v,t,air_density,albedo), open("save_file.p","wb"))
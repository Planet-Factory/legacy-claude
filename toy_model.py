# toy model for use on stream
# Please give me your Twitch prime sub!

# CLimate Analysis using Digital Estimations (CLAuDE)

import numpy as np 
import matplotlib.pyplot as plt
import time, sys, pickle
import claude_low_level_library as low_level
import claude_top_level_library as top_level

######## CONTROL ########

day = 60*60*24					# define length of day (used for calculating Coriolis as well) (s)
resolution = 3					# how many degrees between latitude and longitude gridpoints
nlevels = 10					# how many vertical layers in the atmosphere
top = 50E3						# top of atmosphere (m)
planet_radius = 6.4E6			# define the planet's radius (m)
insolation = 1370				# TOA radiation from star (W m^-2)
gravity = 9.81 					# define surface gravity for planet (m s^-2)
axial_tilt = -23.5				# tilt of rotational axis w.r.t. solar plane
year = 365*day					# length of year (s)

dt_spinup = 60*137
dt_main = 60*9
spinup_length = 0*day

###

advection = True 				# if you want to include advection set this to be True
advection_boundary = 8			# how many gridpoints away from poles to apply advection

save = False 					# save current state to file?
load = False  					# load initial state from file?

plot = True 					# display plots of output?
level_plots = True 				# display plots of output on vertical levels?
nplots = 5						# how many levels you want to see plots of (evenly distributed through column)

###########################

# define coordinate arrays
lat = np.arange(-90,91,resolution)
lon = np.arange(0,360,resolution)
nlat = len(lat)
nlon = len(lon)
lon_plot, lat_plot = np.meshgrid(lon, lat)
heights = np.arange(0,top,top/nlevels)
heights_plot, lat_z_plot = np.meshgrid(lat,heights)

# initialise arrays for various physical fields
temperature_world = np.zeros((nlat,nlon)) + 290
temperature_atmos = np.zeros((nlat,nlon,nlevels))
air_pressure = np.zeros_like(temperature_atmos)
u = np.zeros_like(temperature_atmos)
v = np.zeros_like(temperature_atmos)
w = np.zeros_like(temperature_atmos)
air_density = np.zeros_like(temperature_atmos)

# #######################

# read temperature and density in from standard atmosphere
f = open("standard_atmosphere.txt", "r")
standard_height = []
standard_temp = []
standard_density = []
for x in f:
	h, t, r = x.split()
	standard_height.append(float(h))
	standard_temp.append(float(t))
	standard_density.append(float(r))
f.close()

density_profile = np.interp(x=heights/1E3,xp=standard_height,fp=standard_density)
temp_profile = np.interp(x=heights/1E3,xp=standard_height,fp=standard_temp)
for k in range(nlevels):
	air_density[:,:,k] = density_profile[k]
	temperature_atmos[:,:,k] = temp_profile[k]

###########################

# weight_above = np.interp(x=heights/1E3,xp=standard_height,fp=standard_density)
# top_index = np.argmax(np.array(standard_height) >= top/1E3)
# if standard_height[top_index] == top/1E3:
# 	weight_above = np.trapz(np.interp(x=standard_height[top_index:],xp=standard_height,fp=standard_density),standard_height[top_index:])*gravity*1E3
# else:
# 	weight_above = np.trapz(np.interp(x=np.insert(standard_height[top_index:], 0, top/1E3),xp=standard_height,fp=standard_density),np.insert(standard_height[top_index:], 0, top/1E3))*gravity*1E3

###########################

albedo = np.zeros_like(temperature_world) + 0.2
heat_capacity_earth = np.zeros_like(temperature_world) + 1E6

albedo_variance = 0.001
for i in range(nlat):
	for j in range(nlon):
		albedo[i,j] += np.random.uniform(-albedo_variance,albedo_variance)

specific_gas = 287
thermal_diffusivity_roc = 1.5E-6
sigma = 5.67E-8

air_pressure = air_density*specific_gas*temperature_atmos

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
dz = np.zeros(nlevels)
for k in range(nlevels-1): dz[k] = heights[k+1] - heights[k]
dz[-1] = dz[-2]

#################### SHOW TIME ####################

if plot:
	# set up plot
	f, ax = plt.subplots(2,figsize=(9,9))
	f.canvas.set_window_title('CLAuDE')
	ax[0].contourf(lon_plot, lat_plot, temperature_world, cmap='seismic')
	ax[1].contourf(lon_plot, lat_plot, temperature_atmos[:,:,0], cmap='seismic')
	plt.subplots_adjust(left=0.1, right=0.75)
	ax[0].set_title('Ground temperature')
	ax[1].set_title('Atmosphere temperature')
	cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])

	# allow for live updating as calculations take place

	if level_plots:

		level_divisions = int(np.floor(nlevels/nplots))
		level_plots_levels = range(nlevels)[::level_divisions]

		g, bx = plt.subplots(nplots,figsize=(9,8),sharex=True)
		g.canvas.set_window_title('CLAuDE atmospheric levels')
		for k, z in zip(range(nplots), level_plots_levels):	
			z += 1
			bx[k].contourf(lon_plot, lat_plot, temperature_atmos[:,:,z], cmap='seismic')
			bx[k].set_title(str(heights[z])+' km')
			bx[k].set_ylabel('Latitude')
		bx[-1].set_xlabel('Longitude')

	plt.ion()
	plt.show()

# INITIATE TIME
t = 0

if load:
	# load in previous save file
	temperature_atmos,temperature_world,u,v,w,t,air_density,albedo = pickle.load(open("save_file.p","rb"))

while True:

	initial_time = time.time()

	if t < spinup_length:
		dt = dt_spinup
		velocity = False
	else:
		dt = dt_main
		velocity = True

	# print current time in simulation to command line
	print("+++ t = " + str(round(t/day,2)) + " days +++", end='\r')
	print('T: ',round(temperature_world.max()-273.15,1),' - ',round(temperature_world.min()-273.15,1),' C')
	print('U: ',round(u.max(),2),' - ',round(u.min(),2),' V: ',round(v.max(),2),' - ',round(v.min(),2),' W: ',round(w.max(),2),' - ',round(w.min(),2))
	# print(profile(air_density))
	# print(profile(air_pressure)/100)

	before_radiation = time.time()
	temperature_world, temperature_atmos = top_level.radiation_calculation(temperature_world, temperature_atmos, air_pressure, air_density, heat_capacity_earth, albedo, insolation, lat, lon, heights, dz, t, dt, day, year, axial_tilt)
	time_taken = float(round(time.time() - before_radiation,3))
	# print('Radiation: ',str(time_taken),'s')

	# update air pressure
	old_pressure = np.copy(air_pressure)
	air_pressure = air_density*specific_gas*temperature_atmos

	if velocity:

		before_velocity = time.time()
		u,v,w = top_level.velocity_calculation(u,v,air_pressure,old_pressure,air_density,coriolis,gravity,dx,dy,dt)
		time_taken = float(round(time.time() - before_velocity,3))
		# print('Velocity: ',str(time_taken),'s')

		before_advection = time.time()
		if advection:
			# allow for thermal advection in the atmosphere, and heat diffusion in the atmosphere and the ground
			# atmosp_addition = dt*(thermal_diffusivity_air*laplacian(temperature_atmos))

			atmosp_addition = dt*top_level.divergence_with_scalar(temperature_atmos,u,v,dx,dy)
			temperature_atmos[advection_boundary:-advection_boundary,:,:] -= atmosp_addition[advection_boundary:-advection_boundary,:,:]
			temperature_atmos[advection_boundary-1,:,:] -= 0.5*atmosp_addition[advection_boundary-1,:,:]
			temperature_atmos[-advection_boundary,:,:] -= 0.5*atmosp_addition[-advection_boundary,:,:]

			# as density is now variable, allow for mass advection
			# density_addition = dt*divergence_with_scalar(air_density)
			# air_density[advection_boundary:-advection_boundary,:,:] -= density_addition[advection_boundary:-advection_boundary,:,:]
			# air_density[(advection_boundary-1),:,:] -= 0.5*density_addition[advection_boundary-1,:,:]
			# air_density[-advection_boundary,:,:] -= 0.5*density_addition[-advection_boundary,:,:]

			# temperature_world += dt*(thermal_diffusivity_roc*laplacian(temperature_world))
			
			time_taken = float(round(time.time() - before_advection,3))
			# print('Advection: ',str(time_taken),'s')
	
	before_plot = time.time()
	if plot:
		# update plot
		test = ax[0].contourf(lon_plot, lat_plot, temperature_world, cmap='seismic')
		ax[0].streamplot(lon_plot, lat_plot, u[:,:,0], v[:,:,0], color='white',density=1)
		ax[0].set_title('$\it{Ground} \quad \it{temperature}$')
		ax[0].set_xlim((lon.min(),lon.max()))
		ax[0].set_ylim((lat.min(),lat.max()))
		ax[0].set_ylabel('Latitude')
		ax[0].axhline(y=0,color='black',alpha=0.3)
		ax[0].set_xlabel('Longitude')

		ax[1].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(temperature_atmos,axis=1)), cmap='seismic')
		ax[1].contour(heights_plot,lat_z_plot, np.transpose(np.mean(u,axis=1)), colors='white',levels=20,linewidths=1,alpha=0.8)
		ax[1].streamplot(heights_plot, lat_z_plot, np.transpose(np.mean(v,axis=1)),np.transpose(np.mean(10*w,axis=1)),color='black',density=0.75)
		ax[1].set_title('$\it{Atmospheric} \quad \it{temperature}$')
		ax[1].set_xlim((-90,90))
		ax[1].set_ylim((0,heights.max()))
		ax[1].set_ylabel('Height (m)')
		ax[1].set_xlabel('Latitude')

		f.colorbar(test, cax=cbar_ax)
		cbar_ax.set_title('Temperature (K)')
		f.suptitle( 'Time ' + str(round(24*t/day,2)) + ' hours' )
		
		if level_plots:
			for k, z in zip(range(nplots), level_plots_levels):	
				z += 1
				bx[k].contourf(lon_plot, lat_plot, temperature_atmos[:,:,z], cmap='seismic')
				bx[k].streamplot(lon_plot, lat_plot, u[:,:,z], v[:,:,z], color='white',density=1)
				bx[k].set_title(str(round(heights[z]/1E3))+' km')
				bx[k].set_ylabel('Latitude')
				bx[k].set_xlim((lon.min(),lon.max()))
				bx[k].set_ylim((lat.min(),lat.max()))				
			bx[-1].set_xlabel('Longitude')

		plt.pause(0.01)
		
		ax[0].cla()
		ax[1].cla()

		if level_plots:
			for k in range(nplots):
				bx[k].cla()
	time_taken = float(round(time.time() - before_plot,3))
	# print('Plotting: ',str(time_taken),'s')

	# advance time by one timestep
	t += dt

	time_taken = float(round(time.time() - initial_time,3))

	print('Time: ',str(time_taken),'s')

	if save:
		pickle.dump((temperature_atmos,temperature_world,u,v,w,t,air_density,albedo), open("save_file.p","wb"))

	# sys.exit()
# CLimate Analysis using Digital Estimations (CLAuDE)

import numpy as np 
import matplotlib.pyplot as plt
import time, sys, pickle
import claude_low_level_library as low_level
import claude_top_level_library as top_level
# from twitch import prime_sub

######## CONTROL ########

day = 60*60*24						# define length of day (used for calculating Coriolis as well) (s)
resolution = 5						# how many degrees between latitude and longitude gridpoints
planet_radius = 6.4E6				# define the planet's radius (m)
insolation = 1370					# TOA radiation from star (W m^-2)
gravity = 9.81 						# define surface gravity for planet (m s^-2)
axial_tilt = 23.5					# tilt of rotational axis w.r.t. solar plane
year = 365*day						# length of year (s)

pressure_levels = np.array([1000,950,900,800,700,600,500,400,350,300,250,200,150,100,75,50,25,10,5,2,1])
pressure_levels *= 100
nlevels = len(pressure_levels)

dt_spinup = 60*52.2					# timestep for initial period where the model only calculates radiative effects
dt_main = 60*12.2					# timestep for the main sequence where the model calculates velocities/advection
spinup_length = 0*day 				# how long the model should only calculate radiative effects

###

smoothing = False					# you probably won't need this, but there is the option to smooth out fields using FFTs (NB this slows down computation considerably and can introduce nonphysical errors)
smoothing_parameter_t = 1.0			
smoothing_parameter_u = 0.9			
smoothing_parameter_v = 0.9			
smoothing_parameter_w = 0.3			
smoothing_parameter_add = 0.3		

###

save = False 						# save current state to file?
load = False  						# load initial state from file?

save_frequency = 25					# write to file after this many timesteps have passed
plot_frequency = 1					# how many timesteps between plots (set this low if you want realtime plots, set this high to improve performance)

###

above = False 						# display top down view of a pole? showing polar plane data and regular gridded data
pole = 's'							# which pole to display - 'n' for north, 's' for south
above_level = 17					# which vertical level to display over the pole

plot = True
diagnostic = False 					# display raw fields for diagnostic purposes
level_plots = False					# display plots of output on vertical levels?
nplots = 3							# how many levels you want to see plots of (evenly distributed through column)
top = 17							# top pressure level to display (i.e. trim off sponge layer)

verbose = False						# print times taken to calculate specific processes each timestep

###

pole_lower_latitude_limit = -65		# how far north polar plane data is calculated from the south pole (do not set this beyond 45!) [mirrored to north pole as well]
pole_higher_latitude_limit = -85	# how far south regular gridded data is calculated (do not set beyond about 80) [also mirrored to north pole]
sponge_layer = 10					# at what pressure should the equations of motion stop being applied (to absorb upwelling atmospheric waves) [hPa]

###

# define coordinate arrays
lat = np.arange(-90,91,resolution)
lon = np.arange(0,360,resolution)
nlat = len(lat)
nlon = len(lon)
lon_plot, lat_plot = np.meshgrid(lon, lat)
heights_plot, lat_z_plot = np.meshgrid(lat,pressure_levels[:top]/100)
temperature_world = np.zeros((nlat,nlon))

##########################

if not load:
	# initialise arrays for various physical fields
	temperature_world += 290
	potential_temperature = np.zeros((nlat,nlon,nlevels))
	u = np.zeros_like(potential_temperature)
	v = np.zeros_like(potential_temperature)
	w = np.zeros_like(potential_temperature)
	atmosp_addition = np.zeros_like(potential_temperature)

	# read temperature and density in from standard atmosphere
	f = open("standard_atmosphere.txt", "r")
	standard_temp = []
	standard_pressure = []
	for x in f:
		h, t, r, p = x.split()
		standard_temp.append(float(t))
		standard_pressure.append(float(p))
	f.close()

	# density_profile = np.interp(x=heights/1E3,xp=standard_height,fp=standard_density)
	temp_profile = np.interp(x=pressure_levels[::-1],xp=standard_pressure[::-1],fp=standard_temp[::-1])[::-1]
	for k in range(nlevels):
		potential_temperature[:,:,k] = temp_profile[k]

	potential_temperature = low_level.t_to_theta(potential_temperature,pressure_levels)
	geopotential = np.zeros_like(potential_temperature)

initial_setup = True
if initial_setup:
	sigma = np.zeros_like(pressure_levels)								# the Exner function
	kappa = 287/1000
	for i in range(len(sigma)):
		sigma[i] = 1E3*(pressure_levels[i]/pressure_levels[0])**kappa	# note the 1E3 here is the heat capacity of dry air at constant pressure (accurate to 4 s.f.)

	heat_capacity_earth = np.zeros_like(temperature_world) + 1E6

	# heat_capacity_earth[15:36,30:60] = 1E7
	# heat_capacity_earth[30:40,80:90] = 1E7

	albedo_variance = 0.001
	# albedo = np.random.uniform(-albedo_variance,albedo_variance, (nlat, nlon)) + 0.2
	albedo = np.zeros((nlat, nlon)) + 0.2

	specific_gas = 287
	thermal_diffusivity_roc = 1.5E-6

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

	sponge_index = np.where(pressure_levels < sponge_layer*100)[0][0]

setup_grids = True
if setup_grids:

	pole_low_index_S = np.where(lat > pole_lower_latitude_limit)[0][0]
	pole_high_index_S = np.where(lat > pole_higher_latitude_limit)[0][0]

	# initialise grid
	polar_grid_resolution = dx[pole_low_index_S]
	size_of_grid = planet_radius*np.cos(lat[pole_low_index_S]*np.pi/180.0)

	# how many times does polar_grid_resolution fit into size_of_grid?
	n_grid_divisions = int(np.ceil(size_of_grid/(2*polar_grid_resolution))) + 1

	### south pole ###
	grid_x_values_S = []
	grid_y_values_S = []
	index = -n_grid_divisions
	while index <= n_grid_divisions:
		grid_x_values_S.append(index*polar_grid_resolution)
		grid_y_values_S.append(index*polar_grid_resolution)
		index += 1

	# print(grid_x_values_S)
	# print(grid_y_values_S)

	grid_xx_S,grid_yy_S = np.meshgrid(grid_x_values_S,grid_y_values_S)

	grid_side_length = len(grid_x_values_S)

	grid_lat_coords_S = (-np.arccos(((grid_xx_S**2 + grid_yy_S**2)**0.5)/planet_radius)*180.0/np.pi).flatten()
	grid_lon_coords_S = (180.0 - np.arctan2(grid_yy_S,grid_xx_S)*180.0/np.pi).flatten()

	polar_x_coords_S = []
	polar_y_coords_S = []
	for i in range(pole_low_index_S):
		for j in range(nlon):
			polar_x_coords_S.append( planet_radius*np.cos(lat[i]*np.pi/180.0)*np.sin(lon[j]*np.pi/180.0) )
			polar_y_coords_S.append(-planet_radius*np.cos(lat[i]*np.pi/180.0)*np.cos(lon[j]*np.pi/180.0) )

	### north pole ###
	
	pole_low_index_N  	= 	np.where(lat < -pole_lower_latitude_limit)[0][-1]
	pole_high_index_N 	= 	np.where(lat < -pole_higher_latitude_limit)[0][-1]

	grid_x_values_N = []
	grid_y_values_N = []
	index = -n_grid_divisions
	while index <= n_grid_divisions:
		grid_x_values_N.append(index*polar_grid_resolution)
		grid_y_values_N.append(index*polar_grid_resolution)
		index += 1

	grid_xx_N,grid_yy_N = 	np.meshgrid(grid_x_values_N,grid_y_values_N)

	grid_lat_coords_N 	= 	(np.arccos((grid_xx_N**2 + grid_yy_N**2)**0.5/planet_radius)*180.0/np.pi).flatten()
	grid_lon_coords_N 	= 	(180.0 - np.arctan2(grid_yy_N,grid_xx_N)*180.0/np.pi).flatten()

	polar_x_coords_N 	= 	[]
	polar_y_coords_N 	= 	[]

	for i in np.arange(pole_low_index_N,nlat):
		for j in range(nlon):
			polar_x_coords_N.append( planet_radius*np.cos(lat[i]*np.pi/180.0)*np.sin(lon[j]*np.pi/180.0) )
			polar_y_coords_N.append(-planet_radius*np.cos(lat[i]*np.pi/180.0)*np.cos(lon[j]*np.pi/180.0) )

	indices = (pole_low_index_N,pole_high_index_N,pole_low_index_S,pole_high_index_S)
	grids   = (grid_xx_N.shape[0],grid_xx_S.shape[0])

	# create Coriolis data on north and south planes
	data = np.zeros((nlat-pole_low_index_N,nlon))
	for i in np.arange(pole_low_index_N,nlat):
		data[i-pole_low_index_N,:] = coriolis[i]
	coriolis_plane_N = low_level.beam_me_up_2D(lat[(pole_low_index_N):],lon,data,grids[0],grid_lat_coords_N,grid_lon_coords_N)
	data = np.zeros((pole_low_index_S,nlon))
	for i in range(pole_low_index_S):
		data[i,:] = coriolis[i]
	coriolis_plane_S = low_level.beam_me_up_2D(lat[:(pole_low_index_S)],lon,data,grids[1],grid_lat_coords_S,grid_lon_coords_S)

	x_dot_N = np.zeros((grids[0],grids[0],nlevels))
	y_dot_N = np.zeros((grids[0],grids[0],nlevels))
	x_dot_S = np.zeros((grids[1],grids[1],nlevels))
	y_dot_S = np.zeros((grids[1],grids[1],nlevels))

	grid_x_values_N = np.array(grid_x_values_N)
	grid_y_values_N = np.array(grid_y_values_N)
	grid_x_values_S = np.array(grid_x_values_S)
	grid_y_values_S = np.array(grid_y_values_S)

	coords  = grid_lat_coords_N,grid_lon_coords_N,grid_x_values_N,grid_y_values_N,polar_x_coords_N,polar_y_coords_N,grid_lat_coords_S,grid_lon_coords_S,grid_x_values_S,grid_y_values_S,polar_x_coords_S,polar_y_coords_S

#######################################################################################################################################################################################################################

# INITIATE TIME
t = 0.0

if load:
	# load in previous save file
	potential_temperature,temperature_world,u,v,w,x_dot_N,y_dot_N,x_dot_S,y_dot_S,t,albedo,tracer = pickle.load(open("save_file.p","rb"))

sample_level = 15
tracer = np.zeros_like(potential_temperature)

last_plot = t - 0.1
last_save = t - 0.1

if plot:
	if not diagnostic:
		# set up plot
		f, ax = plt.subplots(2,figsize=(9,9))
		f.canvas.set_window_title('CLAuDE')
		ax[0].contourf(lon_plot, lat_plot, temperature_world, cmap='seismic')
		ax[0].streamplot(lon_plot, lat_plot, u[:,:,0], v[:,:,0], color='white',density=1)
		test = ax[1].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(low_level.theta_to_t(potential_temperature,pressure_levels),axis=1))[:top,:], cmap='seismic',levels=15)
		ax[1].contour(heights_plot,lat_z_plot, np.transpose(np.mean(u,axis=1))[:top,:], colors='white',levels=20,linewidths=1,alpha=0.8)
		ax[1].quiver(heights_plot, lat_z_plot, np.transpose(np.mean(v,axis=1))[:top,:],np.transpose(np.mean(100*w,axis=1))[:top,:],color='black')
		plt.subplots_adjust(left=0.1, right=0.75)
		ax[0].set_title('Surface temperature')
		ax[0].set_xlim(lon.min(),lon.max())
		ax[1].set_title('Atmosphere temperature')
		ax[1].set_xlim(lat.min(),lat.max())
		ax[1].set_ylim((pressure_levels.max()/100,pressure_levels[:top].min()/100))
		ax[1].set_yscale('log')		
		ax[1].set_ylabel('Pressure (hPa)')
		ax[1].set_xlabel('Latitude')
		cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
		f.colorbar(test, cax=cbar_ax)
		cbar_ax.set_title('Temperature (K)')
		f.suptitle( 'Time ' + str(round(t/day,2)) + ' days' )

	else:
		# set up plot
		f, ax = plt.subplots(2,2,figsize=(9,9))
		f.canvas.set_window_title('CLAuDE')
		ax[0,0].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(u,axis=1))[:top,:], cmap='seismic')
		ax[0,0].set_title('u')
		ax[0,1].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(v,axis=1))[:top,:], cmap='seismic')
		ax[0,1].set_title('v')
		ax[1,0].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(w,axis=1))[:top,:], cmap='seismic')
		ax[1,0].set_title('w')
		ax[1,1].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(atmosp_addition,axis=1))[:top,:], cmap='seismic')
		ax[1,1].set_title('atmosp_addition')
		for axis in ax.ravel():
			axis.set_ylim((pressure_levels.max()/100,pressure_levels[:top].min()/100))
			axis.set_yscale('log')
		f.suptitle( 'Time ' + str(round(t/day,2)) + ' days' )

	if level_plots:

		level_divisions = int(np.floor(nlevels/nplots))
		level_plots_levels = range(nlevels)[::level_divisions][::-1]

		g, bx = plt.subplots(nplots,figsize=(9,8),sharex=True)
		g.canvas.set_window_title('CLAuDE pressure levels')
		for k, z in zip(range(nplots), level_plots_levels):	
			z += 1
			bx[k].contourf(lon_plot, lat_plot, potential_temperature[:,:,z], cmap='seismic')
			bx[k].set_title(str(pressure_levels[z]/100)+' hPa')
			bx[k].set_ylabel('Latitude')
		bx[-1].set_xlabel('Longitude')
	
	plt.ion()
	plt.show()
	plt.pause(2)

	if not diagnostic:
		ax[0].cla()
		ax[1].cla()
		if level_plots:
			for k in range(nplots):
				bx[k].cla()		
	else:
		ax[0,0].cla()
		ax[0,1].cla()	
		ax[1,0].cla()
		ax[1,1].cla()

if above:
	g,gx = plt.subplots(1,3, figsize=(15,5))
	plt.ion()
	plt.show()

def plotting_routine():
	
	quiver_padding = int(12/resolution)

	if plot:
		if verbose:	before_plot = time.time()
		# update plot
		if not diagnostic:
			
			# field = temperature_world
			# field = np.copy(w)[:,:,sample_level]
			field = np.copy(atmosp_addition)[:,:,sample_level]
			test = ax[0].contourf(lon_plot, lat_plot, field, cmap='seismic',levels=15)
			ax[0].contour(lon_plot, lat_plot, tracer[:,:,sample_level], alpha=0.5, antialiased=True, levels=np.arange(0.01,1.01,0.01))
			if velocity:	ax[0].quiver(lon_plot[::quiver_padding,::quiver_padding], lat_plot[::quiver_padding,::quiver_padding], u[::quiver_padding,::quiver_padding,sample_level], v[::quiver_padding,::quiver_padding,sample_level], color='white')
			ax[0].set_xlim((lon.min(),lon.max()))
			ax[0].set_ylim((lat.min(),lat.max()))
			ax[0].set_ylabel('Latitude')
			ax[0].axhline(y=0,color='black',alpha=0.3)
			ax[0].set_xlabel('Longitude')

			###

			test = ax[1].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(low_level.theta_to_t(potential_temperature,pressure_levels),axis=1))[:top,:], cmap='seismic',levels=15)
			# test = ax[1].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(atmosp_addition,axis=1))[:top,:], cmap='seismic',levels=15)
			# test = ax[1].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(potential_temperature,axis=1)), cmap='seismic',levels=15)
			ax[1].contour(heights_plot, lat_z_plot, np.transpose(np.mean(tracer,axis=1))[:top,:], alpha=0.5, antialiased=True, levels=np.arange(0.001,1.01,0.01))

			if velocity:
				ax[1].contour(heights_plot,lat_z_plot, np.transpose(np.mean(u,axis=1))[:top,:], colors='white',levels=20,linewidths=1,alpha=0.8)
				ax[1].quiver(heights_plot, lat_z_plot, np.transpose(np.mean(v,axis=1))[:top,:],np.transpose(np.mean(-10*w,axis=1))[:top,:],color='black')
			ax[1].set_title('$\it{Atmospheric} \quad \it{temperature}$')
			ax[1].set_xlim((-90,90))
			ax[1].set_ylim((pressure_levels.max()/100,pressure_levels[:top].min()/100))
			ax[1].set_ylabel('Pressure (hPa)')
			ax[1].set_xlabel('Latitude')
			ax[1].set_yscale('log')
			f.colorbar(test, cax=cbar_ax)
			cbar_ax.set_title('Temperature (K)')
			f.suptitle( 'Time ' + str(round(t/day,2)) + ' days' )
				
		else:
			ax[0,0].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(u,axis=1))[:top,:], cmap='seismic')
			ax[0,0].set_title('u')
			ax[0,1].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(v,axis=1))[:top,:], cmap='seismic')
			ax[0,1].set_title('v')
			ax[1,0].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(w,axis=1))[:top,:], cmap='seismic')
			ax[1,0].set_title('w')
			ax[1,1].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(atmosp_addition,axis=1))[:top,:], cmap='seismic')
			ax[1,1].set_title('atmosp_addition')
			for axis in ax.ravel():
				axis.set_ylim((pressure_levels.max()/100,pressure_levels[:top].min()/100))
				axis.set_yscale('log')
			f.suptitle( 'Time ' + str(round(t/day,2)) + ' days' )

		if level_plots:
			for k, z in zip(range(nplots), level_plots_levels):	
				z += 1
				bx[k].contourf(lon_plot, lat_plot, potential_temperature[:,:,z], cmap='seismic',levels=15)
				bx[k].quiver(lon_plot[::quiver_padding,::quiver_padding], lat_plot[::quiver_padding,::quiver_padding], u[::quiver_padding,::quiver_padding,z], v[::quiver_padding,::quiver_padding,z], color='white')
				bx[k].set_title(str(round(pressure_levels[z]/100))+' hPa')
				bx[k].set_ylabel('Latitude')
				bx[k].set_xlim((lon.min(),lon.max()))
				bx[k].set_ylim((lat.min(),lat.max()))				
			bx[-1].set_xlabel('Longitude')

	if above and velocity:
		gx[0].set_title('Original data')
		gx[1].set_title('Polar plane')
		gx[2].set_title('Reprojected data')
		g.suptitle( 'Time ' + str(round(t/day,2)) + ' days' )

		if pole == 's':
			gx[0].set_title('temperature')
			gx[0].contourf(lon,lat[:pole_low_index_S],potential_temperature[:pole_low_index_S,:,above_level])
			
			gx[1].set_title('polar_plane_advect')
			polar_temps = low_level.beam_me_up(lat[:pole_low_index_S],lon,potential_temperature[:pole_low_index_S,:,:],grids[1],grid_lat_coords_S,grid_lon_coords_S)
			output = low_level.beam_me_up_2D(lat[:pole_low_index_S],lon,w[:pole_low_index_S,:,above_level],grids[1],grid_lat_coords_S,grid_lon_coords_S)

			gx[1].contourf(grid_x_values_S/1E3,grid_y_values_S/1E3,output)
			gx[1].contour(grid_x_values_S/1E3,grid_y_values_S/1E3,polar_temps[:,:,above_level],colors='white',levels=20,linewidths=1,alpha=0.8)
			gx[1].quiver(grid_x_values_S/1E3,grid_y_values_S/1E3,x_dot_S[:,:,above_level],y_dot_S[:,:,above_level])
			
			gx[1].add_patch(plt.Circle((0,0),planet_radius*np.cos(lat[pole_low_index_S]*np.pi/180.0)/1E3,color='r',fill=False))
			gx[1].add_patch(plt.Circle((0,0),planet_radius*np.cos(lat[pole_high_index_S]*np.pi/180.0)/1E3,color='r',fill=False))

			gx[2].set_title('south_addition_smoothed')
			gx[2].contourf(lon,lat[:pole_low_index_S],atmosp_addition[:pole_low_index_S,:,above_level])
			gx[2].quiver(lon[::5],lat[:pole_low_index_S],u[:pole_low_index_S,::5,above_level],v[:pole_low_index_S,::5,above_level])
		else:
			gx[0].set_title('temperature')
			gx[0].contourf(lon,lat[pole_low_index_N:],potential_temperature[pole_low_index_N:,:,above_level])
			
			gx[1].set_title('polar_plane_advect')
			polar_temps = low_level.beam_me_up(lat[pole_low_index_N:],lon,np.flip(potential_temperature[pole_low_index_N:,:,:],axis=1),grids[0],grid_lat_coords_N,grid_lon_coords_N)
			output = low_level.beam_me_up_2D(lat[pole_low_index_N:],lon,atmosp_addition[pole_low_index_N:,:,above_level],grids[0],grid_lat_coords_N,grid_lon_coords_N)
			output = low_level.beam_me_up_2D(lat[pole_low_index_N:],lon,w[pole_low_index_N:,:,above_level],grids[0],grid_lat_coords_N,grid_lon_coords_N)

			gx[1].contourf(grid_x_values_N/1E3,grid_y_values_N/1E3,output)
			gx[1].contour(grid_x_values_N/1E3,grid_y_values_N/1E3,polar_temps[:,:,above_level],colors='white',levels=20,linewidths=1,alpha=0.8)
			gx[1].quiver(grid_x_values_N/1E3,grid_y_values_N/1E3,x_dot_N[:,:,above_level],y_dot_N[:,:,above_level])
			
			gx[1].add_patch(plt.Circle((0,0),planet_radius*np.cos(lat[pole_low_index_N]*np.pi/180.0)/1E3,color='r',fill=False))
			gx[1].add_patch(plt.Circle((0,0),planet_radius*np.cos(lat[pole_high_index_N]*np.pi/180.0)/1E3,color='r',fill=False))
	
			gx[2].set_title('south_addition_smoothed')
			# gx[2].contourf(lon,lat[pole_low_index_N:],north_addition_smoothed[:,:,above_level])
			gx[2].contourf(lon,lat[pole_low_index_N:],atmosp_addition[pole_low_index_N:,:,above_level])
			gx[2].quiver(lon[::5],lat[pole_low_index_N:],u[pole_low_index_N:,::5,above_level],v[pole_low_index_N:,::5,above_level])
		
	# clear plots
	if plot or above:	plt.pause(0.001)
	if plot:
		if not diagnostic:
			ax[0].cla()
			ax[1].cla()
			cbar_ax.cla()
					
		else:
			ax[0,0].cla()
			ax[0,1].cla()
			ax[1,0].cla()
			ax[1,1].cla()
		if level_plots:
			for k in range(nplots):
				bx[k].cla()	
		if verbose:		
			time_taken = float(round(time.time() - before_plot,3))
			print('Plotting: ',str(time_taken),'s')	
	if above:
		gx[0].cla()
		gx[1].cla()
		gx[2].cla()

while True:

	initial_time = time.time()

	if t < spinup_length:
		dt = dt_spinup
		velocity = False
	else:
		dt = dt_main
		velocity = True

	# print current time in simulation to command line
	print("+++ t = " + str(round(t/day,2)) + " days +++")
	print('T: ',round(temperature_world.max()-273.15,1),' - ',round(temperature_world.min()-273.15,1),' C')
	print('U: ',round(u[:,:,:sponge_index-1].max(),2),' - ',round(u[:,:,:sponge_index-1].min(),2),' V: ',round(v[:,:,:sponge_index-1].max(),2),' - ',round(v[:,:,:sponge_index-1].min(),2),' W: ',round(w[:,:,:sponge_index-1].max(),2),' - ',round(w[:,:,:sponge_index-1].min(),4))

	# tracer[40,50,sample_level] = 1
	# tracer[20,50,sample_level] = 1
	tracer[18,20,sample_level] = 1

	if verbose: before_radiation = time.time()
	temperature_world, potential_temperature = top_level.radiation_calculation(temperature_world, potential_temperature, pressure_levels, heat_capacity_earth, albedo, insolation, lat, lon, t, dt, day, year, axial_tilt)
	if smoothing: potential_temperature = top_level.smoothing_3D(potential_temperature,smoothing_parameter_t)
	if verbose:
		time_taken = float(round(time.time() - before_radiation,3))
		print('Radiation: ',str(time_taken),'s')

	diffusion = top_level.laplacian_2d(temperature_world,dx,dy)
	diffusion[0,:] = np.mean(diffusion[1,:],axis=0)
	diffusion[-1,:] = np.mean(diffusion[-2,:],axis=0)
	temperature_world -= dt*1E-5*diffusion

	# update geopotential field
	geopotential = np.zeros_like(potential_temperature)
	for k in np.arange(1,nlevels):	geopotential[:,:,k] = geopotential[:,:,k-1] - potential_temperature[:,:,k]*(sigma[k]-sigma[k-1])
	geopotential[:,:,0] = geopotential[:,:,1] - 287*low_level.theta_to_t(potential_temperature,pressure_levels)[:,:,0]/pressure_levels[0]

	if velocity:

		if verbose:	before_velocity = time.time()
		
		u_add,v_add = top_level.velocity_calculation(u,v,w,pressure_levels,geopotential,potential_temperature,coriolis,gravity,dx,dy,dt,sponge_index,resolution)

		if verbose:	
			time_taken = float(round(time.time() - before_velocity,3))
			print('Velocity: ',str(time_taken),'s')

		if verbose:	before_projection = time.time()
		
		grid_velocities = (x_dot_N,y_dot_N,x_dot_S,y_dot_S)
	
		u_add,v_add,x_dot_N,y_dot_N,x_dot_S,y_dot_S = top_level.polar_planes(u,v,u_add,v_add,potential_temperature,geopotential,grid_velocities,indices,grids,coords,coriolis_plane_N,coriolis_plane_S,grid_side_length,pressure_levels,lat,lon,dt,polar_grid_resolution,gravity,sponge_index)
		
		u += u_add
		v += v_add

		if smoothing: u = top_level.smoothing_3D(u,smoothing_parameter_u)
		if smoothing: v = top_level.smoothing_3D(v,smoothing_parameter_v)

		# x_dot_N,y_dot_N,x_dot_S,y_dot_S = top_level.update_plane_velocities(lat,lon,pole_low_index_N,pole_low_index_S,np.flip(u[pole_low_index_N:,:,:],axis=1),np.flip(v[pole_low_index_N:,:,:],axis=1),grids,grid_lat_coords_N,grid_lon_coords_N,u[:pole_low_index_S,:,:],v[:pole_low_index_S,:,:],grid_lat_coords_S,grid_lon_coords_S)
		grid_velocities = (x_dot_N,y_dot_N,x_dot_S,y_dot_S)
		
		if verbose:	
			time_taken = float(round(time.time() - before_projection,3))
			print('Projection: ',str(time_taken),'s')

		### allow for thermal advection in the atmosphere
		if verbose:	before_advection = time.time()

		if verbose: before_w = time.time()
		# using updated u,v fields calculated w
		# https://www.sjsu.edu/faculty/watkins/omega.htm
		w = top_level.w_calculation(u,v,w,pressure_levels,geopotential,potential_temperature,coriolis,gravity,dx,dy,dt,indices,coords,grids,grid_velocities,polar_grid_resolution,lat,lon)
		if smoothing: w = top_level.smoothing_3D(w,smoothing_parameter_w,0.25)

		# w[:,:,18:] *= 0

		# plt.semilogy(w[30,25,:sponge_index],pressure_levels[:sponge_index])
		# plt.gca().invert_yaxis()
		# plt.show()

		if verbose:	
			time_taken = float(round(time.time() - before_w,3))
			print('Calculate w: ',str(time_taken),'s')

		#################################
		
		atmosp_addition = top_level.divergence_with_scalar(potential_temperature,u,v,w,dx,dy,lat,lon,pressure_levels,polar_grid_resolution,indices,coords,grids,grid_velocities)

		if smoothing: atmosp_addition = top_level.smoothing_3D(atmosp_addition,smoothing_parameter_add)

		atmosp_addition[:,:,sponge_index-1] *= 0.5
		atmosp_addition[:,:,sponge_index:] *= 0

		potential_temperature -= dt*atmosp_addition

		###################################################################

		tracer_addition = top_level.divergence_with_scalar(tracer,u,v,w,dx,dy,lat,lon,pressure_levels,polar_grid_resolution,indices,coords,grids,grid_velocities)
		tracer -= dt*tracer_addition

		diffusion = top_level.laplacian_3d(potential_temperature,dx,dy,pressure_levels)
		diffusion[0,:,:] = np.mean(diffusion[1,:,:],axis=0)
		diffusion[-1,:,:] = np.mean(diffusion[-2,:,:],axis=0)
		potential_temperature -= dt*1E-4*diffusion

		courant = w*dt
		for k in range(nlevels-1):
			courant[:,:,k] /= (pressure_levels[k+1] - pressure_levels[k])
		courant_max_coords = np.where(abs(courant) == abs(courant).max())
		print('Courant max: ',round(abs(courant).max(),3),' at lat = ',str(lat[courant_max_coords[0]][0]),', lon = ',str(lon[courant_max_coords[1]][0]),', pressure = ',str(pressure_levels[courant_max_coords[2]][0]/100.),'hPa')

		###################################################################

		if verbose:	
			time_taken = float(round(time.time() - before_advection,3))
			print('Advection: ',str(time_taken),'s')

	if t-last_plot >= plot_frequency*dt:
		plotting_routine()
		last_plot = t

	if save:
		if t-last_save >= save_frequency*dt:
			pickle.dump((potential_temperature,temperature_world,u,v,w,x_dot_N,y_dot_N,x_dot_S,y_dot_S,t,albedo,tracer), open("save_file.p","wb"))
			last_save = t

	if np.isnan(u.max()):
		sys.exit()

	# advance time by one timestep
	t += dt

	time_taken = float(round(time.time() - initial_time,3))

	print('Time: ',str(time_taken),'s')
	print('777777777777777777')
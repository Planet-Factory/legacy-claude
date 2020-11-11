# toy model for use on stream
# Please give me your Twitch prime sub!

# CLimate Analysis using Digital Estimations (CLAuDE)

import numpy as np 
import matplotlib.pyplot as plt
import time, sys, pickle
import claude_low_level_library as low_level
import claude_top_level_library as top_level
from scipy.interpolate import interp2d, RectBivariateSpline
# from twitch import prime_sub

######## CONTROL ########

day = 60*60*24					# define length of day (used for calculating Coriolis as well) (s)
resolution = 5					# how many degrees between latitude and longitude gridpoints
planet_radius = 6.4E6			# define the planet's radius (m)
insolation = 1370				# TOA radiation from star (W m^-2)
gravity = 9.81 					# define surface gravity for planet (m s^-2)
axial_tilt = -23.5/2			# tilt of rotational axis w.r.t. solar plane
year = 365*day					# length of year (s)

pressure_levels = np.array([1000,950,900,800,700,600,500,400,350,300,250,200,150,100,75,50,25,10,5,2,1])
pressure_levels *= 100
nlevels = len(pressure_levels)

top = nlevels

dt_spinup = 60*137
dt_main = 60*3.5
spinup_length = 1*day

###

advection = True 				# if you want to include advection set this to be True
advection_boundary = 5			# how many gridpoints away from poles to apply advection
smoothing_parameter_t = 0.6
smoothing_parameter_u = 0.6
smoothing_parameter_v = 0.6
smoothing_parameter_w = 0.4

save = False 					# save current state to file?
load = False  					# load initial state from file?

plot = False 					# display plots of output?
diagnostic = False 				# display raw fields for diagnostic purposes
level_plots = False 			# display plots of output on vertical levels?
nplots = 3						# how many levels you want to see plots of (evenly distributed through column)

###########################

# define coordinate arrays
lat = np.arange(-90,91,resolution)
lon = np.arange(0,360,resolution)
nlat = len(lat)
nlon = len(lon)
lon_plot, lat_plot = np.meshgrid(lon, lat)
heights_plot, lat_z_plot = np.meshgrid(lat,pressure_levels/100)

# initialise arrays for various physical fields
temperature_world = np.zeros((nlat,nlon)) + 290
potential_temperature = np.zeros((nlat,nlon,nlevels))
u = np.zeros_like(potential_temperature)
v = np.zeros_like(potential_temperature)
w = np.zeros_like(potential_temperature)
atmosp_addition = np.zeros_like(potential_temperature)

##########################

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

sigma = np.zeros_like(pressure_levels)
kappa = 287/1000
for i in range(len(sigma)):
	sigma[i] = 1E3*(pressure_levels[i]/pressure_levels[0])**kappa

###########################

albedo = np.zeros_like(temperature_world) + 0.2
heat_capacity_earth = np.zeros_like(temperature_world) + 1E6

albedo_variance = 0.001
for i in range(nlat):
	for j in range(nlon):
		albedo[i,j] += np.random.uniform(-albedo_variance,albedo_variance)

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

#################### SHOW TIME ####################

# initialise grid
pole_lower_latitude_limit = 5
polar_grid_resolution = dx[-pole_lower_latitude_limit]/5
size_of_grid = planet_radius*np.cos(lat[-pole_lower_latitude_limit]*np.pi/180)
grid_x_values = np.arange(-size_of_grid,size_of_grid,polar_grid_resolution)
grid_y_values = np.arange(-size_of_grid,size_of_grid,polar_grid_resolution)
grid_xx,grid_yy = np.meshgrid(grid_x_values,grid_y_values)

grid_lat_coords = []
grid_lon_coords = []
for i in range(grid_xx.shape[0]):
	for j in range(grid_xx.shape[1]):
		lat_point = -np.arccos((grid_xx[i,j]**2 + grid_yy[i,j]**2)**0.5/planet_radius)*180/np.pi
		lon_point = np.arctan2(grid_yy[i,j],grid_xx[i,j])*180/np.pi
		grid_lat_coords.append(lat_point)
		grid_lon_coords.append(lon_point)

polar_x_coords = []
polar_y_coords = []
for i in range(pole_lower_latitude_limit):
	for j in range(nlon):
		polar_x_coords.append( -planet_radius*np.cos(lat[i]*np.pi/180)*np.sin(lon[j]*np.pi/180) )
		polar_y_coords.append( -planet_radius*np.cos(lat[i]*np.pi/180)*np.cos(lon[j]*np.pi/180) )


def beam_me_up(lat,lon,data,pole_lower_latitude_limit,grid_size,grid_lat_coords,grid_lon_coords):
	'''Projects data on lat-lon grid to x-y polar grid'''
	f = RectBivariateSpline(lat[:pole_lower_latitude_limit], lon, data)
	polar_plane = f(grid_lat_coords,grid_lon_coords,grid=False).reshape((grid_size,grid_size))
	return polar_plane

def beam_me_down(lat,lon,data,pole_lower_latitude_limit,polar_x_coords,polar_y_coords):
	'''projects data from x-y polar grid onto lat-lon grid'''
	f = RectBivariateSpline(x=grid_x_values, y=grid_y_values, z=data)
	resample = f(polar_x_coords,polar_y_coords,grid=False).reshape((pole_lower_latitude_limit,len(lon)))
	return resample


#######################################################################################################################################################################################################################

# INITIATE TIME
t = 0

if load:
	# load in previous save file
	potential_temperature,temperature_world,u,v,w,t,albedo = pickle.load(open("save_file.p","rb"))

if plot:
	if not diagnostic:
		# set up plot
		f, ax = plt.subplots(2,figsize=(9,9))
		f.canvas.set_window_title('CLAuDE')
		test = ax[0].contourf(lon_plot, lat_plot, temperature_world, cmap='seismic')
		ax[0].streamplot(lon_plot, lat_plot, u[:,:,0], v[:,:,0], color='white',density=1)
		ax[1].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(low_level.theta_to_t(potential_temperature,pressure_levels),axis=1))[:top,:], cmap='seismic',levels=15)
		ax[1].contour(heights_plot,lat_z_plot, np.transpose(np.mean(u,axis=1))[:top,:], colors='white',levels=20,linewidths=1,alpha=0.8)
		ax[1].quiver(heights_plot, lat_z_plot, np.transpose(np.mean(v,axis=1))[:top,:],np.transpose(np.mean(10*w,axis=1))[:top,:],color='black')
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
	print('U: ',round(u.max(),2),' - ',round(u.min(),2),' V: ',round(v.max(),2),' - ',round(v.min(),2),' W: ',round(w.max(),2),' - ',round(w.min(),2))

	before_radiation = time.time()
	temperature_world, potential_temperature = top_level.radiation_calculation(temperature_world, potential_temperature, pressure_levels, heat_capacity_earth, albedo, insolation, lat, lon, t, dt, day, year, axial_tilt)
	potential_temperature = top_level.smoothing_3D(potential_temperature,smoothing_parameter_t)
	time_taken = float(round(time.time() - before_radiation,3))
	print('Radiation: ',str(time_taken),'s')

	# update geopotential field
	for k in np.arange(1,nlevels):
		geopotential[:,:,k] = geopotential[:,:,k-1] - potential_temperature[:,:,k]*(sigma[k]-sigma[k-1])
	geopotential = top_level.smoothing_3D(geopotential,smoothing_parameter_t)

	if velocity:

		before_velocity = time.time()
		u,v,w = top_level.velocity_calculation(u,v,w,pressure_levels,geopotential,potential_temperature,coriolis,gravity,dx,dy,dt)
		u = top_level.smoothing_3D(u,smoothing_parameter_u)
		v = top_level.smoothing_3D(v,smoothing_parameter_v)
		w = top_level.smoothing_3D(w,smoothing_parameter_w,0.1)
		
		# boundary shite
		u[(advection_boundary,-advection_boundary-1),:,:] *= 0.5
		v[(advection_boundary,-advection_boundary-1),:,:] *= 0.5
		w[(advection_boundary,-advection_boundary-1),:,:] *= 0.5

		u[:advection_boundary,:,:] = 0
		v[:advection_boundary,:,:] = 0
		w[:advection_boundary,:,:] = 0
		u[-advection_boundary:,:,:] = 0
		v[-advection_boundary:,:,:] = 0
		w[-advection_boundary:,:,:] = 0

		w[:,:,-1] *= 0
		w[:,:,-2] *= 0.1
		w[:,:,-3] *= 0.5

		time_taken = float(round(time.time() - before_velocity,3))
		print('Velocity: ',str(time_taken),'s')

		if advection:
			before_advection = time.time()

			# allow for thermal advection in the atmosphere, and heat diffusion in the atmosphere and the ground
			atmosp_addition = dt*top_level.divergence_with_scalar(potential_temperature,u,v,w,dx,dy,pressure_levels)
			atmosp_addition[(-advection_boundary,advection_boundary-1),:,:] *= 0.5
			atmosp_addition[:advection_boundary,:,:] *= 0
			atmosp_addition[-advection_boundary:,:,:] *= 0
			atmosp_addition[:,:,-1] *= 0
			atmosp_addition[:,:,-2] *= 0.5
			potential_temperature -= atmosp_addition

			# temperature_world -= dt*(thermal_diffusivity_roc*top_level.laplacian_2D(temperature_world,dx,dy))

			time_taken = float(round(time.time() - before_advection,3))
			print('Advection: ',str(time_taken),'s')
	
	# before_plot = time.time()
	if plot:

		# update plot
		if not diagnostic:
			ax[0].contourf(lon_plot, lat_plot, temperature_world, cmap='seismic',levels=15)
			if velocity:	
				ax[0].streamplot(lon_plot, lat_plot, u[:,:,0], v[:,:,0], color='white',density=0.75)
			ax[0].set_title('$\it{Ground} \quad \it{temperature}$')
			ax[0].set_xlim((lon.min(),lon.max()))
			ax[0].set_ylim((lat.min(),lat.max()))
			ax[0].set_ylabel('Latitude')
			ax[0].axhline(y=0,color='black',alpha=0.3)
			ax[0].set_xlabel('Longitude')

			test = ax[1].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(low_level.theta_to_t(potential_temperature,pressure_levels),axis=1))[:top,:], cmap='seismic',levels=15)
			# test = ax[1].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(potential_temperature,axis=1)), cmap='seismic',levels=15)
			# test = ax[1].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(v,axis=1))[:top,:], cmap='seismic',levels=15)
			if velocity:
				ax[1].contour(heights_plot,lat_z_plot, np.transpose(np.mean(u,axis=1))[:top,:], colors='white',levels=20,linewidths=1,alpha=0.8)
				ax[1].quiver(heights_plot, lat_z_plot, np.transpose(np.mean(v,axis=1))[:top,:],np.transpose(np.mean(10*w,axis=1))[:top,:],color='black')
			ax[1].set_title('$\it{Atmospheric} \quad \it{temperature}$')
			ax[1].set_xlim((-90,90))
			ax[1].set_ylim((pressure_levels.max()/100,pressure_levels[:top].min()/100))
			ax[1].set_ylabel('Pressure (hPa)')
			ax[1].set_xlabel('Latitude')
			ax[1].set_yscale('log')
			f.colorbar(test, cax=cbar_ax)
			cbar_ax.set_title('Temperature (K)')
			f.suptitle( 'Time ' + str(round(t/day,2)) + ' days' )
		
			if level_plots:
				quiver_padding = int(50/resolution)
				skip=(slice(None,None,2),slice(None,None,2))
				for k, z in zip(range(nplots), level_plots_levels):	
					z += 1
					bx[k].contourf(lon_plot, lat_plot, potential_temperature[:,:,z], cmap='seismic',levels=15)
					bx[k].streamplot(lon_plot, lat_plot, u[:,:,z], v[:,:,z], color='white',density=1.5)
					bx[k].set_title(str(round(pressure_levels[z]/100))+' hPa')
					bx[k].set_ylabel('Latitude')
					bx[k].set_xlim((lon.min(),lon.max()))
					bx[k].set_ylim((lat.min(),lat.max()))				
				bx[-1].set_xlabel('Longitude')		
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
		
		plt.pause(0.01)
		
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

	polar_plane = beam_me_up(lat,lon,potential_temperature[:pole_lower_latitude_limit,:,0],pole_lower_latitude_limit,grid_xx.shape[0],grid_lat_coords,grid_lon_coords)
	resample = beam_me_down(lat,lon,polar_plane,pole_lower_latitude_limit,polar_x_coords,polar_y_coords)

	# f,ax = plt.subplots(2)
	# ax[0].contourf(potential_temperature[:pole_lower_latitude_limit,:,0])
	# ax[1].contourf(polar_plane)
	# plt.show()

	# time_taken = float(round(time.time() - before_plot,3))
	# print('Plotting: ',str(time_taken),'s')

	# advance time by one timestep
	t += dt

	time_taken = float(round(time.time() - initial_time,3))

	print('Time: ',str(time_taken),'s')

	if save:
		pickle.dump((potential_temperature,temperature_world,u,v,w,t,albedo), open("save_file.p","wb"))

	if np.isnan(u.max()):
		sys.exit()
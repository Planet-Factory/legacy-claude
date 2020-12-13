# toy model for use on stream
# Please give me your Twitch prime sub!

# CLimate Analysis using Digital Estimations (CLAuDE)

import numpy as np 
import matplotlib.pyplot as plt
import time, sys, pickle
import claude_low_level_library_numba as low_level
import claude_top_level_library_numba as top_level
from scipy.interpolate import interp2d, RectBivariateSpline
# from twitch import prime_sub

import atexit
from multiprocessing import Process, Queue
from plotting_utils import plotter_thread

from numba import jit, njit, prange, vectorize, guvectorize, float64, int64

######## CONTROL ########

day = 60*60*24					# define length of day (used for calculating Coriolis as well) (s)
resolution = 5					# how many degrees between latitude and longitude gridpoints
planet_radius = 6.4E6			# define the planet's radius (m)
insolation = 1370				# TOA radiation from star (W m^-2)
gravity = 9.81 					# define surface gravity for planet (m s^-2)
axial_tilt = 23.5/2				# tilt of rotational axis w.r.t. solar plane
year = 365*day					# length of year (s)

pressure_levels = np.array([1000,950,900,800,700,600,500,400,350,300,250,200,150,100,75,50,25,10,5,2,1])
pressure_levels *= 100
nlevels = len(pressure_levels)

top = -1

dt_spinup = 60*137
dt_main = 60*15.5
spinup_length = 2*day

###

advection = True 				# if you want to include advection set this to be True
smoothing_parameter_t = 0.9
smoothing_parameter_u = 0.7
smoothing_parameter_v = 0.7
smoothing_parameter_w = 0.4

save = False 					# save current state to file?
load = True  					# load initial state from file?

above = False

plot = True					# display plots of output?
diagnostic = False 				# display raw fields for diagnostic purposes
level_plots = False 			# display plots of output on vertical levels?
nplots = 3						# how many levels you want to see plots of (evenly distributed through column)

pole_lower_latitude_limit = -60
pole_higher_latitude_limit = -75

#######################################
# Chattercube/FractalMachinist

# At this point, the main math is tens of times too fast for matplotlib to update
# Set DROP_FRAMES = True to prevent a backup/queue of frames to draw
# Set DROP_FRAMES = False to allow a backlog and ensure that every timestep gets drawn
DROP_FRAMES = True

###########################

# define coordinate arrays
lat = np.arange(-90,91,resolution)
lon = np.arange(0,360,resolution)
nlat = len(lat)
nlon = len(lon)
lon_plot, lat_plot = np.meshgrid(lon, lat)
heights_plot, lat_z_plot = np.meshgrid(lat,pressure_levels[:top]/100)

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

pole_low_index_S = np.where(lat > pole_lower_latitude_limit)[0][0]
pole_high_index_S = np.where(lat > pole_higher_latitude_limit)[0][0]

# initialise grid
polar_grid_resolution = dx[-pole_low_index_S]
size_of_grid = planet_radius*np.cos(lat[-pole_low_index_S]*np.pi/180)

### south pole ###
grid_x_values_S = np.arange(-size_of_grid,size_of_grid,polar_grid_resolution)
grid_y_values_S = np.arange(-size_of_grid,size_of_grid,polar_grid_resolution)
grid_xx_S,grid_yy_S = np.meshgrid(grid_x_values_S,grid_y_values_S)

grid_side_length = len(grid_x_values_S)

grid_lat_coords_S = []
grid_lon_coords_S = []
for i in range(grid_xx_S.shape[0]):
	for j in range(grid_xx_S.shape[1]):
		lat_point = -np.arccos((grid_xx_S[i,j]**2 + grid_yy_S[i,j]**2)**0.5/planet_radius)*180/np.pi
		lon_point = 180 - np.arctan2(grid_yy_S[i,j],grid_xx_S[i,j])*180/np.pi
		grid_lat_coords_S.append(lat_point)
		grid_lon_coords_S.append(lon_point)

polar_x_coords_S = []
polar_y_coords_S = []
for i in range(pole_low_index_S):
	for j in range(nlon):
		polar_x_coords_S.append( planet_radius*np.cos(lat[i]*np.pi/180)*np.sin(lon[j]*np.pi/180) )
		polar_y_coords_S.append( -planet_radius*np.cos(lat[i]*np.pi/180)*np.cos(lon[j]*np.pi/180) )

### north pole ###
pole_low_index_N = np.where(lat < -pole_lower_latitude_limit)[0][-1]
pole_high_index_N = np.where(lat < -pole_higher_latitude_limit)[0][-1]

grid_x_values_N = np.arange(-size_of_grid,size_of_grid,polar_grid_resolution)
grid_y_values_N = np.arange(-size_of_grid,size_of_grid,polar_grid_resolution)
grid_xx_N,grid_yy_N = np.meshgrid(grid_x_values_N,grid_y_values_N)

grid_lat_coords_N = []
grid_lon_coords_N = []
for i in range(grid_xx_N.shape[0]):
	for j in range(grid_xx_N.shape[1]):
		lat_point = np.arccos((grid_xx_N[i,j]**2 + grid_yy_N[i,j]**2)**0.5/planet_radius)*180/np.pi
		lon_point = 180 - np.arctan2(grid_yy_N[i,j],grid_xx_N[i,j])*180/np.pi
		grid_lat_coords_N.append(lat_point)
		grid_lon_coords_N.append(lon_point)

polar_x_coords_N = []
polar_y_coords_N = []
for i in np.arange(pole_low_index_N,nlat):
	for j in range(nlon):
		polar_x_coords_N.append( planet_radius*np.cos(lat[i]*np.pi/180)*np.sin(lon[j]*np.pi/180) )
		polar_y_coords_N.append( -planet_radius*np.cos(lat[i]*np.pi/180)*np.cos(lon[j]*np.pi/180) )

print(pole_low_index_S,pole_high_index_S)
print(pole_low_index_N,pole_high_index_N)

def beam_me_up_2D(lats,lon,data,pole_low_index,grid_size,grid_lat_coords,grid_lon_coords):
	'''Projects data on lat-lon grid to x-y polar grid'''
	f = RectBivariateSpline(lats, lon, data)
	polar_plane = f(grid_lat_coords,grid_lon_coords,grid=False).reshape((grid_size,grid_size))
	return polar_plane
def beam_me_up(lats,lon,data,pole_low_index,grid_size,grid_lat_coords,grid_lon_coords):
	'''Projects data on lat-lon grid to x-y polar grid'''
	polar_plane = np.zeros((grid_size,grid_size,data.shape[2]))
	for k in range(data.shape[2]):
		f = RectBivariateSpline(lats, lon, data[:,:,k])
		polar_plane[:,:,k] = f(grid_lat_coords,grid_lon_coords,grid=False).reshape((grid_size,grid_size))
	return polar_plane
def beam_me_down(lon,data,pole_low_index,grid_x_values,grid_y_values,polar_x_coords,polar_y_coords):
	'''projects data from x-y polar grid onto lat-lon grid'''
	resample = np.zeros((int(len(polar_x_coords)/len(lon)),len(lon),data.shape[2]))
	for k in range(data.shape[2]):
		f = RectBivariateSpline(x=grid_x_values, y=grid_y_values, z=data[:,:,k])
		resample[:,:,k] = f(polar_x_coords,polar_y_coords,grid=False).reshape((int(len(polar_x_coords)/len(lon)),len(lon)))
	return resample

@jit(parallel=True)
def combine_data(pole_low_index,pole_high_index,polar_data,reprojected_data): 
	output = np.zeros_like(polar_data)
	overlap = abs(pole_low_index - pole_high_index)

	if lat[pole_low_index] < 0:
		for k in range(output.shape[2]):
			for i in range(pole_low_index):
				
				if i < pole_high_index:
					scale_polar_data = 0.0
					scale_reprojected_data = 1.0
				else:
					scale_polar_data = (i-pole_high_index)/overlap
					scale_reprojected_data = 1 - (i-pole_high_index)/overlap
				output[i,:,k] = scale_reprojected_data*reprojected_data[i,:,k] + scale_polar_data*polar_data[i,:,k]
	else:	# PROBLEM APPEARS TO BE HERE IN NORTH POLE BOUNDARY
		for k in range(output.shape[2]):
			for i in range(nlat-pole_low_index):
				
				if i + pole_low_index + 1 > pole_high_index:
					scale_polar_data = 0.0
					scale_reprojected_data = 1.0
				else:
					scale_polar_data = i/overlap
					scale_reprojected_data = 1-(i/overlap)
				output[i,:,k] = scale_reprojected_data*reprojected_data[i,:,k] + scale_polar_data*polar_data[i,:,k]

	return output

@jit
def grid_x_gradient(data,i,j,k):
	if j == 0:
		value = (data[i,j+1,k] - data[i,j,k])/(polar_grid_resolution)
	elif j == grid_side_length-1:
		value = (data[i,j,k] - data[i,j-1,k])/(polar_grid_resolution)
	else:
		value = (data[i,j+1,k] - data[i,j-1,k])/(2*polar_grid_resolution)
	return value


@jit
def grid_y_gradient(data,i,j,k):
	if i == 0:
		value = (data[i+1,j,k] - data[i,j,k])/(polar_grid_resolution)
	elif i == grid_side_length-1:
		value = (data[i,j,k] - data[i-1,j,k])/(polar_grid_resolution)
	else:
		value = (data[i+1,j,k] - data[i-1,j,k])/(2*polar_grid_resolution)
	return value

@jit
def grid_p_gradient(data,i,j,k,pressure_levels):
	if k == 0:
		value = (data[i,j,k+1]-data[i,j,k])/(pressure_levels[k+1]-pressure_levels[k])
	elif k == nlevels-1:
		value = (data[i,j,k]-data[i,j,k-1])/(pressure_levels[k]-pressure_levels[k-1])
	else:
		value = (data[i,j,k+1]-data[i,j,k-1])/(pressure_levels[k+1]-pressure_levels[k-1])
	return value

@jit
def grid_velocities_north(polar_plane,grid_side_length,coriolis_plane):
	x_dot = np.zeros_like(polar_plane)
	y_dot = np.zeros_like(polar_plane)
	for i in prange(grid_side_length):
		for j in prange(grid_side_length):
			for k in prange(polar_plane.shape[2]):
				# x_dot[i,j,k] = -grid_y_gradient(polar_plane,i,j,k)/coriolis_plane[i,j]
				# y_dot[i,j,k] = grid_x_gradient(polar_plane,i,j,k)/coriolis_plane[i,j]
				x_dot[i,j,k] = dt_main*(- x_dot[i,j,k]*grid_x_gradient(x_dot,i,j,k) - y_dot[i,j,k]*grid_y_gradient(x_dot,i,j,k) + coriolis_plane[i,j]*y_dot[i,j,k] - grid_x_gradient(polar_plane,i,j,k) - 1E-4*x_dot[i,j,k])
				y_dot[i,j,k] = dt_main*(- x_dot[i,j,k]*grid_x_gradient(y_dot,i,j,k) - y_dot[i,j,k]*grid_y_gradient(y_dot,i,j,k) - coriolis_plane[i,j]*x_dot[i,j,k] - grid_y_gradient(polar_plane,i,j,k) - 1E-4*y_dot[i,j,k])
	return x_dot,y_dot


@njit(parallel=True)
def grid_velocities_south(polar_plane,grid_side_length,coriolis_plane):
	x_dot = np.zeros_like(polar_plane)
	y_dot = np.zeros_like(polar_plane)
	for i in prange(grid_side_length):
		for j in prange(grid_side_length):
			for k in prange(polar_plane.shape[2]):
				# x_dot[i,j,k] = -grid_y_gradient(polar_plane,i,j,k)/coriolis_plane[i,j]
				# y_dot[i,j,k] = grid_x_gradient(polar_plane,i,j,k)/coriolis_plane[i,j]
				x_dot[i,j,k] = dt_main*(- x_dot[i,j,k]*grid_x_gradient(x_dot,i,j,k) - y_dot[i,j,k]*grid_y_gradient(x_dot,i,j,k) + coriolis_plane[i,j]*y_dot[i,j,k] - grid_x_gradient(polar_plane,i,j,k) - 1E-4*x_dot[i,j,k])
				y_dot[i,j,k] = dt_main*(- x_dot[i,j,k]*grid_x_gradient(y_dot,i,j,k) - y_dot[i,j,k]*grid_y_gradient(y_dot,i,j,k) - coriolis_plane[i,j]*x_dot[i,j,k] - grid_y_gradient(polar_plane,i,j,k) - 1E-4*y_dot[i,j,k])
	return x_dot,y_dot

@njit(parallel=True)
def grid_vertical_velocity(x_dot,y_dot,pressure_levels,gravity,temperature):
	output = np.zeros_like(x_dot)
	for i in prange(output.shape[0]):
		for j in prange(output.shape[1]):
			for k in prange(output.shape[2]):
				output[i,j,k] = - (pressure_levels[k]-pressure_levels[k-1])*pressure_levels[k]*gravity*( grid_x_gradient(x_dot,i,j,k) + grid_y_gradient(y_dot,i,j,k) )/(287*temperature[i,j,k])
	return output




def project_velocities_north(lon,x_dot,y_dot,pole_low_index_N,pole_high_index_N,grid_x_values_N,grid_y_values_N,polar_x_coords_N,polar_y_coords_N,data):
	reproj_x_dot = beam_me_down(lon,x_dot,pole_low_index_N,grid_x_values_N,grid_y_values_N,polar_x_coords_N,polar_y_coords_N)		
	reproj_y_dot = beam_me_down(lon,y_dot,pole_low_index_N,grid_x_values_N,grid_y_values_N,polar_x_coords_N,polar_y_coords_N)

	return _project_velocities_north(data, reproj_x_dot, reproj_y_dot, lon);

@njit(parallel=True)
def _project_velocities_north(data, reproj_x_dot, reproj_y_dot, lon):
	reproj_u = np.zeros_like(data)
	reproj_v = np.zeros_like(data)

	for i in prange(data.shape[0]):
		for j in prange(data.shape[1]):
			for k in prange(data.shape[2]):
				reproj_u[i,j,k] = - reproj_x_dot[i,j,k]*np.sin(lon[j]*np.pi/180) - reproj_y_dot[i,j,k]*np.cos(lon[j]*np.pi/180)
				reproj_v[i,j,k] = reproj_x_dot[i,j,k]*np.cos(lon[j]*np.pi/180) - reproj_y_dot[i,j,k]*np.sin(lon[j]*np.pi/180)

	return reproj_u, reproj_v





def project_velocities_south(lon,x_dot,y_dot,pole_low_index_S,pole_high_index_S,grid_x_values_S,grid_y_values_S,polar_x_coords_S,polar_y_coords_S,data):
	reproj_x_dot = beam_me_down(lon,x_dot,pole_low_index_S,grid_x_values_S,grid_y_values_S,polar_x_coords_S,polar_y_coords_S)		
	reproj_y_dot = beam_me_down(lon,y_dot,pole_low_index_S,grid_x_values_S,grid_y_values_S,polar_x_coords_S,polar_y_coords_S)

	return _project_velocities_south(data, reproj_x_dot, reproj_y_dot, lon)

@njit(parallel=True)
def _project_velocities_south(data, reproj_x_dot, reproj_y_dot, lon):
	reproj_u = np.zeros_like(data)
	reproj_v = np.zeros_like(data)

	for i in prange(data.shape[0]):
		for j in prange(data.shape[1]):
			for k in prange(data.shape[2]):
				reproj_u[i,j,k] = reproj_x_dot[i,j,k]*np.sin(lon[j]*np.pi/180) + reproj_y_dot[i,j,k]*np.cos(lon[j]*np.pi/180)
				reproj_v[i,j,k] = - reproj_x_dot[i,j,k]*np.cos(lon[j]*np.pi/180) + reproj_y_dot[i,j,k]*np.sin(lon[j]*np.pi/180)

	return reproj_u, reproj_v

@njit(parallel=True)
def polar_plane_advect(data,x_dot,y_dot,z_dot,pressure_levels):
	output = np.zeros_like(data)
	data_x_dot = data*x_dot
	data_y_dot = data*y_dot
	data_z_dot = data*z_dot
	for i in prange(data.shape[0]):
		for j in prange(data.shape[1]):
			for k in prange(data.shape[2]):
				output[i,j,k] = grid_x_gradient(data_x_dot,i,j,k) + grid_y_gradient(data_y_dot,i,j,k) + grid_p_gradient(data_z_dot,i,j,k,pressure_levels)
	return output

def init_plot():
	plots = lon_plot, lat_plot, heights_plot, lat_z_plot, nplots
	uvw = u, v, w
	latlon = lat, lon
	data = potential_temperature, pressure_levels, atmosp_addition, temperature_world, t
	misc = top, resolution, day
	flags = above, advection, diagnostic, plot, level_plots

	pole_indices = pole_low_index_N, pole_low_index_S
	grid_values = grid_x_values_N, grid_x_values_S, grid_y_values_N, grid_y_values_S


	q = Queue()
	p = Process(target=plotter_thread, args=((q, plots, uvw, latlon, data, misc, flags, pole_indices, grid_values)))
	p.start()
	atexit.register(p.join)
	return q

def update_plot():
	if not DROP_FRAMES or Q.qsize() < 2:
		uvw = u, v, w
		data = potential_temperature, pressure_levels, atmosp_addition, temperature_world, t, north_temperature_data, north_temperature_resample, north_polar_plane_temperature, south_temperature_data, south_temperature_resample, south_polar_plane_temperature


		reprojections = reproj_u_N, reproj_u_S, reproj_v_N, reproj_v_S

		Q.put([uvw, data, reprojections, velocity]);
	else:
		print("[NOTE] Skipping Drawing to prevent backlog")

def kill_plot():
	Q.put('stop');
atexit.register(kill_plot)	

# create Coriolis data on north and south planes
data = np.zeros((nlat-pole_low_index_N,nlon))
for i in np.arange(pole_low_index_N,nlat):
	data[i-pole_low_index_N,:] = coriolis[i]
coriolis_plane_N = beam_me_up_2D(lat[pole_low_index_N:],lon,data,pole_low_index_N,grid_xx_N.shape[0],grid_lat_coords_N,grid_lon_coords_N)
data = np.zeros((pole_low_index_S,nlon))
for i in range(pole_low_index_S):
	data[i,:] = coriolis[i]
coriolis_plane_S = beam_me_up_2D(lat[:pole_low_index_S],lon,data,pole_low_index_S,grid_xx_S.shape[0],grid_lat_coords_S,grid_lon_coords_S)

x_dot_N = np.zeros((grid_side_length,grid_side_length,nlevels))
y_dot_N = np.zeros((grid_side_length,grid_side_length,nlevels))

x_dot_S = np.zeros((grid_side_length,grid_side_length,nlevels))
y_dot_S = np.zeros((grid_side_length,grid_side_length,nlevels))

#######################################################################################################################################################################################################################

# INITIATE TIME
t = 0

if load:
	# load in previous save file
	potential_temperature,temperature_world,u,v,w,t,albedo = pickle.load(open("save_file.p","rb"))

Q = init_plot()
print("[NOTE] The Numba compiler will spend the first 20-30 seconds compiling. You'll notice when it's done.")
while True:
# while t/day < 2.5:

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
	print('U: ',round(u.max(),2),' - ',round(u.min(),2),' V: ',round(v.max(),2),' - ',round(v.min(),2),' W: ',round(w.max(),2),' - ',round(w.min(),4))

	before_radiation = time.time()
	temperature_world, potential_temperature = top_level.radiation_calculation(temperature_world, potential_temperature, pressure_levels, heat_capacity_earth, albedo, insolation, lat, lon, t, dt, day, year, axial_tilt)
	potential_temperature = top_level.smoothing_3D(potential_temperature,smoothing_parameter_t)
	

	time_taken = float(round(time.time() - before_radiation,3))
	print('Radiation: ',str(time_taken),'s')

	# update geopotential field
	for k in np.arange(1,nlevels):
		geopotential[:,:,k] = geopotential[:,:,k-1] - potential_temperature[:,:,k]*(sigma[k]-sigma[k-1])

	if velocity:

		before_velocity = time.time()
		u,v = top_level.velocity_calculation(u,v,w,pressure_levels,geopotential,potential_temperature,coriolis,gravity,dx,dy,dt)

		u = top_level.smoothing_3D(u,smoothing_parameter_u)
		v = top_level.smoothing_3D(v,smoothing_parameter_v)

		w = top_level.w_calculation(u,v,w,pressure_levels,geopotential,potential_temperature,coriolis,gravity,dx,dy,dt)
		w = top_level.smoothing_3D(w,smoothing_parameter_w,0.25)

		u[:,:,-1] *= 0.1
		v[:,:,-1] *= 0.1

		for k in range(nlevels):
			w[:,:,k] *= pressure_levels[k]/pressure_levels[0]
		
		w[:,:,0] = -w[:,:,1]
		# w[:,:,2] *= 0.1
		# w[:,:,3] *= 0.5

		# u *= 0
		# v *= 0
		# w *= 0

		time_taken = float(round(time.time() - before_velocity,3))
		print('Velocity: ',str(time_taken),'s')

		if advection:
			before_advection = time.time()

			# allow for thermal advection in the atmosphere
			atmosp_addition = dt*top_level.divergence_with_scalar(potential_temperature,u,v,w,dx,dy,pressure_levels)

			time_taken = float(round(time.time() - before_advection,3))
			print('Advection: ',str(time_taken),'s')

			before_projection = time.time()

			###################################################################

			### north pole ###
			north_temperature_data = potential_temperature[pole_low_index_N:,:,:]
			north_polar_plane_temperature = beam_me_up(lat[pole_low_index_N:],lon,north_temperature_data,pole_low_index_N,grid_xx_N.shape[0],grid_lat_coords_N,grid_lon_coords_N)
			north_polar_plane_actual_temperature = low_level.theta_to_t(north_polar_plane_temperature,pressure_levels)
			
			north_geopotential_data = geopotential[pole_low_index_N:,:,:]
			north_polar_plane_geopotential = beam_me_up(lat[pole_low_index_N:],lon,north_geopotential_data,pole_low_index_N,grid_xx_N.shape[0],grid_lat_coords_N,grid_lon_coords_N)
			
			# calculate local velocity on Cartesian grid (CARTESIAN)
			x_dot_add,y_dot_add = grid_velocities_north(north_polar_plane_geopotential,grid_side_length,coriolis_plane_N)
			x_dot_N += x_dot_add
			y_dot_N += y_dot_add
			z_dot = grid_vertical_velocity(x_dot_N,y_dot_N,pressure_levels,gravity,north_polar_plane_actual_temperature)
			# advect temperature field, isolate field to subtract from existing temperature field (CARTESIAN)
			north_polar_plane_addition = polar_plane_advect(north_polar_plane_temperature,x_dot_N,y_dot_N,z_dot,pressure_levels)
			
			# project velocities onto polar grid (POLAR)
			reproj_u_N, reproj_v_N = project_velocities_north(lon,x_dot_N,y_dot_N,pole_low_index_N,pole_high_index_N,grid_x_values_N,grid_y_values_N,polar_x_coords_N,polar_y_coords_N,north_temperature_data)
			reproj_w_N = beam_me_down(lon,z_dot,pole_low_index_N,grid_x_values_N,grid_y_values_N,polar_x_coords_N,polar_y_coords_N)
			# combine velocities with those calculated on polar grid (POLAR)
			reproj_u_N = combine_data(pole_low_index_N,pole_high_index_N,u[pole_low_index_N:,:,:],reproj_u_N)
			reproj_v_N = combine_data(pole_low_index_N,pole_high_index_N,v[pole_low_index_N:,:,:],reproj_v_N)
			reproj_w_N = combine_data(pole_low_index_N,pole_high_index_N,w[pole_low_index_N:,:,:],reproj_w_N)
			# add the combined velocities to the global velocity arrays
			u[pole_low_index_N:,:,:] = reproj_u_N
			v[pole_low_index_N:,:,:] = reproj_v_N
			w[pole_low_index_N:,:,:] = reproj_w_N

			north_temperature_resample = combine_data(pole_low_index_N,pole_high_index_N,north_temperature_data,beam_me_down(lon,north_polar_plane_temperature,pole_low_index_N,grid_x_values_N,grid_y_values_N,polar_x_coords_N,polar_y_coords_N))

			# project addition to temperature field onto polar grid (POLAR)
			north_reprojected_addition = -beam_me_down(lon,north_polar_plane_addition,pole_low_index_N,grid_x_values_N,grid_y_values_N,polar_x_coords_N,polar_y_coords_N)
			# combine addition calculated on polar grid with that calculated on the cartestian grid (POLAR)
			north_addition_smoothed = combine_data(pole_low_index_N,pole_high_index_N,atmosp_addition[pole_low_index_N:,:,:],north_reprojected_addition)
			# add the blended/combined addition to global temperature addition array
			atmosp_addition[pole_low_index_N:,:,:] = north_addition_smoothed

			###################################################################

			### south pole ###
			south_temperature_data = potential_temperature[:pole_low_index_S,:,:]
			south_polar_plane_temperature = beam_me_up(lat[:pole_low_index_S],lon,south_temperature_data,pole_low_index_S,grid_xx_S.shape[0],grid_lat_coords_S,grid_lon_coords_S)
			south_polar_plane_actual_temperature = low_level.theta_to_t(south_polar_plane_temperature,pressure_levels)

			south_geopotential_data = geopotential[:pole_low_index_S,:,:]
			south_polar_plane_geopotential = beam_me_up(lat[:pole_low_index_S],lon,south_geopotential_data,pole_low_index_S,grid_xx_S.shape[0],grid_lat_coords_S,grid_lon_coords_S)
			
			x_dot_add,y_dot_add = grid_velocities_south(south_polar_plane_geopotential,grid_side_length,coriolis_plane_S)
			x_dot_S += x_dot_add
			y_dot_S += y_dot_add
			z_dot = grid_vertical_velocity(x_dot_S,y_dot_S,pressure_levels,gravity,south_polar_plane_actual_temperature)
			south_polar_plane_addition = polar_plane_advect(south_polar_plane_temperature,x_dot_S,y_dot_S,z_dot,pressure_levels)

			reproj_u_S, reproj_v_S = project_velocities_south(lon,x_dot_S,y_dot_S,pole_low_index_S,pole_high_index_S,grid_x_values_S,grid_y_values_S,polar_x_coords_S,polar_y_coords_S,south_temperature_data)
			reproj_w_S = beam_me_down(lon,z_dot,pole_low_index_S,grid_x_values_S,grid_y_values_S,polar_x_coords_S,polar_y_coords_S)
			
			reproj_u_S = combine_data(pole_low_index_S,pole_high_index_S,u[:pole_low_index_S,:,:],reproj_u_S)
			reproj_v_S = combine_data(pole_low_index_S,pole_high_index_S,v[:pole_low_index_S,:,:],reproj_v_S)
			reproj_w_S = combine_data(pole_low_index_S,pole_high_index_S,w[:pole_low_index_S,:,:],reproj_w_S)

			south_temperature_resample = combine_data(pole_low_index_S,pole_high_index_S,south_temperature_data,beam_me_down(lon,south_polar_plane_temperature,pole_low_index_S,grid_x_values_S,grid_y_values_S,polar_x_coords_S,polar_y_coords_S))
			
			south_reprojected_addition = beam_me_down(lon,south_polar_plane_addition,pole_low_index_S,grid_x_values_S,grid_y_values_S,polar_x_coords_S,polar_y_coords_S)	
			south_addition_smoothed = combine_data(pole_low_index_S,pole_high_index_S,atmosp_addition[:pole_low_index_S,:,:],south_reprojected_addition)
			atmosp_addition[:pole_low_index_S,:,:] = south_addition_smoothed		

			u[:pole_low_index_S,:,:] = reproj_u_S
			v[:pole_low_index_S,:,:] = reproj_v_S
			w[:pole_low_index_S,:,:] = reproj_w_S

			# atmosp_addition = top_level.smoothing_3D(atmosp_addition,0.4,0.4)
			atmosp_addition[:,:,0] = atmosp_addition[:,:,1]

			###################################################################

			potential_temperature -= atmosp_addition

			time_taken = float(round(time.time() - before_projection,3))
			print('Projection: ',str(time_taken),'s')

	update_plot()

	# advance time by one timesteps
	t += dt

	time_taken = float(round(time.time() - initial_time,3))

	print('Time: ',str(time_taken),'s')

	if save:
		pickle.dump((potential_temperature,temperature_world,u,v,w,t,albedo), open("save_file.p","wb"))

	if np.isnan(u.max()):
		Q.put('STOP');
		sys.exit()
Q.put('STOP')
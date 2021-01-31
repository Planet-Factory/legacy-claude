# CLimate Analysis using Digital Estimations (CLAuDE)

import numpy as np 
import matplotlib.pyplot as plt
import time, sys, pickle, os, yaml
import claude_low_level_library as low_level
import claude_top_level_library as top_level

from yaml import Loader
from definitions import CONFIG_PATH
from model.claude_config_file import ClaudeConfigFile, PlanetConfigFile, SaveConfigFile, ViewConfigFile
from model.claude_config import ClaudeConfig
from model.pole_enum import PoleType
# from twitch import prime_sub

######## CONTROL ########

DEFAULT_CONFIG_FILE = "DefaultClaudeConfig.yaml"

## Load Configuration
config_file = open(os.path.join(CONFIG_PATH, DEFAULT_CONFIG_FILE))
claude_config_file = yaml.load(config_file, Loader=Loader)
claude_config = ClaudeConfig.load_from_file(claude_config_file=claude_config_file)

###
planet_config = claude_config.planet_config
dt_spinup = 60*17.2					# timestep for initial period where the model only calculates radiative effects
dt_main = 60*9.2					# timestep for the main sequence where the model calculates velocities/advection
spinup_length = 0*planet_config.day 				# how long the model should only calculate radiative effects
###
smoothing_config = claude_config.smoothing_config
###
save_config = claude_config.save_config

###
view_config = claude_config.view_config
###

pole_lower_latitude_limit = -75		# how far north polar plane data is calculated from the south pole (do not set this beyond 45!) [mirrored to north pole as well]
pole_higher_latitude_limit = -85	# how far south regular gridded data is calculated (do not set beyond about 80) [also mirrored to north pole]

###
coordinate_grid = claude_config.coordinate_grid

##########################

if not save_config.load:
	# initialise arrays for various physical fields
	coordinate_grid.temperature_world += 290
	potential_temperature = np.zeros((coordinate_grid.nlat,coordinate_grid.nlon,planet_config.nlevels))
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
	temp_profile = np.interp(x=planet_config.pressure_levels[::-1],xp=standard_pressure[::-1],fp=standard_temp[::-1])[::-1]
	for k in range(planet_config.nlevels):
		potential_temperature[:,:,k] = temp_profile[k]

	potential_temperature = low_level.t_to_theta(potential_temperature,planet_config.pressure_levels)
	geopotential = np.zeros_like(potential_temperature)

initial_setup = True
if initial_setup:
	sigma = np.zeros_like(planet_config.pressure_levels)
	kappa = 287/1000
	for i in range(len(sigma)):
		sigma[i] = 1E3*(planet_config.pressure_levels[i]/planet_config.pressure_levels[0])**kappa

	heat_capacity_earth = np.zeros_like(coordinate_grid.temperature_world) + 1E6

	# heat_capacity_earth[15:36,30:60] = 1E7
	# heat_capacity_earth[30:40,80:90] = 1E7

	albedo_variance = 0.001
	albedo = np.random.uniform(-albedo_variance,albedo_variance, (coordinate_grid.nlat, coordinate_grid.nlon)) + 0.2
	albedo = np.zeros((coordinate_grid.nlat, coordinate_grid.nlon)) + 0.2

	specific_gas = 287
	thermal_diffusivity_roc = 1.5E-6

	# define planet size and various geometric constants
	circumference = 2*np.pi*planet_config.planet_radius
	circle = np.pi*planet_config.planet_radius**2
	sphere = 4*np.pi*planet_config.planet_radius**2

	# define how far apart the gridpoints are: note that we use central difference derivatives, and so these distances are actually twice the distance between gridboxes
	dy = circumference/coordinate_grid.nlat
	dx = np.zeros(coordinate_grid.nlat)
	coriolis = np.zeros(coordinate_grid.nlat)	# also define the coriolis parameter here
	angular_speed = 2*np.pi/planet_config.day
	for i in range(coordinate_grid.nlat):
		dx[i] = dy*np.cos(coordinate_grid.lat[i]*np.pi/180)
		coriolis[i] = angular_speed*np.sin(coordinate_grid.lat[i]*np.pi/180)

setup_grids = True
if setup_grids:

	grid_pad = 2
	
	pole_low_index_S = np.where(coordinate_grid.lat > pole_lower_latitude_limit)[0][0]
	pole_high_index_S = np.where(coordinate_grid.lat > pole_higher_latitude_limit)[0][0]

	# initialise grid
	polar_grid_resolution = dx[pole_low_index_S]
	size_of_grid = planet_config.planet_radius*np.cos(coordinate_grid.lat[pole_low_index_S+grid_pad]*np.pi/180.0)

	### south pole ###
	grid_x_values_S = np.arange(-size_of_grid,size_of_grid,polar_grid_resolution)
	grid_y_values_S = np.arange(-size_of_grid,size_of_grid,polar_grid_resolution)
	grid_xx_S,grid_yy_S = np.meshgrid(grid_x_values_S,grid_y_values_S)

	grid_side_length = len(grid_x_values_S)

	grid_lat_coords_S = (-np.arccos(((grid_xx_S**2 + grid_yy_S**2)**0.5)/planet_config.planet_radius)*180.0/np.pi).flatten()
	grid_lon_coords_S = (180.0 - np.arctan2(grid_yy_S,grid_xx_S)*180.0/np.pi).flatten()

	polar_x_coords_S = []
	polar_y_coords_S = []
	for i in range(pole_low_index_S):
		for j in range(coordinate_grid.nlon):
			polar_x_coords_S.append( planet_config.planet_radius*np.cos(coordinate_grid.lat[i]*np.pi/180.0)*np.sin(coordinate_grid.lon[j]*np.pi/180.0) )
			polar_y_coords_S.append(-planet_config.planet_radius*np.cos(coordinate_grid.lat[i]*np.pi/180.0)*np.cos(coordinate_grid.lon[j]*np.pi/180.0) )

	### north pole ###
	
	pole_low_index_N  	= 	np.where(coordinate_grid.lat < -pole_lower_latitude_limit)[0][-1]
	pole_high_index_N 	= 	np.where(coordinate_grid.lat < -pole_higher_latitude_limit)[0][-1]

	grid_x_values_N 	= 	np.arange(-size_of_grid,size_of_grid,polar_grid_resolution)
	grid_y_values_N 	= 	np.arange(-size_of_grid,size_of_grid,polar_grid_resolution)
	grid_xx_N,grid_yy_N = 	np.meshgrid(grid_x_values_N,grid_y_values_N)

	grid_lat_coords_N 	= 	(np.arccos((grid_xx_N**2 + grid_yy_N**2)**0.5/planet_config.planet_radius)*180.0/np.pi).flatten()
	grid_lon_coords_N 	= 	(180.0 - np.arctan2(grid_yy_N,grid_xx_N)*180.0/np.pi).flatten()

	polar_x_coords_N 	= 	[]
	polar_y_coords_N 	= 	[]

	for i in np.arange(pole_low_index_N,coordinate_grid.nlat):
		for j in range(coordinate_grid.nlon):
			polar_x_coords_N.append( planet_config.planet_radius*np.cos(coordinate_grid.lat[i]*np.pi/180.0)*np.sin(coordinate_grid.lon[j]*np.pi/180.0) )
			polar_y_coords_N.append(-planet_config.planet_radius*np.cos(coordinate_grid.lat[i]*np.pi/180.0)*np.cos(coordinate_grid.lon[j]*np.pi/180.0) )

	indices = (pole_low_index_N,pole_high_index_N,pole_low_index_S,pole_high_index_S)
	grids   = (grid_xx_N.shape[0],grid_xx_S.shape[0])

	# create Coriolis data on north and south planes
	data = np.zeros((coordinate_grid.nlat-pole_low_index_N+grid_pad,coordinate_grid.nlon))
	for i in np.arange(pole_low_index_N-grid_pad,coordinate_grid.nlat):
		data[i-pole_low_index_N,:] = coriolis[i]
	coriolis_plane_N = low_level.beam_me_up_2D(coordinate_grid.lat[(pole_low_index_N-grid_pad):],coordinate_grid.lon,data,grids[0],grid_lat_coords_N,grid_lon_coords_N)
	data = np.zeros((pole_low_index_S+grid_pad,coordinate_grid.nlon))
	for i in range(pole_low_index_S+grid_pad):
		data[i,:] = coriolis[i]
	coriolis_plane_S = low_level.beam_me_up_2D(coordinate_grid.lat[:(pole_low_index_S+grid_pad)],coordinate_grid.lon,data,grids[1],grid_lat_coords_S,grid_lon_coords_S)

	x_dot_N = np.zeros((grids[0],grids[0],planet_config.nlevels))
	y_dot_N = np.zeros((grids[0],grids[0],planet_config.nlevels))
	x_dot_S = np.zeros((grids[1],grids[1],planet_config.nlevels))
	y_dot_S = np.zeros((grids[1],grids[1],planet_config.nlevels))

	coords  = grid_lat_coords_N,grid_lon_coords_N,grid_x_values_N,grid_y_values_N,polar_x_coords_N,polar_y_coords_N,grid_lat_coords_S,grid_lon_coords_S,grid_x_values_S,grid_y_values_S,polar_x_coords_S,polar_y_coords_S

#######################################################################################################################################################################################################################

# INITIATE TIME
t = 0.0

if save_config.load:
	# load in previous save file
	potential_temperature,coordinate_grid.temperature_world,u,v,w,x_dot_N,y_dot_N,x_dot_S,y_dot_S,t,albedo,tracer = pickle.load(open("save_file.p","rb"))

sample_level = 5
tracer = np.zeros_like(potential_temperature)

last_plot = t-0.1
last_save = t-0.1

if view_config.plot:
	if not view_config.diagnostic:
		# set up plot
		f, ax = plt.subplots(2,figsize=(9,9))
		f.canvas.set_window_title('CLAuDE')
		ax[0].contourf(coordinate_grid.lon_plot, coordinate_grid.lat_plot, coordinate_grid.temperature_world, cmap='seismic')
		ax[0].streamplot(coordinate_grid.lon_plot, coordinate_grid.lat_plot, u[:,:,0], v[:,:,0], color='white',density=1)
		test = ax[1].contourf(coordinate_grid.heights_plot, coordinate_grid.lat_z_plot, np.transpose(np.mean(low_level.theta_to_t(potential_temperature,planet_config.pressure_levels),axis=1))[:view_config.top,:], cmap='seismic',levels=15)
		ax[1].contour(coordinate_grid.heights_plot,coordinate_grid.lat_z_plot, np.transpose(np.mean(u,axis=1))[:view_config.top,:], colors='white',levels=20,linewidths=1,alpha=0.8)
		ax[1].quiver(coordinate_grid.heights_plot, coordinate_grid.lat_z_plot, np.transpose(np.mean(v,axis=1))[:view_config.top,:],np.transpose(np.mean(10*w,axis=1))[:view_config.top,:],color='black')
		plt.subplots_adjust(left=0.1, right=0.75)
		ax[0].set_title('Surface temperature')
		ax[0].set_xlim(coordinate_grid.lon.min(),coordinate_grid.lon.max())
		ax[1].set_title('Atmosphere temperature')
		ax[1].set_xlim(coordinate_grid.lat.min(),coordinate_grid.lat.max())
		ax[1].set_ylim((planet_config.pressure_levels.max()/100,planet_config.pressure_levels[:view_config.top].min()/100))
		ax[1].set_yscale('log')		
		ax[1].set_ylabel('Pressure (hPa)')
		ax[1].set_xlabel('Latitude')
		cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
		f.colorbar(test, cax=cbar_ax)
		cbar_ax.set_title('Temperature (K)')
		f.suptitle( 'Time ' + str(round(t/planet_config.day,2)) + ' days' )

	else:
		# set up plot
		f, ax = plt.subplots(2,2,figsize=(9,9))
		f.canvas.set_window_title('CLAuDE')
		ax[0,0].contourf(coordinate_grid.heights_plot, coordinate_grid.lat_z_plot, np.transpose(np.mean(u,axis=1))[:view_config.top,:], cmap='seismic')
		ax[0,0].set_title('u')
		ax[0,1].contourf(coordinate_grid.heights_plot, coordinate_grid.lat_z_plot, np.transpose(np.mean(v,axis=1))[:view_config.top,:], cmap='seismic')
		ax[0,1].set_title('v')
		ax[1,0].contourf(coordinate_grid.heights_plot, coordinate_grid.lat_z_plot, np.transpose(np.mean(w,axis=1))[:view_config.top,:], cmap='seismic')
		ax[1,0].set_title('w')
		ax[1,1].contourf(coordinate_grid.heights_plot, coordinate_grid.lat_z_plot, np.transpose(np.mean(atmosp_addition,axis=1))[:view_config.top,:], cmap='seismic')
		ax[1,1].set_title('atmosp_addition')
		for axis in ax.ravel():
			axis.set_ylim((planet_config.pressure_levels.max()/100,planet_config.pressure_levels[:view_config.top].min()/100))
			axis.set_yscale('log')
		f.suptitle( 'Time ' + str(round(t/planet_config.day,2)) + ' days' )

	if view_config.level_plots:

		level_divisions = int(np.floor(planet_config.nlevels/view_config.nplots))
		level_plots_levels = range(planet_config.nlevels)[::level_divisions][::-1]

		g, bx = plt.subplots(view_config.nplots,figsize=(9,8),sharex=True)
		g.canvas.set_window_title('CLAuDE pressure levels')
		for k, z in zip(range(view_config.nplots), level_plots_levels):	
			z += 1
			bx[k].contourf(coordinate_grid.lon_plot, coordinate_grid.lat_plot, potential_temperature[:,:,z], cmap='seismic')
			bx[k].set_title(str(planet_config.pressure_levels[z]/100)+' hPa')
			bx[k].set_ylabel('Latitude')
		bx[-1].set_xlabel('Longitude')
	
	plt.ion()
	plt.show()
	plt.pause(2)

	if not view_config.diagnostic:
		ax[0].cla()
		ax[1].cla()
		if view_config.level_plots:
			for k in range(view_config.nplots):
				bx[k].cla()		
	else:
		ax[0,0].cla()
		ax[0,1].cla()	
		ax[1,0].cla()
		ax[1,1].cla()

if view_config.above:
	g,gx = plt.subplots(1,3, figsize=(15,5))
	plt.ion()
	plt.show()

def plotting_routine():
	
	quiver_padding = int(12/planet_config.resolution)

	if view_config.plot:
		if view_config.verbose:	before_plot = time.time()
		# update plot
		if not view_config.diagnostic:
			
			# ax[0].contourf(coordinate_grid.lon_plot, coordinate_grid.lat_plot, coordinate_grid.temperature_world, cmap='seismic',levels=15)
					
			# field = np.copy(w)[:,:,sample_level]
			field = np.copy(atmosp_addition)[:,:,sample_level]
			ax[0].contourf(coordinate_grid.lon_plot, coordinate_grid.lat_plot, field, cmap='seismic',levels=15)
			ax[0].contour(coordinate_grid.lon_plot, coordinate_grid.lat_plot, tracer[:,:,sample_level], alpha=0.5, antialiased=True, levels=np.arange(0.01,1.01,0.01))
			
			if velocity:	ax[0].quiver(coordinate_grid.lon_plot[::quiver_padding,::quiver_padding], coordinate_grid.lat_plot[::quiver_padding,::quiver_padding], u[::quiver_padding,::quiver_padding,sample_level], v[::quiver_padding,::quiver_padding,sample_level], color='white')
			# ax[0].set_title('$\it{Ground} \quad \it{temperature}$')

			ax[0].set_xlim((coordinate_grid.lon.min(),coordinate_grid.lon.max()))
			ax[0].set_ylim((coordinate_grid.lat.min(),coordinate_grid.lat.max()))
			ax[0].set_ylabel('Latitude')
			ax[0].axhline(y=0,color='black',alpha=0.3)
			ax[0].set_xlabel('Longitude')

			test = ax[1].contourf(coordinate_grid.heights_plot, coordinate_grid.lat_z_plot, np.transpose(np.mean(low_level.theta_to_t(potential_temperature,planet_config.pressure_levels),axis=1))[:view_config.top,:], cmap='seismic',levels=15)
			# test = ax[1].contourf(coordinate_grid.heights_plot, coordinate_grid.lat_z_plot, np.transpose(np.mean(atmosp_addition,axis=1))[:view_config.top,:], cmap='seismic',levels=15)
			# test = ax[1].contourf(coordinate_grid.heights_plot, coordinate_grid.lat_z_plot, np.transpose(np.mean(potential_temperature,axis=1)), cmap='seismic',levels=15)
			ax[1].contour(coordinate_grid.heights_plot, coordinate_grid.lat_z_plot, np.transpose(np.mean(tracer,axis=1))[:view_config.top,:], alpha=0.5, antialiased=True, levels=np.arange(0.001,1.01,0.01))

			if velocity:
				ax[1].contour(coordinate_grid.heights_plot,coordinate_grid.lat_z_plot, np.transpose(np.mean(u,axis=1))[:view_config.top,:], colors='white',levels=20,linewidths=1,alpha=0.8)
				ax[1].quiver(coordinate_grid.heights_plot, coordinate_grid.lat_z_plot, np.transpose(np.mean(v,axis=1))[:view_config.top,:],np.transpose(np.mean(5*w,axis=1))[:view_config.top,:],color='black')
			ax[1].set_title('$\it{Atmospheric} \quad \it{temperature}$')
			ax[1].set_xlim((-90,90))
			ax[1].set_ylim((planet_config.pressure_levels.max()/100,planet_config.pressure_levels[:view_config.top].min()/100))
			ax[1].set_ylabel('Pressure (hPa)')
			ax[1].set_xlabel('Latitude')
			ax[1].set_yscale('log')
			f.colorbar(test, cax=cbar_ax)
			cbar_ax.set_title('Temperature (K)')
			f.suptitle( 'Time ' + str(round(t/planet_config.day,2)) + ' days' )
				
		else:
			ax[0,0].contourf(coordinate_grid.heights_plot, coordinate_grid.lat_z_plot, np.transpose(np.mean(u,axis=1))[:view_config.top,:], cmap='seismic')
			ax[0,0].set_title('u')
			ax[0,1].contourf(coordinate_grid.heights_plot, coordinate_grid.lat_z_plot, np.transpose(np.mean(v,axis=1))[:view_config.top,:], cmap='seismic')
			ax[0,1].set_title('v')
			ax[1,0].contourf(coordinate_grid.heights_plot, coordinate_grid.lat_z_plot, np.transpose(np.mean(w,axis=1))[:view_config.top,:], cmap='seismic')
			ax[1,0].set_title('w')
			ax[1,1].contourf(coordinate_grid.heights_plot, coordinate_grid.lat_z_plot, np.transpose(np.mean(atmosp_addition,axis=1))[:view_config.top,:], cmap='seismic')
			ax[1,1].set_title('atmosp_addition')
			for axis in ax.ravel():
				axis.set_ylim((planet_config.pressure_levels.max()/100,planet_config.pressure_levels[:view_config.top].min()/100))
				axis.set_yscale('log')
			f.suptitle( 'Time ' + str(round(t/planet_config.day,2)) + ' days' )

		if view_config.level_plots:
			for k, z in zip(range(view_config.nplots), level_plots_levels):	
				z += 1
				bx[k].contourf(coordinate_grid.lon_plot, coordinate_grid.lat_plot, potential_temperature[:,:,z], cmap='seismic',levels=15)
				bx[k].quiver(coordinate_grid.lon_plot[::quiver_padding,::quiver_padding], coordinate_grid.lat_plot[::quiver_padding,::quiver_padding], u[::quiver_padding,::quiver_padding,z], v[::quiver_padding,::quiver_padding,z], color='white')
				bx[k].set_title(str(round(planet_config.pressure_levels[z]/100))+' hPa')
				bx[k].set_ylabel('Latitude')
				bx[k].set_xlim((coordinate_grid.lon.min(),coordinate_grid.lon.max()))
				bx[k].set_ylim((coordinate_grid.lat.min(),coordinate_grid.lat.max()))				
			bx[-1].set_xlabel('Longitude')

	if view_config.above and velocity:
		gx[0].set_title('Original data')
		gx[1].set_title('Polar plane')
		gx[2].set_title('Reprojected data')
		g.suptitle( 'Time ' + str(round(t/planet_config.day,2)) + ' days' )

		if view_config.pole == PoleType.SOUTH:
			gx[0].set_title('temperature')
			gx[0].contourf(coordinate_grid.lon,coordinate_grid.lat[:pole_low_index_S],potential_temperature[:pole_low_index_S,:,view_config.view_config.above_level])
			
			gx[1].set_title('polar_plane_advect')
			polar_temps = low_level.beam_me_up(coordinate_grid.lat[:pole_low_index_S],coordinate_grid.lon,potential_temperature[:pole_low_index_S,:,:],grids[1],grid_lat_coords_S,grid_lon_coords_S)
			output = low_level.beam_me_up(coordinate_grid.lat[:pole_low_index_S],coordinate_grid.lon,south_reprojected_addition,grids[1],grid_lat_coords_S,grid_lon_coords_S)

			gx[1].contourf(grid_x_values_S/1E3,grid_y_values_S/1E3,output[:,:,view_config.view_config.above_level])
			gx[1].contour(grid_x_values_S/1E3,grid_y_values_S/1E3,polar_temps[:,:,view_config.view_config.above_level],colors='white',levels=20,linewidths=1,alpha=0.8)
			gx[1].quiver(grid_x_values_S/1E3,grid_y_values_S/1E3,x_dot_S[:,:,view_config.view_config.above_level],y_dot_S[:,:,view_config.above_level])
			
			gx[1].add_patch(plt.Circle((0,0),planet_config.planet_radius*np.cos(coordinate_grid.lat[pole_low_index_S]*np.pi/180.0)/1E3,color='r',fill=False))
			gx[1].add_patch(plt.Circle((0,0),planet_config.planet_radius*np.cos(coordinate_grid.lat[pole_high_index_S]*np.pi/180.0)/1E3,color='r',fill=False))

			gx[2].set_title('south_addition_smoothed')
			gx[2].contourf(coordinate_grid.lon,coordinate_grid.lat[:pole_low_index_S],south_addition_smoothed[:pole_low_index_S,:,view_config.above_level])
			# gx[2].contourf(coordinate_grid.lon,coordinate_grid.lat[:pole_low_index_S],u[:pole_low_index_S,:,view_config.above_level])
			gx[2].quiver(coordinate_grid.lon[::5],coordinate_grid.lat[:pole_low_index_S],u[:pole_low_index_S,::5,view_config.above_level],v[:pole_low_index_S,::5,view_config.above_level])
		else:
			gx[0].set_title('temperature')
			gx[0].contourf(coordinate_grid.lon,coordinate_grid.lat[pole_low_index_N:],potential_temperature[pole_low_index_N:,:,view_config.above_level])
			
			gx[1].set_title('polar_plane_advect')
			polar_temps = low_level.beam_me_up(coordinate_grid.lat[pole_low_index_N:],coordinate_grid.lon,np.flip(potential_temperature[pole_low_index_N:,:,:],axis=1),grids[0],grid_lat_coords_N,grid_lon_coords_N)
			output = low_level.beam_me_up(coordinate_grid.lat[pole_low_index_N:],coordinate_grid.lon,north_reprojected_addition,grids[0],grid_lat_coords_N,grid_lon_coords_N)
			gx[1].contourf(grid_x_values_N/1E3,grid_y_values_N/1E3,output[:,:,view_config.above_level])
			gx[1].contour(grid_x_values_N/1E3,grid_y_values_N/1E3,polar_temps[:,:,view_config.above_level],colors='white',levels=20,linewidths=1,alpha=0.8)
			gx[1].quiver(grid_x_values_N/1E3,grid_y_values_N/1E3,x_dot_N[:,:,view_config.above_level],y_dot_N[:,:,view_config.above_level])
			
			gx[1].add_patch(plt.Circle((0,0),planet_config.planet_radius*np.cos(coordinate_grid.lat[pole_low_index_N]*np.pi/180.0)/1E3,color='r',fill=False))
			gx[1].add_patch(plt.Circle((0,0),planet_config.planet_radius*np.cos(coordinate_grid.lat[pole_high_index_N]*np.pi/180.0)/1E3,color='r',fill=False))
	
			gx[2].set_title('south_addition_smoothed')
			# gx[2].contourf(coordinate_grid.lon,coordinate_grid.lat[pole_low_index_N:],north_addition_smoothed[:,:,view_config.above_level])
			gx[2].contourf(coordinate_grid.lon,coordinate_grid.lat[pole_low_index_N:],u[pole_low_index_N:,:,view_config.above_level])
			gx[2].quiver(coordinate_grid.lon[::5],coordinate_grid.lat[pole_low_index_N:],u[pole_low_index_N:,::5,view_config.above_level],v[pole_low_index_N:,::5,view_config.above_level])
		
	# clear plots
	if view_config.plot or view_config.above:	plt.pause(0.001)
	if view_config.plot:
		if not view_config.diagnostic:
			ax[0].cla()
			ax[1].cla()
			cbar_ax.cla()
					
		else:
			ax[0,0].cla()
			ax[0,1].cla()
			ax[1,0].cla()
			ax[1,1].cla()
		if view_config.level_plots:
			for k in range(view_config.nplots):
				bx[k].cla()	
		if view_config.verbose:		
			time_taken = float(round(time.time() - before_plot,3))
			print('Plotting: ',str(time_taken),'s')	
	if view_config.above:
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
	print("+++ t = " + str(round(t/planet_config.day,2)) + " days +++")
	print('T: ',round(coordinate_grid.temperature_world.max()-273.15,1),' - ',round(coordinate_grid.temperature_world.min()-273.15,1),' C')
	print('U: ',round(u.max(),2),' - ',round(u.min(),2),' V: ',round(v.max(),2),' - ',round(v.min(),2),' W: ',round(w.max(),2),' - ',round(w.min(),4))

	tracer[40,50,sample_level] = 1
	tracer[20,50,sample_level] = 1

	if view_config.verbose: before_radiation = time.time()
	coordinate_grid.temperature_world, potential_temperature = top_level.radiation_calculation(coordinate_grid.temperature_world, potential_temperature, planet_config.pressure_levels, heat_capacity_earth, albedo, planet_config.insolation, coordinate_grid.lat, coordinate_grid.lon, t, dt, planet_config.day, planet_config.year, planet_config.axial_tilt)
	if smoothing_config.smoothing: potential_temperature = top_level.smoothing_3D(potential_temperature,smoothing_config.smoothing_parameter_t)
	if view_config.verbose:
		time_taken = float(round(time.time() - before_radiation,3))
		print('Radiation: ',str(time_taken),'s')

	diffusion = top_level.laplacian_2d(coordinate_grid.temperature_world,dx,dy)
	diffusion[0,:] = np.mean(diffusion[1,:],axis=0)
	diffusion[-1,:] = np.mean(diffusion[-2,:],axis=0)
	coordinate_grid.temperature_world -= dt*1E-5*diffusion

	# update geopotential field
	geopotential = np.zeros_like(potential_temperature)
	for k in np.arange(1,planet_config.nlevels):	geopotential[:,:,k] = geopotential[:,:,k-1] - potential_temperature[:,:,k]*(sigma[k]-sigma[k-1])

	if velocity:

		if view_config.verbose:	before_velocity = time.time()
		
		u_add,v_add = top_level.velocity_calculation(u,v,w,planet_config.pressure_levels,geopotential,potential_temperature,coriolis,planet_config.gravity,dx,dy,dt)

		if view_config.verbose:	
			time_taken = float(round(time.time() - before_velocity,3))
			print('Velocity: ',str(time_taken),'s')

		if view_config.verbose:	before_projection = time.time()
		
		grid_velocities = (x_dot_N,y_dot_N,x_dot_S,y_dot_S)
	
		u_add,v_add,north_reprojected_addition,south_reprojected_addition,x_dot_N,y_dot_N,x_dot_S,y_dot_S = top_level.polar_planes(u,v,u_add,v_add,potential_temperature,geopotential,grid_velocities,indices,grids,coords,coriolis_plane_N,coriolis_plane_S,grid_side_length,planet_config.pressure_levels,coordinate_grid.lat,coordinate_grid.lon,dt,polar_grid_resolution,planet_config.gravity)
		
		u += u_add
		v += v_add

		if smoothing_config.smoothing: u = top_level.smoothing_3D(u,smoothing_config.smoothing_parameter_u)
		if smoothing_config.smoothing: v = top_level.smoothing_3D(v,smoothing_config.smoothing_parameter_v)

		x_dot_N,y_dot_N,x_dot_S,y_dot_S = top_level.update_plane_velocities(coordinate_grid.lat,coordinate_grid.lon,pole_low_index_N,pole_low_index_S,np.flip(u[pole_low_index_N:,:,:],axis=1),np.flip(v[pole_low_index_N:,:,:],axis=1),grids,grid_lat_coords_N,grid_lon_coords_N,u[:pole_low_index_S,:,:],v[:pole_low_index_S,:,:],grid_lat_coords_S,grid_lon_coords_S)
		
		if view_config.verbose:	
			time_taken = float(round(time.time() - before_projection,3))
			print('Projection: ',str(time_taken),'s')

		### allow for thermal advection in the atmosphere
		if view_config.verbose:	before_advection = time.time()



		if view_config.verbose: before_w = time.time()
		# using updated u,v fields calculated w
		# https://www.sjsu.edu/faculty/watkins/omega.htm
		w = top_level.w_calculation(u,v,w,planet_config.pressure_levels,geopotential,potential_temperature,coriolis,planet_config.gravity,dx,dy,dt)
		if smoothing_config.smoothing: w = top_level.smoothing_3D(w,smoothing_config.smoothing_parameter_w,0.25)

		theta_N = low_level.beam_me_up(coordinate_grid.lat[pole_low_index_N:],coordinate_grid.lon,potential_temperature[pole_low_index_N:,:,:],grids[0],grid_lat_coords_N,grid_lon_coords_N)
		w_N = top_level.w_plane(x_dot_N,y_dot_N,theta_N,planet_config.pressure_levels,polar_grid_resolution,planet_config.gravity)
		w_N = np.flip(low_level.beam_me_down(coordinate_grid.lon,w_N,pole_low_index_N, grid_x_values_N, grid_y_values_N,polar_x_coords_N, polar_y_coords_N),axis=1)
		w[pole_low_index_N:,:,:] = low_level.combine_data(pole_low_index_N,pole_high_index_N,w[pole_low_index_N:,:,:],w_N,coordinate_grid.lat)
		
		w_S = top_level.w_plane(x_dot_S,y_dot_S,low_level.beam_me_up(coordinate_grid.lat[:pole_low_index_S],coordinate_grid.lon,potential_temperature[:pole_low_index_S,:,:],grids[1],grid_lat_coords_S,grid_lon_coords_S),planet_config.pressure_levels,polar_grid_resolution,planet_config.gravity)
		w_S = low_level.beam_me_down(coordinate_grid.lon,w_S,pole_low_index_S, grid_x_values_S, grid_y_values_S,polar_x_coords_S, polar_y_coords_S)
		w[:pole_low_index_S,:,:] = low_level.combine_data(pole_low_index_S,pole_high_index_S,w[:pole_low_index_S,:,:],w_S,coordinate_grid.lat)

		# for k in np.arange(1,planet_config.nlevels-1):
		# 	north_reprojected_addition[:,:,k] += 0.5*(w_N[:,:,k] + abs(w_N[:,:,k]))*(potential_temperature[pole_low_index_N:,:,k] - potential_temperature[pole_low_index_N:,:,k-1])/(planet_config.pressure_levels[k] - planet_config.pressure_levels[k-1])
		# 	north_reprojected_addition[:,:,k] += 0.5*(w_N[:,:,k] - abs(w_N[:,:,k]))*(potential_temperature[pole_low_index_N:,:,k+1] - potential_temperature[pole_low_index_N:,:,k])/(planet_config.pressure_levels[k+1] - planet_config.pressure_levels[k])

		# 	south_reprojected_addition[:,:,k] += 0.5*(w_S[:,:,k] + abs(w_S[:,:,k]))*(potential_temperature[:pole_low_index_S,:,k] - potential_temperature[:pole_low_index_S,:,k-1])/(planet_config.pressure_levels[k] - planet_config.pressure_levels[k-1])
		# 	south_reprojected_addition[:,:,k] += 0.5*(w_S[:,:,k] - abs(w_S[:,:,k]))*(potential_temperature[:pole_low_index_S,:,k+1] - potential_temperature[:pole_low_index_S,:,k])/(planet_config.pressure_levels[k+1] - planet_config.pressure_levels[k])

		w[:,:,18:] *= 0

		if view_config.verbose:	
			time_taken = float(round(time.time() - before_w,3))
			print('Calculate w: ',str(time_taken),'s')

		#################################

		atmosp_addition = top_level.divergence_with_scalar(potential_temperature,u,v,w,dx,dy,planet_config.pressure_levels)

		# combine addition calculated on polar grid with that calculated on the cartestian grid
		north_addition_smoothed = low_level.combine_data(pole_low_index_N,pole_high_index_N,atmosp_addition[pole_low_index_N:,:,:],north_reprojected_addition,coordinate_grid.lat)
		south_addition_smoothed = low_level.combine_data(pole_low_index_S,pole_high_index_S,atmosp_addition[:pole_low_index_S,:,:],south_reprojected_addition,coordinate_grid.lat)
		
		# add the blended/combined addition to global temperature addition array
		atmosp_addition[:pole_low_index_S,:,:] = south_addition_smoothed
		atmosp_addition[pole_low_index_N:,:,:] = north_addition_smoothed

		if smoothing_config.smoothing: atmosp_addition = top_level.smoothing_3D(atmosp_addition,smoothing_config.smoothing_parameter_add)

		atmosp_addition[:,:,17] *= 0.5
		atmosp_addition[:,:,18:] *= 0

		potential_temperature -= dt*atmosp_addition

		###################################################################

		tracer_addition = top_level.divergence_with_scalar(tracer,u,v,w,dx,dy,planet_config.pressure_levels)
		tracer_addition[:4,:,:] *= 0
		tracer_addition[-4:,:,:] *= 0

		for k in np.arange(1,planet_config.nlevels-1):

			tracer_addition[:,:,k] += 0.5*(w[:,:,k] - abs(w[:,:,k]))*(tracer[:,:,k] - tracer[:,:,k-1])/(planet_config.pressure_levels[k] - planet_config.pressure_levels[k-1])
			tracer_addition[:,:,k] += 0.5*(w[:,:,k] + abs(w[:,:,k]))*(tracer[:,:,k+1] - tracer[:,:,k])/(planet_config.pressure_levels[k] - planet_config.pressure_levels[k-1])

		tracer -= dt*tracer_addition

		diffusion = top_level.laplacian_3d(potential_temperature,dx,dy,planet_config.pressure_levels)
		diffusion[0,:,:] = np.mean(diffusion[1,:,:],axis=0)
		diffusion[-1,:,:] = np.mean(diffusion[-2,:,:],axis=0)
		potential_temperature -= dt*1E-4*diffusion

		###################################################################

		if view_config.verbose:	
			time_taken = float(round(time.time() - before_advection,3))
			print('Advection: ',str(time_taken),'s')

	if t-last_plot >= save_config.plot_frequency*dt:
		plotting_routine()
		last_plot = t

	if save_config.save:
		if t-last_save >= save_config.save_frequency*dt:
			pickle.dump((potential_temperature,coordinate_grid.temperature_world,u,v,w,x_dot_N,y_dot_N,x_dot_S,y_dot_S,t,albedo,tracer), open("save_file.p","wb"))
			last_save = t

	if np.isnan(u.max()):
		sys.exit()

	# advance time by one timestep
	t += dt

	time_taken = float(round(time.time() - initial_time,3))

	print('Time: ',str(time_taken),'s')
	# print('777777777777777777')
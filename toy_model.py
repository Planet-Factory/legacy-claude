# toy model for use on stream
# Please give me your Twitch prime sub!

# CLimate Analysis using Digital Estimations (CLAuDE)

import numpy as np 
import matplotlib.pyplot as plt
import time, sys, pickle

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

###

advection = True 				# if you want to include advection set this to be True
advection_boundary = 4			# how many gridpoints away from poles to apply advection

save = True
load = False

plot = False
level_plots = False

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
temperature_planet = np.zeros((nlat,nlon)) + 290
temperature_atmosp = np.zeros((nlat,nlon,nlevels))
air_pressure = np.zeros_like(temperature_atmosp)
u = np.zeros_like(temperature_atmosp)
v = np.zeros_like(temperature_atmosp)
w = np.zeros_like(temperature_atmosp)
air_density = np.zeros_like(temperature_atmosp)

# #######################

def profile(a):
	return np.mean(np.mean(a,axis=0),axis=0)

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
	temperature_atmosp[:,:,k] = temp_profile[k]

###########################

# weight_above = np.interp(x=heights/1E3,xp=standard_height,fp=standard_density)
# top_index = np.argmax(np.array(standard_height) >= top/1E3)
# if standard_height[top_index] == top/1E3:
# 	weight_above = np.trapz(np.interp(x=standard_height[top_index:],xp=standard_height,fp=standard_density),standard_height[top_index:])*gravity*1E3
# else:
# 	weight_above = np.trapz(np.interp(x=np.insert(standard_height[top_index:], 0, top/1E3),xp=standard_height,fp=standard_density),np.insert(standard_height[top_index:], 0, top/1E3))*gravity*1E3

###########################

albedo = np.zeros_like(temperature_planet) + 0.2
heat_capacity_earth = np.zeros_like(temperature_planet) + 1E6

albedo_variance = 0.001
for i in range(nlat):
	for j in range(nlon):
		albedo[i,j] += np.random.uniform(-albedo_variance,albedo_variance)

specific_gas = 287
thermal_diffusivity_roc = 1.5E-6
sigma = 5.67E-8

air_pressure = air_density*specific_gas*temperature_atmosp

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

###################### FUNCTIONS ######################

# define various useful differential functions:
# gradient of scalar field a in the local x direction at point i,j
def scalar_gradient_x(a,i,j,k=999):
	if k == 999:
		return (a[i,(j+1)%nlon]-a[i,(j-1)%nlon])/dx[i]
	else:
		return (a[i,(j+1)%nlon,k]-a[i,(j-1)%nlon,k])/dx[i]

# gradient of scalar field a in the local y direction at point i,j
def scalar_gradient_y(a,i,j,k=999):
	if k == 999:
		if i == 0:
			return 2*(a[i+1,j]-a[i,j])/dy
		elif i == nlat-1:
			return 2*(a[i,j]-a[i-1,j])/dy
		else:
			return (a[i+1,j]-a[i-1,j])/dy
	else:
		if i == 0:
			return 2*(a[i+1,j,k]-a[i,j,k])/dy
		elif i == nlat-1:
			return 2*(a[i,j,k]-a[i-1,j,k])/dy
		else:
			return (a[i+1,j,k]-a[i-1,j,k])/dy

def scalar_gradient_z(a,i,j,k):
	output = np.zeros_like(a)
	if output.ndim == 1:
		if k == 0:
			return (a[k+1]-a[k])/dz[k]
		elif k == nlevels-1:
			return (a[k]-a[k-1])/dz[k]
		else:
			return (a[k+1]-a[k-1])/(2*dz[k])
	else:
		if k == 0:
			return (a[i,j,k+1]-a[i,j,k])/dz[k]
		elif k == nlevels-1:
			return (a[i,j,k]-a[i,j,k-1])/dz[k]
		else:
			return (a[i,j,k+1]-a[i,j,k-1])/(2*dz[k])

# laplacian of scalar field a
def laplacian(a):
	output = np.zeros_like(a)
	if output.ndim == 2:
		for i in np.arange(1,nlat-1):
			for j in range(nlon):
				output[i,j] = (scalar_gradient_x(a,i,(j+1)%nlon) - scalar_gradient_x(a,i,(j-1)%nlon))/dx[i] + (scalar_gradient_y(a,i+1,j) - scalar_gradient_y(a,i-1,j))/dy
		return output
	if output.ndim == 3:
		for i in np.arange(1,nlat-1):
			for j in range(nlon):
				for k in range(nlevels-1):
					output[i,j,k] = (scalar_gradient_x(a,i,(j+1)%nlon,k) - scalar_gradient_x(a,i,(j-1)%nlon,k))/dx[i] + (scalar_gradient_y(a,i+1,j,k) - scalar_gradient_y(a,i-1,j,k))/dy + (scalar_gradient_z(a,i,j,k+1)-scalar_gradient_z(a,i,j,k-1))/(2*dz[k])
		return output

# divergence of (a*u) where a is a scalar field and u is the atmospheric velocity field
def divergence_with_scalar(a):
	output = np.zeros_like(a)
	for i in range(nlat):
		for j in range(nlon):
			for k in range(nlevels):
				output[i,j] = scalar_gradient_x(a*u,i,j,k) + scalar_gradient_y(a*v,i,j,k) #+ 0.1*scalar_gradient_z(a*w,i,j,k)
	return output	

# power incident on (lat,lon) at time t
def solar(insolation, lat, lon, t):
	sun_longitude = -t % day
	sun_longitude *= 360/day
	sun_latitude = axial_tilt*np.cos(t*2*np.pi/year)

	value = insolation*np.cos((lat-sun_latitude)*np.pi/180)

	if value < 0:	
		return 0
	else:

		lon_diff = lon-sun_longitude
		value *= np.cos(lon_diff*np.pi/180)
		
		if value < 0:
			if lat + sun_latitude > 90:
				return insolation*np.cos((lat+sun_latitude)*np.pi/180)*np.cos(lon_diff*np.pi/180)
			elif lat + sun_latitude < -90:
				return insolation*np.cos((lat+sun_latitude)*np.pi/180)*np.cos(lon_diff*np.pi/180)
			else:
				return 0
		else:
			return value

def surface_optical_depth(lat):
	return 4 + np.cos(lat*np.pi/90)*2.5/2

def smooth(a,window):
	output = np.zeros_like(a)
	diff = int(np.floor(window/2))
	for i in range(len(output)):
		if i-diff < 0:
			output[i] = np.mean(a[i:i+diff+1])
		elif i+diff+1 > len(output):
			output[i] = np.mean(a[i-diff:i])
		else:
			output[i] = np.mean(a[i-diff:i+diff+1])
	return output

def thermal_radiation(a):
	return sigma*(a**4)

#################### SHOW TIME ####################

if plot:
	# set up plot
	f, ax = plt.subplots(2,figsize=(9,9))
	f.canvas.set_window_title('CLAuDE')
	ax[0].contourf(lon_plot, lat_plot, temperature_planet, cmap='seismic')
	ax[1].contourf(lon_plot, lat_plot, temperature_atmosp[:,:,0], cmap='seismic')
	plt.subplots_adjust(left=0.1, right=0.75)
	ax[0].set_title('Ground temperature')
	ax[1].set_title('Atmosphere temperature')
	# allow for live updating as calculations take place

	if level_plots:
		g, bx = plt.subplots(nlevels,figsize=(9,7),sharex=True)
		g.canvas.set_window_title('CLAuDE atmospheric levels')
		for k in range(nlevels):
			bx[k].contourf(lon_plot, lat_plot, temperature_atmosp[:,:,k], cmap='seismic')
			bx[k].set_title(str(heights[k])+' km')
			bx[k].set_ylabel('Latitude')
		bx[-1].set_xlabel('Longitude')

	plt.ion()
	plt.show()

# INITIATE TIME
t = 0

if load:
	# load in previous save file
	temperature_atmosp,temperature_planet,u,v,w,t,air_density,albedo = pickle.load(open("save_file.p","rb"))

while True:

	initial_time = time.time()

	if t < 30*day:
		dt = 60*137
		velocity = False
	else:
		dt = 60*9
		velocity = True

	# print current time in simulation to command line
	print("+++ t = " + str(round(t/day,2)) + " days +++", end='\r')
	print('T: ',round(temperature_planet.max()-273.15,1),' - ',round(temperature_planet.min()-273.15,1),' C')
	print('U: ',round(u.max(),2),' - ',round(u.min(),2),' V: ',round(v.max(),2),' - ',round(v.min(),2),' W: ',round(w.max(),2),' - ',round(w.min(),2))
	# print(profile(air_density))
	# print(profile(air_pressure)/100)

	# calculate change in temperature of ground and atmosphere due to radiative imbalance
	for i in range(nlat):
		for j in range(nlon):

			upward_radiation = np.zeros(nlevels)
			downward_radiation = np.zeros(nlevels)
			optical_depth = np.zeros(nlevels)
			Q = np.zeros(nlevels)

			# calculate optical depth
			pressure_profile = air_pressure[i,j,:]
			density_profile = air_density[i,j,:]
			fl = 0.1
			optical_depth = surface_optical_depth(lat[i])*(fl*(pressure_profile/pressure_profile[0]) + (1-fl)*(pressure_profile/pressure_profile[0])**4)
			
			# calculate upward longwave flux, bc is thermal radiation at surface
			upward_radiation[0] = thermal_radiation(temperature_planet[i,j])
			for k in np.arange(1,nlevels):
				upward_radiation[k] = (upward_radiation[k-1] - (optical_depth[k]-optical_depth[k-1])*(thermal_radiation(temperature_atmosp[i,j,k])))/(1+optical_depth[k-1]-optical_depth[k])

			# calculate downward longwave flux, bc is zero at TOA (in model)
			downward_radiation[-1] = 0
			for k in np.arange(0,nlevels-1)[::-1]:
				downward_radiation[k] = (downward_radiation[k+1] - thermal_radiation(temperature_atmosp[i,j,k])*(optical_depth[k+1]-optical_depth[k]))/(1 + optical_depth[k] - optical_depth[k+1])
			
			# gradient of difference provides heating at each level
			for k in np.arange(nlevels):
				Q[k] = -scalar_gradient_z(upward_radiation-downward_radiation,0,0,k)/(1E3*density_profile[k])
				# make sure model does not have a higher top than 50km!!
				# approximate SW heating of ozone
				if heights[k] > 20E3:
					Q[k] += solar(5,lat[i],lon[j],t)*((((heights[k]-20E3)/1E3)**2)/(30**2))/(24*60*60)

			temperature_atmosp[i,j,:] += Q*dt

			# update surface temperature with shortwave radiation flux
			temperature_planet[i,j] += dt*((1-albedo[i,j])*(solar(insolation,lat[i],lon[j],t) + downward_radiation[0]) - upward_radiation[0])/heat_capacity_earth[i,j] 

	# update air pressure
	old_pressure = np.copy(air_pressure)
	air_pressure = air_density*specific_gas*temperature_atmosp

	if velocity:

		# introduce temporary arrays to update velocity in the atmosphere
		u_temp = np.zeros_like(u)
		v_temp = np.zeros_like(v)
		w_temp = np.zeros_like(w)

		# calculate acceleration of atmosphere using primitive equations on beta-plane
		for i in np.arange(1,nlat-1):
			for j in range(nlon):
				for k in range(nlevels):
					u_temp[i,j,k] += dt*( -u[i,j,k]*scalar_gradient_x(u,i,j,k) - v[i,j,k]*scalar_gradient_y(u,i,j,k) + coriolis[i]*v[i,j,k] - scalar_gradient_x(air_pressure,i,j,k)/air_density[i,j,k] )
					v_temp[i,j,k] += dt*( -u[i,j,k]*scalar_gradient_x(v,i,j,k) - v[i,j,k]*scalar_gradient_y(v,i,j,k) - coriolis[i]*u[i,j,k] - scalar_gradient_y(air_pressure,i,j,k)/air_density[i,j,k] )
					w_temp[i,j,k] += -(air_pressure[i,j,k]-old_pressure[i,j,k])/(dt*air_density[i,j,k]*gravity)

				# plt.plot(w_temp[i,j,:],heights)
				# plt.title('Vertical acceleration')
				# plt.show()
				# sys.exit()


		u += u_temp
		v += v_temp
		w += w_temp

		# approximate friction
		u *= 0.95
		v *= 0.95

		if advection:
			# allow for thermal advection in the atmosphere, and heat diffusion in the atmosphere and the ground
			# atmosp_addition = dt*(thermal_diffusivity_air*laplacian(temperature_atmosp))
			atmosp_addition = dt*divergence_with_scalar(temperature_atmosp)
			temperature_atmosp[advection_boundary:-advection_boundary,:,:] -= atmosp_addition[advection_boundary:-advection_boundary,:,:]
			temperature_atmosp[advection_boundary-1,:,:] -= 0.5*atmosp_addition[advection_boundary-1,:,:]
			temperature_atmosp[-advection_boundary,:,:] -= 0.5*atmosp_addition[-advection_boundary,:,:]

			# as density is now variable, allow for mass advection
			density_addition = dt*divergence_with_scalar(air_density)
			air_density[advection_boundary:-advection_boundary,:,:] -= density_addition[advection_boundary:-advection_boundary,:,:]
			air_density[(advection_boundary-1),:,:] -= 0.5*density_addition[advection_boundary-1,:,:]
			air_density[-advection_boundary,:,:] -= 0.5*density_addition[-advection_boundary,:,:]

			temperature_planet += dt*(thermal_diffusivity_roc*laplacian(temperature_planet))
	
	if plot:
		# update plot
		test = ax[0].contourf(lon_plot, lat_plot, temperature_planet, cmap='seismic')
		ax[0].streamplot(lon_plot, lat_plot, u[:,:,0], v[:,:,0], color='white',density=1)
		ax[0].set_title('$\it{Ground} \quad \it{temperature}$')
		ax[0].set_xlim((lon.min(),lon.max()))
		ax[0].set_ylim((lat.min(),lat.max()))
		ax[0].set_ylabel('Latitude')
		ax[0].axhline(y=0,color='black',alpha=0.3)
		ax[0].set_xlabel('Longitude')

		ax[1].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(temperature_atmosp,axis=1)), cmap='seismic')
		ax[1].contour(heights_plot,lat_z_plot, np.transpose(np.mean(u,axis=1)), colors='white',levels=20,linewidths=1,alpha=0.8)
		ax[1].streamplot(heights_plot, lat_z_plot, np.transpose(np.mean(v,axis=1)),np.transpose(np.mean(10*w,axis=1)),color='black',density=0.75)
		ax[1].set_title('$\it{Atmospheric} \quad \it{temperature}$')
		ax[1].set_xlim((-90,90))
		ax[1].set_ylim((0,heights.max()))
		ax[1].set_ylabel('Height (m)')
		ax[1].set_xlabel('Latitude')

		cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
		f.colorbar(test, cax=cbar_ax)
		cbar_ax.set_title('Temperature (K)')
		f.suptitle( 'Time ' + str(round(24*t/day,2)) + ' hours' )
		
		if level_plots:
			for k in range(nlevels):
				index = nlevels-1-k
				test = bx[index].contourf(lon_plot, lat_plot, temperature_atmosp[:,:,k], cmap='seismic')
				bx[index].streamplot(lon_plot, lat_plot, u[:,:,k], v[:,:,k],density=0.5,color='black')
				# g.colorbar(test,cax=bx[nlevels-1-k])
				bx[index].set_title(str(heights[k]/1E3)+' km')
				bx[index].set_ylabel('Latitude')
				bx[index].set_xlim((lon.min(),lon.max()))
				bx[index].set_ylim((lat.min(),lat.max()))
			bx[-1].set_xlabel('Longitude')

		plt.pause(0.01)
		
		ax[0].cla()
		ax[1].cla()
		if level_plots:
			for k in range(nlevels):
				bx[k].cla()

	# advance time by one timestep
	t += dt

	time_taken = float(round(time.time() - initial_time,3))

	print('Time: ',str(time_taken),'s')

	if save:
		pickle.dump((temperature_atmosp,temperature_planet,u,v,w,t,air_density,albedo), open("save_file.p","wb"))

	# sys.exit()
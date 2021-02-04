# claude low level library

import numpy as np
cimport numpy as np
cimport cython
from scipy.interpolate import interp2d, RectBivariateSpline

ctypedef np.float64_t DTYPE_f
cdef float inv_180 = np.pi/180
cdef float inv_90 = np.pi/90
cdef DTYPE_f sigma = 5.67E-8

# define various useful differential functions:
# gradient of scalar field a in the local x direction at point i,j
cpdef scalar_gradient_x(np.ndarray a, np.ndarray dx, np.int_t nlon, np.int_t i, np.int_t j, np.int_t k):
	return (a[i,(j+1)%nlon,k]-a[i,(j-1)%nlon,k])/(dx[i])

cpdef scalar_gradient_x_matrix(np.ndarray a,np.ndarray dx):
	cdef np.ndarray output = (np.roll(a, -1, axis=1) - np.roll(a, 1, axis=1)) / dx[:, None, None]
	return output

cpdef scalar_gradient_x_matrix_primitive(np.ndarray a,np.ndarray dx):
	cdef np.ndarray output = np.zeros_like(a)
	cdef np.int_t i,j,k
	for i in range(a.shape[0]):
		for j in range(a.shape[1]):
			for k in range(a.shape[2]):
				output[i,j,k] = scalar_gradient_x(a,dx,a.shape[1],i,j,k)

	output[0,:,:] *= 0
	output[-1,:,:] *= 0
	return output

cpdef scalar_gradient_x_2D(np.ndarray a,np.ndarray dx,np.int_t nlon,np.int_t i,np.int_t j):
	return (a[i,(j+1)%nlon]-a[i,(j-1)%nlon])/dx[i]

# gradient of scalar field a in the local y direction at point i,j
cpdef scalar_gradient_y(np.ndarray a,DTYPE_f dy,np.int_t nlat,np.int_t i,np.int_t j,np.int_t k):
	if i == 0:
		return 2*(a[i+1,j,k]-a[i,j,k])/dy
	elif i == nlat-1:
		return 2*(a[i,j,k]-a[i-1,j,k])/dy
	else:
		return (a[i+1,j,k]-a[i-1,j,k])/dy
	
cpdef scalar_gradient_y_matrix(np.ndarray a,DTYPE_f dy):
	shift_south = np.pad(a, ((1,0), (0,0), (0,0)), 'reflect', reflect_type='odd')[:-1,:,:]
	shift_north = np.pad(a, ((0,1), (0,0), (0,0)), 'reflect', reflect_type='odd')[1:,:,:]
	return (shift_north - shift_south)/dy

cpdef scalar_gradient_y_matrix_primitive(np.ndarray a,DTYPE_f dy):
	cdef np.ndarray output = np.zeros_like(a)
	cdef np.int_t i,j,k
	for i in range(a.shape[0]):
		for j in range(a.shape[1]):
			for k in range(a.shape[2]):
				output[i,j,k] = scalar_gradient_y(a,dy,a.shape[0],i,j,k)
	return output

cpdef scalar_gradient_y_2D(np.ndarray a,DTYPE_f dy,np.int_t nlat,np.int_t i,np.int_t j):
	if i == 0:
		return 2*(a[i+1,j]-a[i,j])/dy
	elif i == nlat-1:
		return 2*(a[i,j]-a[i-1,j])/dy
	else:
		return (a[i+1,j]-a[i-1,j])/dy

cpdef scalar_gradient_z_1D(np.ndarray a,np.ndarray pressure_levels,np.int_t k):
	cdef np.int_t nlevels = len(pressure_levels)
	if k == 0:
		return -(a[k+1]-a[k])/(pressure_levels[k+1]-pressure_levels[k])
	elif k == nlevels-1:
		return -(a[k]-a[k-1])/(pressure_levels[k]-pressure_levels[k-1])
	else:
		return -(a[k+1]-a[k-1])/(pressure_levels[k+1]-pressure_levels[k-1])

cpdef scalar_gradient_z_3D(np.ndarray a,np.ndarray pressure_levels,np.int_t k):
	cdef np.int_t nlevels = len(pressure_levels)
	if k == 0:
		return -(a[:,:,k+1]-a[:,:,k])/(pressure_levels[k+1]-pressure_levels[k])
	elif k == nlevels-1:
		return -(a[:,:,k]-a[:,:,k-1])/(pressure_levels[k]-pressure_levels[k-1])
	else:
		return -(a[:,:,k+1]-a[:,:,k-1])/(pressure_levels[k+1]-pressure_levels[k-1])

cpdef scalar_gradient_z_matrix(np.ndarray a, np.ndarray pressure_levels):
	shift_up = np.pad(a, ((0,0), (0,0), (1,0)), 'edge')[:,:,:-1]
	shift_down = np.pad(a, ((0,0), (0,0), (0,1)), 'edge')[:,:,1:]
	shift_pressure_up = np.pad(pressure_levels, (1,0), 'edge')[:-1]
	shift_pressure_down = np.pad(pressure_levels, (0,1), 'edge')[1:]
	return - (shift_down - shift_up) / (shift_pressure_down - shift_pressure_up)

cpdef scalar_gradient_z_matrix_primitive(np.ndarray a, np.ndarray pressure_levels):
	cdef np.ndarray output = np.zeros_like(a)
	cdef np.int_t i,j,k 
	for k in range(a.shape[2]):
		output[:,:,k] = scalar_gradient_z_3D(a,pressure_levels,k)
	return output

cpdef surface_optical_depth(DTYPE_f lat):
	return 4# + np.cos(lat*inv_90)*2

cpdef surface_optical_depth_array(np.ndarray lat):
	return np.full_like(lat, 4)# + np.cos(lat*inv_90)*2

cpdef thermal_radiation(DTYPE_f a):
	return sigma*(a**4)

cpdef thermal_radiation_matrix(np.ndarray a):
	return sigma*(a**4)

# power incident on (lat,lon) at time t
cpdef solar(DTYPE_f insolation,DTYPE_f  lat,DTYPE_f lon,np.int_t t,DTYPE_f  day,DTYPE_f  year,DTYPE_f  axial_tilt):
	cdef float sun_longitude = -t % day
	cdef float sun_latitude = axial_tilt*np.cos(t*2*np.pi/year)
	cdef float value = insolation*np.cos((lat-sun_latitude)*inv_180)
	cdef float lon_diff, cos_lon

	if value < 0:	
		return 0
	else:
		sun_longitude *= 360/day
		lon_diff = lon-sun_longitude
		cos_lon = np.cos(lon_diff*inv_180) 
		value *= cos_lon
		
		if value < 0:
			if lat + sun_latitude > 90:
				return insolation*np.cos((lat+sun_latitude)*inv_180)*cos_lon
			elif lat + sun_latitude < -90:
				return insolation*np.cos((lat+sun_latitude)*inv_180)*cos_lon
			else:
				return 0
		else:
			return value

cpdef solar_matrix(DTYPE_f insolation, np.ndarray  lat, np.ndarray lon, np.int_t t, DTYPE_f  day, DTYPE_f  year, DTYPE_f  axial_tilt):
	cdef float sun_longitude = -t % day
	# cdef float sun_latitude = axial_tilt*np.cos(t*2*np.pi/year)
	cdef float sun_latitude = axial_tilt*np.sin(t*2*np.pi/year)
	cdef np.ndarray values = insolation*np.cos((lat-sun_latitude)*inv_180)
	cdef np.ndarray lon_diff, cos_lon

	values = np.fmax(values,0)

	sun_longitude *= 360/day
	lon_diff = lon-sun_longitude
	cos_lon = np.cos(lon_diff*inv_180) 
	values = np.outer(values, cos_lon)
	
	cdef np.ndarray in_range_mask = np.logical_and((lat + sun_latitude > -90), (lat + sun_latitude < 90))
	cdef np.ndarray mask1 = np.logical_and(values < 0, np.logical_not(in_range_mask)[:,None])
	cdef np.ndarray mask2 = np.logical_and(values < 0, in_range_mask[:,None])
	cdef np.int_t i
	for i in range(lat.shape[0]):
		values[i,:][mask1[i,:]] = insolation*np.cos((lat[i]+sun_latitude)*inv_180)*cos_lon[mask1[i,:]]
		values[i,:][mask2[i,:]] = 0

	return values

cpdef profile(np.ndarray a):
	return np.mean(np.mean(a,axis=0),axis=0)

cpdef t_to_theta(np.ndarray temperature_atmos, np.ndarray pressure_levels):
	cdef DTYPE_f inv_p0 = 1/pressure_levels[0]

	return temperature_atmos*(pressure_levels*inv_p0)**(-0.286)

cpdef theta_to_t(np.ndarray theta, np.ndarray pressure_levels):
	cdef DTYPE_f inv_p0 = 1/pressure_levels[0]

	return theta*(pressure_levels*inv_p0)**(0.286)

########################################################################################################

cpdef beam_me_up_2D(np.ndarray lats,np.ndarray lon,np.ndarray data,np.int_t grid_size,np.ndarray grid_lat_coords,np.ndarray grid_lon_coords):
	'''Projects data on lat-lon grid to x-y polar grid'''
	f = RectBivariateSpline(lats, lon, data)
	cdef np.ndarray polar_plane = f(grid_lat_coords,grid_lon_coords,grid=False).reshape((grid_size,grid_size))
	return polar_plane

cpdef beam_me_up(np.ndarray lats,np.ndarray lon,np.ndarray data,np.int_t grid_size,np.ndarray grid_lat_coords,np.ndarray grid_lon_coords):
	'''Projects data on lat-lon grid to x-y polar grid'''
	cdef np.ndarray polar_plane = np.zeros((grid_size,grid_size,data.shape[2]))
	cdef np.int_t k
	for k in range(data.shape[2]):
		f = RectBivariateSpline(lats, lon, data[:,:,k])
		polar_plane[:,:,k] = f(grid_lat_coords,grid_lon_coords,grid=False).reshape((grid_size,grid_size))
	return polar_plane

cpdef beam_me_down(lon,data,np.int_t pole_low_index, grid_x_values, grid_y_values,polar_x_coords, polar_y_coords):
	'''projects data from x-y polar grid onto lat-lon grid'''
	cdef np.ndarray resample = np.zeros((int(len(polar_x_coords)/len(lon)),len(lon),data.shape[2]))
	cdef np.int_t k
	for k in range(data.shape[2]):
		f = RectBivariateSpline(x=grid_x_values, y=grid_y_values, z=data[:,:,k])
		resample[:,:,k] = f(polar_x_coords,polar_y_coords,grid=False).reshape((int(len(polar_x_coords)/len(lon)),len(lon)))
	return resample

cpdef combine_data(np.int_t pole_low_index,np.int_t pole_high_index,np.ndarray polar_data,np.ndarray reprojected_data,np.ndarray lat): 
	cdef np.ndarray output = np.zeros_like(polar_data)
	cdef np.int_t overlap = abs(pole_low_index - pole_high_index)
	cdef DTYPE_f scale_reprojected_data, scale_polar_data
	cdef np.int_t nlat = len(lat)
	cdef np.int_t k,i

	if lat[pole_low_index] < 0:		# SOUTH POLE
		for k in range(output.shape[2]):
			for i in range(pole_low_index):
				
				if i < pole_high_index:
					scale_polar_data = 0.0
					scale_reprojected_data = 1.0
				else:
					scale_polar_data = (i+1-pole_high_index)/overlap
					scale_reprojected_data = 1 - (i+1-pole_high_index)/overlap
				
				output[i,:,k] = scale_reprojected_data*reprojected_data[i,:,k] + scale_polar_data*polar_data[i,:,k]
	
	else:							# NORTH POLE
		# polar_data = np.roll(polar_data,int(polar_data.shape[1]/2),axis=1)
		for k in range(output.shape[2]):
			for i in range(nlat-pole_low_index):
				
				if i + pole_low_index + 1 > pole_high_index:
					scale_polar_data = 0.0
					scale_reprojected_data = 1.0
				else:
					scale_polar_data = 1 - i/overlap
					scale_reprojected_data = i/overlap

				output[i,:,k] = scale_reprojected_data*reprojected_data[i,:,k] + scale_polar_data*polar_data[i,:,k]
	return output

cpdef grid_x_gradient_matrix(np.ndarray data,DTYPE_f polar_grid_resolution):
	cdef np.ndarray shift_east = np.pad(data, ((0,0), (1,0), (0,0)), 'reflect', reflect_type='odd')[:,:-1,:]
	cdef np.ndarray shift_west = np.pad(data, ((0,0), (0,1), (0,0)), 'reflect', reflect_type='odd')[:,1:,:]
	return (shift_west - shift_east) / (2 * polar_grid_resolution)

cpdef grid_y_gradient_matrix(np.ndarray data,DTYPE_f polar_grid_resolution):
	cdef np.ndarray shift_south = np.pad(data, ((1,0), (0,0), (0,0)), 'reflect', reflect_type='odd')[:-1,:,:]
	cdef np.ndarray shift_north = np.pad(data, ((0,1), (0,0), (0,0)), 'reflect', reflect_type='odd')[1:,:,:]
	return (shift_north - shift_south) / (2 * polar_grid_resolution)

cpdef grid_p_gradient_matrix(np.ndarray data, np.ndarray pressure_levels):
	cpdef np.ndarray shift_up = np.pad(data, ((0,0), (0,0), (1,0)), 'edge')[:,:,:-1]
	cpdef np.ndarray shift_down = np.pad(data, ((0,0), (0,0), (0,1)), 'edge')[:,:,1:]
	cpdef np.ndarray shift_pressures_up = np.pad(pressure_levels, (1,0), 'edge')[:-1]
	cpdef np.ndarray shift_pressures_down = np.pad(pressure_levels, (0,1), 'edge')[1:]

	return (shift_down - shift_up)/(shift_pressures_down - shift_pressures_up)

cpdef grid_velocities(np.ndarray polar_plane,np.int_t grid_side_length,np.ndarray coriolis_plane,np.ndarray x_dot,np.ndarray y_dot,DTYPE_f polar_grid_resolution, np.int_t sponge_index, np.ndarray temperature, np.ndarray pressure_levels):
	
	cdef np.ndarray x_dot_add = np.zeros_like(x_dot)
	cdef np.ndarray y_dot_add = np.zeros_like(y_dot)

	# cdef np.ndarray w = w_plane(x_dot,y_dot,temperature,pressure_levels,polar_grid_resolution)

	x_dot_add -= 0.5*(x_dot + abs(x_dot))*(x_dot - np.pad(x_dot,((0,0), (1,0), (0,0)), 'reflect', reflect_type='odd')[:,:-1,:])/polar_grid_resolution + 0.5*(x_dot - abs(x_dot))*(np.pad(x_dot, ((0,0), (0,1), (0,0)), 'reflect', reflect_type='odd')[:,1:,:] - x_dot)/polar_grid_resolution
	x_dot_add -= 0.5*(y_dot + abs(y_dot))*(x_dot - np.pad(x_dot,((1,0), (0,0), (0,0)), 'reflect', reflect_type='odd')[:-1,:,:])/polar_grid_resolution + 0.5*(y_dot - abs(y_dot))*(np.pad(x_dot, ((0,1), (0,0), (0,0)), 'reflect', reflect_type='odd')[1:,:,:] - x_dot)/polar_grid_resolution
	# x_dot_add -= 0.5*(w + abs(w))*(x_dot - np.pad(x_dot,((0,0), (0,0), (1,0)), 'reflect', reflect_type='odd')[:,:,:-1])/(pressure_levels - np.pad(pressure_levels,(1,0), 'reflect', reflect_type='odd')[:-1]) + 0.5*(w - abs(w))*(np.pad(x_dot, ((0,0), (0,0), (0,1)), 'reflect', reflect_type='odd')[:,:,1:] - x_dot)/(np.pad(pressure_levels,(0,1), 'reflect', reflect_type='odd')[1:] - pressure_levels)
	x_dot_add += coriolis_plane[:,:,None]*y_dot - grid_x_gradient_matrix(polar_plane,polar_grid_resolution) - 1E-5*x_dot

	y_dot_add -= 0.5*(x_dot + abs(x_dot))*(y_dot - np.pad(y_dot,((0,0), (1,0), (0,0)), 'reflect', reflect_type='odd')[:,:-1,:])/polar_grid_resolution + 0.5*(x_dot - abs(x_dot))*(np.pad(y_dot, ((0,0), (0,1), (0,0)), 'reflect', reflect_type='odd')[:,1:,:] - y_dot)/polar_grid_resolution
	y_dot_add -= 0.5*(y_dot + abs(y_dot))*(y_dot - np.pad(y_dot,((1,0), (0,0), (0,0)), 'reflect', reflect_type='odd')[:-1,:,:])/polar_grid_resolution + 0.5*(y_dot - abs(y_dot))*(np.pad(y_dot, ((0,1), (0,0), (0,0)), 'reflect', reflect_type='odd')[1:,:,:] - y_dot)/polar_grid_resolution
	# x_dot_add -= 0.5*(w + abs(w))*(y_dot - np.pad(y_dot,((0,0), (0,0), (1,0)), 'reflect', reflect_type='odd')[:,:,:-1])/(pressure_levels - np.pad(pressure_levels,(1,0), 'reflect', reflect_type='odd')[:-1]) + 0.5*(w - abs(w))*(np.pad(y_dot, ((0,0), (0,0), (0,1)), 'reflect', reflect_type='odd')[:,:,1:] - y_dot)/(np.pad(pressure_levels,(0,1), 'reflect', reflect_type='odd')[1:] - pressure_levels)
	y_dot_add += - coriolis_plane[:,:,None]*x_dot - grid_y_gradient_matrix(polar_plane,polar_grid_resolution) - 1E-5*y_dot

	x_dot_add[:,:,sponge_index:] *= 0
	y_dot_add[:,:,sponge_index:] *= 0

	# sponge layer
	x_dot_add[:,:,sponge_index:] -= 0.5*(x_dot[:,:,sponge_index:] + abs(x_dot[:,:,sponge_index:]))*(x_dot[:,:,sponge_index:] - np.pad(x_dot[:,:,sponge_index:],((0,0), (1,0), (0,0)), 'reflect', reflect_type='odd')[:,:-1,:])/polar_grid_resolution + 0.5*(x_dot[:,:,sponge_index:] - abs(x_dot[:,:,sponge_index:]))*(np.pad(x_dot[:,:,sponge_index:], ((0,0), (0,1), (0,0)), 'reflect', reflect_type='odd')[:,1:,:] - x_dot[:,:,sponge_index:])/polar_grid_resolution
	x_dot_add[:,:,sponge_index:] -= 0.5*(y_dot[:,:,sponge_index:] + abs(y_dot[:,:,sponge_index:]))*(x_dot[:,:,sponge_index:] - np.pad(x_dot[:,:,sponge_index:],((1,0), (0,0), (0,0)), 'reflect', reflect_type='odd')[:-1,:,:])/polar_grid_resolution + 0.5*(y_dot[:,:,sponge_index:] - abs(y_dot[:,:,sponge_index:]))*(np.pad(x_dot[:,:,sponge_index:], ((0,1), (0,0), (0,0)), 'reflect', reflect_type='odd')[1:,:,:] - x_dot[:,:,sponge_index:])/polar_grid_resolution
	# x_dot_add[:,:,sponge_index:] -= 0.5*(w[:,:,sponge_index:] + abs(w[:,:,sponge_index:]))*(x_dot[:,:,sponge_index:] - np.pad(x_dot[:,:,sponge_index:],((0,0), (0,0), (1,0)), 'reflect', reflect_type='odd')[:,:,:-1])/(pressure_levels[sponge_index:] - np.pad(pressure_levels[sponge_index:],(1,0), 'reflect', reflect_type='odd')[:-1]) + 0.5*(w[:,:,sponge_index:] - abs(w[:,:,sponge_index:]))*(np.pad(x_dot[:,:,sponge_index:], ((0,0), (0,0), (0,1)), 'reflect', reflect_type='odd')[:,:,1:] - x_dot[:,:,sponge_index:])/(np.pad(pressure_levels[sponge_index:],(0,1), 'reflect', reflect_type='odd')[1:] - pressure_levels[sponge_index:])
	x_dot_add[:,:,sponge_index:] -= 1E-3*x_dot[:,:,sponge_index:]

	y_dot_add[:,:,sponge_index:] -= 0.5*(x_dot[:,:,sponge_index:] + abs(x_dot[:,:,sponge_index:]))*(y_dot[:,:,sponge_index:] - np.pad(y_dot[:,:,sponge_index:],((0,0), (1,0), (0,0)), 'reflect', reflect_type='odd')[:,:-1,:])/polar_grid_resolution + 0.5*(x_dot[:,:,sponge_index:] - abs(x_dot[:,:,sponge_index:]))*(np.pad(y_dot[:,:,sponge_index:], ((0,0), (0,1), (0,0)), 'reflect', reflect_type='odd')[:,1:,:] - y_dot[:,:,sponge_index:])/polar_grid_resolution
	y_dot_add[:,:,sponge_index:] -= 0.5*(y_dot[:,:,sponge_index:] + abs(y_dot[:,:,sponge_index:]))*(y_dot[:,:,sponge_index:] - np.pad(y_dot[:,:,sponge_index:],((1,0), (0,0), (0,0)), 'reflect', reflect_type='odd')[:-1,:,:])/polar_grid_resolution + 0.5*(y_dot[:,:,sponge_index:] - abs(y_dot[:,:,sponge_index:]))*(np.pad(y_dot[:,:,sponge_index:], ((0,1), (0,0), (0,0)), 'reflect', reflect_type='odd')[1:,:,:] - y_dot[:,:,sponge_index:])/polar_grid_resolution
	# y_dot_add[:,:,sponge_index:] -= 0.5*(w[:,:,sponge_index:] + abs(w[:,:,sponge_index:]))*(y_dot[:,:,sponge_index:] - np.pad(y_dot[:,:,sponge_index:],((0,0), (0,0), (1,0)), 'reflect', reflect_type='odd')[:,:,:-1])/(pressure_levels[sponge_index:] - np.pad(pressure_levels[sponge_index:],(1,0), 'reflect', reflect_type='odd')[:-1]) + 0.5*(w[:,:,sponge_index:] - abs(w[:,:,sponge_index:]))*(np.pad(y_dot[:,:,sponge_index:], ((0,0), (0,0), (0,1)), 'reflect', reflect_type='odd')[:,:,1:] - y_dot[:,:,sponge_index:])/(np.pad(pressure_levels[sponge_index:],(0,1), 'reflect', reflect_type='odd')[1:] - pressure_levels[sponge_index:])
	y_dot_add[:,:,sponge_index:] -= 1E-3*y_dot[:,:,sponge_index:]

	return x_dot_add,y_dot_add

cpdef project_velocities_north(np.ndarray lon,np.ndarray x_dot,np.ndarray y_dot,np.int_t pole_low_index_N,np.int_t pole_high_index_N,np.ndarray grid_x_values_N,np.ndarray grid_y_values_N,list polar_x_coords_N,list polar_y_coords_N):

	cdef np.ndarray reproj_x_dot = beam_me_down(lon,x_dot,pole_low_index_N,grid_x_values_N,grid_y_values_N,polar_x_coords_N,polar_y_coords_N)		
	cdef np.ndarray reproj_y_dot = beam_me_down(lon,y_dot,pole_low_index_N,grid_x_values_N,grid_y_values_N,polar_x_coords_N,polar_y_coords_N)

	cdef np.ndarray reproj_u = + reproj_x_dot*np.sin(lon[None,:,None]*np.pi/180) + reproj_y_dot*np.cos(lon[None,:,None]*np.pi/180)
	cdef np.ndarray reproj_v = + reproj_x_dot*np.cos(lon[None,:,None]*np.pi/180) - reproj_y_dot*np.sin(lon[None,:,None]*np.pi/180)

	reproj_u = np.flip(reproj_u,axis=1)
	reproj_v = np.flip(reproj_v,axis=1)
	
	return reproj_u, reproj_v

cpdef project_velocities_south(np.ndarray lon,np.ndarray x_dot,np.ndarray y_dot,np.int_t pole_low_index_S,np.int_t pole_high_index_S,np.ndarray grid_x_values_S,np.ndarray grid_y_values_S,list polar_x_coords_S,list polar_y_coords_S):
	cdef np.ndarray reproj_x_dot = beam_me_down(lon,x_dot,pole_low_index_S,grid_x_values_S,grid_y_values_S,polar_x_coords_S,polar_y_coords_S)		
	cdef np.ndarray reproj_y_dot = beam_me_down(lon,y_dot,pole_low_index_S,grid_x_values_S,grid_y_values_S,polar_x_coords_S,polar_y_coords_S)

	cdef np.ndarray reproj_u = + reproj_x_dot*np.sin(lon[None,:,None]*np.pi/180) + reproj_y_dot*np.cos(lon[None,:,None]*np.pi/180)
	cdef np.ndarray reproj_v = - reproj_x_dot*np.cos(lon[None,:,None]*np.pi/180) + reproj_y_dot*np.sin(lon[None,:,None]*np.pi/180)

	return reproj_u, reproj_v

cpdef polar_plane_advect(np.ndarray data,np.ndarray x_dot,np.ndarray y_dot, np.ndarray w, DTYPE_f polar_grid_resolution):
	
	cpdef np.ndarray output = np.zeros_like(data)

	output += 0.5*(x_dot + abs(x_dot))*(data - np.pad(data, ((0,0), (1,0), (0,0)), 'reflect', reflect_type='odd')[:,:-1,:])/polar_grid_resolution 
	output += 0.5*(x_dot - abs(x_dot))*(np.pad(data, ((0,0), (0,1), (0,0)), 'reflect', reflect_type='odd')[:,1:,:] - data)/polar_grid_resolution
	
	output += 0.5*(y_dot + abs(y_dot))*(data - np.pad(data, ((1,0), (0,0), (0,0)), 'reflect', reflect_type='odd')[:-1,:,:])/polar_grid_resolution
	output += 0.5*(y_dot - abs(y_dot))*(np.pad(data, ((0,1), (0,0), (0,0)), 'reflect', reflect_type='odd')[1:,:,:] - data)/polar_grid_resolution

	output += 0.5*(w + abs(w))*(data - np.pad(data, ((1,0), (0,0), (0,0)), 'reflect', reflect_type='odd')[:-1,:,:])/polar_grid_resolution
	output += 0.5*(w - abs(w))*(np.pad(data, ((0,1), (0,0), (0,0)), 'reflect', reflect_type='odd')[1:,:,:] - data)/polar_grid_resolution
	
	return output

cpdef upload_velocities(np.ndarray lat,np.ndarray lon,np.ndarray reproj_u,np.ndarray reproj_v,np.int_t grid_size,np.ndarray grid_lat_coords,np.ndarray grid_lon_coords):
	
	cdef np.ndarray grid_u = beam_me_up(lat,lon,reproj_u,grid_size,grid_lat_coords,grid_lon_coords)
	cdef np.ndarray grid_v = beam_me_up(lat,lon,reproj_v,grid_size,grid_lat_coords,grid_lon_coords)

	cdef np.int_t nlevels = reproj_u.shape[2]

	cdef np.ndarray x_dot = np.zeros((grid_size,grid_size,nlevels))	
	cdef np.ndarray y_dot = np.zeros((grid_size,grid_size,nlevels))	

	grid_lon_coords = grid_lon_coords.reshape((grid_size,grid_size))

	if lat[0] < 0:
		for k in range(nlevels):
			x_dot[:,:,k] = grid_u[:,:,k]*np.sin(grid_lon_coords*np.pi/180) - grid_v[:,:,k]*np.cos(grid_lon_coords*np.pi/180)
			y_dot[:,:,k] = grid_u[:,:,k]*np.cos(grid_lon_coords*np.pi/180) + grid_v[:,:,k]*np.sin(grid_lon_coords*np.pi/180)
	else:
		for k in range(nlevels):
			x_dot[:,:,k] = -grid_u[:,:,k]*np.sin(grid_lon_coords*np.pi/180) + grid_v[:,:,k]*np.cos(grid_lon_coords*np.pi/180)
			y_dot[:,:,k] = -grid_u[:,:,k]*np.cos(grid_lon_coords*np.pi/180) - grid_v[:,:,k]*np.sin(grid_lon_coords*np.pi/180)

	return x_dot,y_dot

cpdef w_plane(np.ndarray x_dot,np.ndarray y_dot,np.ndarray temperature,np.ndarray pressure_levels,DTYPE_f polar_grid_resolution):
	''' calculates vertical velocity omega on a given cartesian polar plane'''

	cdef np.ndarray w_temp = np.zeros_like(x_dot)
	cdef np.int_t k
	temperature = theta_to_t(temperature,pressure_levels)

	cdef np.ndarray flow_divergence = grid_x_gradient_matrix(x_dot, polar_grid_resolution) + grid_y_gradient_matrix(y_dot, polar_grid_resolution)
	
	for k in np.arange(1,len(pressure_levels)).tolist():
		w_temp[:,:,k] = - np.trapz(flow_divergence[:,:,k:],pressure_levels[k:])
	
	return w_temp
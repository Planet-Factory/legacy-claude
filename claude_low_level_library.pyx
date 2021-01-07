# claude low level library

import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t DTYPE_f
cdef float inv_180 = np.pi/180
cdef float inv_90 = np.pi/90
cdef DTYPE_f sigma = 5.67E-8

# define various useful differential functions:
# gradient of scalar field a in the local x direction at point i,j
cpdef scalar_gradient_x(np.ndarray a, np.ndarray dx, np.int_t nlon, np.int_t i, np.int_t j, np.int_t k):
	return (a[i,(j+1)%nlon,k]-a[i,(j-1)%nlon,k])/(dx[i])

cpdef scalar_gradient_x_matrix(np.ndarray a,np.ndarray dx):
	return (np.roll(a, -1, axis=1) - np.roll(a, 1, axis=1)) / dx[:, None, None]

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
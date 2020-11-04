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
def scalar_gradient_x(np.ndarray a,np.ndarray dx,np.int_t nlon,np.int_t i,np.int_t j,np.int_t k):
	return (a[i,(j+1)%nlon,k]-a[i,(j-1)%nlon,k])/dx[i]

def scalar_gradient_x_2D(np.ndarray a,np.ndarray dx,np.int_t nlon,np.int_t i,np.int_t j):
	return (a[i,(j+1)%nlon]-a[i,(j-1)%nlon])/dx[i]

# gradient of scalar field a in the local y direction at point i,j
def scalar_gradient_y(np.ndarray a,DTYPE_f dy,np.int_t nlat,np.int_t i,np.int_t j,np.int_t k):
	if i == 0:
		return 2*(a[i+1,j,k]-a[i,j,k])/dy
	elif i == nlat-1:
		return 2*(a[i,j,k]-a[i-1,j,k])/dy
	else:
		return (a[i+1,j,k]-a[i-1,j,k])/dy

def scalar_gradient_y_2D(np.ndarray a,DTYPE_f dy,np.int_t nlat,np.int_t i,np.int_t j):
	if i == 0:
		return 2*(a[i+1,j]-a[i,j])/dy
	elif i == nlat-1:
		return 2*(a[i,j]-a[i-1,j])/dy
	else:
		return (a[i+1,j]-a[i-1,j])/dy

def scalar_gradient_z_1D(np.ndarray a,np.ndarray pressure_levels,np.int_t k):
	cdef np.int_t nlevels = len(pressure_levels)
	if k == 0:
		return -(a[k+1]-a[k])/(pressure_levels[k+1]-pressure_levels[k])
	elif k == nlevels-1:
		return -(a[k]-a[k-1])/(pressure_levels[k]-pressure_levels[k-1])
	else:
		return -(a[k+1]-a[k-1])/(pressure_levels[k+1]-pressure_levels[k-1])

def surface_optical_depth(DTYPE_f lat):
	return 4# + np.cos(lat*inv_90)*2

def thermal_radiation(DTYPE_f a):
	return sigma*(a**4)

# power incident on (lat,lon) at time t
def solar(DTYPE_f insolation,DTYPE_f  lat,DTYPE_f lon,np.int_t t,DTYPE_f  day,DTYPE_f  year,DTYPE_f  axial_tilt):
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

def profile(np.ndarray a):
	return np.mean(np.mean(a,axis=0),axis=0)

def t_to_theta(np.ndarray temperature_atmos, np.ndarray pressure_levels):
	cdef np.ndarray output = np.zeros_like(temperature_atmos)
	cdef np.int_t k
	cdef DTYPE_f inv_p0

	inv_p0 = 1/pressure_levels[0]
	for k in range(len(pressure_levels)):
		output[:,:,k] = temperature_atmos[:,:,k]*(pressure_levels[k]*inv_p0)**(-0.286)

	return output

def theta_to_t(np.ndarray theta, np.ndarray pressure_levels):
	cdef np.ndarray output = np.zeros_like(theta)
	cdef np.int_t k
	cdef DTYPE_f inv_p0

	inv_p0 = 1/pressure_levels[0]
	for k in range(len(pressure_levels)):
		output[:,:,k] = theta[:,:,k]*(pressure_levels[k]*inv_p0)**(0.286)

	return output
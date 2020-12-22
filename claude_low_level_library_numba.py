# claude_low_level_library with numba acceleration

# claude low level library

import numpy as np
from numba import njit, prange

inv_180 = np.pi/180
inv_90 = np.pi/90
sigma = 5.67E-8

# define various useful differential functions:
# gradient of scalar field a in the local x direction at point i,j
@njit(cache=True)
def scalar_gradient_x(a, dx, nlon, i, j, k):
	return (a[i,(j+1)%nlon,k]-a[i,(j-1)%nlon,k])/dx[i]

@njit(cache=True)
def scalar_gradient_x_2D(a, dx, nlon, i, j):
	return (a[i,(j+1)%nlon]-a[i,(j-1)%nlon])/dx[i]


# gradient of scalar field a in the local y direction at point i,j
@njit(cache=True)
def scalar_gradient_y(a, dy, nlat, i, j, k):
	if i == 0:
		return 2*(a[i+1,j,k]-a[i,j,k])/dy
	elif i == nlat-1:
		return 2*(a[i,j,k]-a[i-1,j,k])/dy
	else:
		return (a[i+1,j,k]-a[i-1,j,k])/dy


@njit(cache=True)
def scalar_gradient_y_2D(dy, nlat, i, j):
	if i == 0:
		return 2*(a[i+1,j]-a[i,j])/dy
	elif i == nlat-1:
		return 2*(a[i,j]-a[i-1,j])/dy
	else:
		return (a[i+1,j]-a[i-1,j])/dy

@njit(cache=True)
def scalar_gradient_z_1D(a, pressure_levels, k):
	nlevels = len(pressure_levels)
	if k == 0:
		return -(a[k+1]-a[k])/(pressure_levels[k+1]-pressure_levels[k])
	elif k == nlevels-1:
		return -(a[k]-a[k-1])/(pressure_levels[k]-pressure_levels[k-1])
	else:
		return -(a[k+1]-a[k-1])/(pressure_levels[k+1]-pressure_levels[k-1])


@njit(cache=True)
def surface_optical_depth(lat):
	return 4# + np.cos(lat*inv_90)*2

@njit(cache=True)
def thermal_radiation(a):
	return sigma*(a**4)

# power incident on (lat,lon) at time t
@njit(cache=True)
def solar(insolation, lat, lon, t, day, year, axial_tilt):
	sun_longitude = -t % day
	sun_latitude = axial_tilt*np.cos(t*2*np.pi/year)
	value = insolation*np.cos((lat-sun_latitude)*inv_180)
	lon_diff = cos_lon = 0.0

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

@njit(cache=True)
def profile(a):
	return np.mean(np.mean(a,axis=0),axis=0)

@njit(cache=True, parallel=True)
def t_to_theta(temperature_atmos, pressure_levels):
	output = np.zeros_like(temperature_atmos)
	k = 0
	inv_p0 = 0.0

	inv_p0 = 1/pressure_levels[0]
	for k in prange(len(pressure_levels)):
		output[:,:,k] = temperature_atmos[:,:,k]*(pressure_levels[k]*inv_p0)**(-0.286)

	return output

@njit(cache=True, parallel=True)
def theta_to_t(theta, pressure_levels):
	output = np.zeros_like(theta)

	inv_p0 = 1/pressure_levels[0]
	for k in prange(len(pressure_levels)):
		output[:,:,k] = theta[:,:,k]*(pressure_levels[k]*inv_p0)**(0.286)

	return output
# claude low level library

import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_f
sigma = 5.67E-8

# define various useful differential functions:
# gradient of scalar field a in the local x direction at point i,j
def scalar_gradient_x(a,dx,nlon,i,j,k):
	return (a[i,(j+1)%nlon,k]-a[i,(j-1)%nlon,k])/dx[i]

def scalar_gradient_x_2D(a,dx,nlon,i,j)
	return (a[i,(j+1)%nlon]-a[i,(j-1)%nlon])/dx[i]

# gradient of scalar field a in the local y direction at point i,j
def scalar_gradient_y(a,dy,nlat,i,j,k):
	if i == 0:
		return 2*(a[i+1,j]-a[i,j])/dy
	elif i == nlat-1:
		return 2*(a[i,j]-a[i-1,j])/dy
	else:
		return (a[i+1,j]-a[i-1,j])/dy

def scalar_gradient_y_2d(a,dy,nlat,i,j)
	if i == 0:
		return 2*(a[i+1,j]-a[i,j])/dy
	elif i == nlat-1:
		return 2*(a[i,j]-a[i-1,j])/dy
	else:
		return (a[i+1,j]-a[i-1,j])/dy

def scalar_gradient_z(a,dz,i,j,k):
	output = np.zeros_like(a)
	nlevels = len(dz)
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

def surface_optical_depth(lat):
	return 4 + np.cos(lat*np.pi/90)*2.5/2

def thermal_radiation(a):
	return sigma*(a**4)

# power incident on (lat,lon) at time t
def solar(insolation, lat, lon, t, day, year, axial_tilt):
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

def profile(a):
	return np.mean(np.mean(a,axis=0),axis=0)
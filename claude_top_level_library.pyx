# claude_top_level_library

import claude_low_level_library as low_level
import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t DTYPE_f

# laplacian of scalar field a
def laplacian_2D(np.ndarray a,np.ndarray dx,DTYPE_f dy):
	cdef np.ndarray output = np.zeros_like(a)
	cdef np.int_t nlat,nlon,i,j
	cdef DTYPE_f inv_dx, inv_dy
	nlat = a.shape[0]
	nlon = a.shape[1]
	inv_dy = 1/dy
	for i in np.arange(1,nlat-1):
		inv_dx = dx[i]
		for j in range(nlon):
			output[i,j] = (low_level.scalar_gradient_x_2D(a,dx,nlon,i,j) - low_level.scalar_gradient_x_2D(a,dx,nlon,i,j))*inv_dx + (low_level.scalar_gradient_y_2D(a,dy,nlat,i+1,j) - low_level.scalar_gradient_y_2D(a,dy,nlat,i-1,j))*inv_dy
	return output

def laplacian_3D(np.ndarray a,np.ndarray dx,DTYPE_f dy,np.ndarray dz):
	cdef np.ndarray output = np.zeros_like(a)
	cdef np.int_t nlat,nlon,nlevels,i,j,k
	cdef DTYPE_f inv_dx, inv_dy
	nlat = a.shape[0]
	nlon = a.shape[1]
	nlevels = a.shape[2]
	inv_dy = 1/dy
	for i in np.arange(1,nlat-1):
		inv_dx = 1/dx[i]
		for j in range(nlon):
			for k in range(nlevels-1):
				output[i,j,k] = (low_level.scalar_gradient_x(a,dx,nlon,i,j,k) - low_level.scalar_gradient_x(a,dx,nlon,i,j,k))*inv_dx + (low_level.scalar_gradient_y(a,dy,nlat,i+1,j,k) - low_level.scalar_gradient_y(a,dy,nlat,i-1,j,k))*inv_dy + (low_level.scalar_gradient_z(a,dz,i,j,k+1)-low_level.scalar_gradient_z(a,dz,i,j,k-1))/(2*dz[k])
	return output

# divergence of (a*u) where a is a scalar field and u is the atmospheric velocity field
def divergence_with_scalar(np.ndarray a,np.ndarray u,np.ndarray v,np.ndarray w,np.ndarray dx,DTYPE_f dy,np.ndarray dz):
	cdef np.ndarray output = np.zeros_like(a)
	cdef np.ndarray au, av, aw
	cdef np.int_t nlat, nlon, nlevels, i, j, k 

	nlat = output.shape[0]
	nlon = output.shape[1]
	nlevels = output.shape[2]

	au = a*u
	av = a*v
	aw = a*w

	for i in range(nlat):
		for j in range(nlon):
			for k in range(nlevels):
				output[i,j,k] = low_level.scalar_gradient_x(au,dx,nlon,i,j,k) + low_level.scalar_gradient_y(av,dy,nlat,i,j,k) + 1*low_level.scalar_gradient_z(aw,dz,i,j,k)
	return output

# specifically thermal advection, converts temperature field to potential temperature field, then advects (adiabatically), then converts back to temperature
def thermal_advection(np.ndarray temperature_atmos,np.ndarray air_pressure,np.ndarray u,np.ndarray v,np.ndarray w,np.ndarray dx,DTYPE_f dy,np.ndarray dz):
	cdef np.ndarray output = np.zeros_like(temperature_atmos)
	cdef np.ndarray au, av, aw
	cdef np.int_t nlat, nlon, nlevels, i, j, k 
	cdef np.ndarray theta = low_level.t_to_theta(temperature_atmos,air_pressure)

	nlat = output.shape[0]
	nlon = output.shape[1]
	nlevels = output.shape[2]

	au = theta*u
	av = theta*v
	aw = theta*w

	for i in range(nlat):
		for j in range(nlon):
			for k in range(nlevels):
				output[i,j,k] = low_level.scalar_gradient_x(au,dx,nlon,i,j,k) + low_level.scalar_gradient_y(av,dy,nlat,i,j,k) + 1*low_level.scalar_gradient_z(aw,dz,i,j,k)
	
	output = low_level.theta_to_t(output,air_pressure)

	return output

def radiation_calculation(np.ndarray temperature_world, np.ndarray temperature_atmos, np.ndarray air_pressure, np.ndarray air_density, np.ndarray heat_capacity_earth, np.ndarray albedo, DTYPE_f insolation, np.ndarray lat, np.ndarray lon, np.ndarray heights, np.ndarray dz, np.int_t t, np.int_t dt, DTYPE_f day, DTYPE_f year, DTYPE_f axial_tilt):
	# calculate change in temperature of ground and atmosphere due to radiative imbalance
	cdef np.int_t nlat,nlon,nlevels,i,j
	cdef DTYPE_f fl = 0.1
	cdef np.ndarray upward_radiation,downward_radiation,optical_depth,Q, pressure_profile, density_profile
	cdef DTYPE_f sun_lat

	nlat = temperature_atmos.shape[0]	
	nlon = temperature_atmos.shape[1]
	nlevels = temperature_atmos.shape[2]

	upward_radiation = np.zeros(nlevels)
	downward_radiation = np.zeros(nlevels)
	optical_depth = np.zeros(nlevels)
	Q = np.zeros(nlevels)

	for i in range(nlat):
		sun_lat = low_level.surface_optical_depth(lat[i])
		for j in range(nlon):
			# calculate optical depth
			pressure_profile = air_pressure[i,j,:]
			density_profile = air_density[i,j,:]
			optical_depth = sun_lat*(fl*(pressure_profile/pressure_profile[0]) + (1-fl)*(pressure_profile/pressure_profile[0])**4)
			
			# calculate upward longwave flux, bc is thermal radiation at surface
			upward_radiation[0] = low_level.thermal_radiation(temperature_world[i,j])
			for k in np.arange(1,nlevels):
				upward_radiation[k] = (upward_radiation[k-1] - (optical_depth[k]-optical_depth[k-1])*(low_level.thermal_radiation(temperature_atmos[i,j,k])))/(1+optical_depth[k-1]-optical_depth[k])

			# calculate downward longwave flux, bc is zero at TOA (in model)
			downward_radiation[-1] = 0
			for k in np.arange(0,nlevels-1)[::-1]:
				downward_radiation[k] = (downward_radiation[k+1] - low_level.thermal_radiation(temperature_atmos[i,j,k])*(optical_depth[k+1]-optical_depth[k]))/(1 + optical_depth[k] - optical_depth[k+1])
			
			# gradient of difference provides heating at each level
			for k in np.arange(nlevels):
				Q[k] = -low_level.scalar_gradient_z_1D(upward_radiation-downward_radiation,dz,k)/(1E3*density_profile[k])
				# make sure model does not have a higher top than 50km!!
				# approximate SW heating of ozone
				if heights[k] > 20E3:
					Q[k] += low_level.solar(5,lat[i],lon[j],t,day, year, axial_tilt)*((((heights[k]-20E3)/1E3)**2)/(30**2))/(24*60*60)

			temperature_atmos[i,j,:] += Q*dt

			# update surface temperature with shortwave radiation flux
			temperature_world[i,j] += dt*((1-albedo[i,j])*(low_level.solar(insolation,lat[i],lon[j],t, day, year, axial_tilt) + downward_radiation[0]) - upward_radiation[0])/heat_capacity_earth[i,j] 
	
	return temperature_world, temperature_atmos

def velocity_calculation(np.ndarray u,np.ndarray v,np.ndarray air_pressure,np.ndarray old_pressure,np.ndarray air_density,np.ndarray coriolis,DTYPE_f gravity,np.ndarray dx,DTYPE_f dy,DTYPE_f dt):
	# introduce temporary arrays to update velocity in the atmosphere
	cdef np.ndarray u_temp = np.zeros_like(u)
	cdef np.ndarray v_temp = np.zeros_like(v)
	cdef np.ndarray w_temp = np.zeros_like(u)

	cdef np.int_t nlat,nlon,nlevels,i,j,k
	cdef DTYPE_f inv_density,inv_dt_grav

	nlat = air_pressure.shape[0]
	nlon = air_pressure.shape[1]
	nlevels = air_pressure.shape[2]
	inv_dt_grav = 1/(dt*gravity)

	# calculate acceleration of atmosphere using primitive equations on beta-plane
	for i in np.arange(3,nlat-3).tolist():
		for j in range(nlon):
			for k in range(nlevels):
				inv_density = 1/air_density[i,j,k]
				u_temp[i,j,k] += dt*( -u[i,j,k]*low_level.scalar_gradient_x(u,dx,nlon,i,j,k) - v[i,j,k]*low_level.scalar_gradient_y(u,dy,nlat,i,j,k) + coriolis[i]*v[i,j,k] - low_level.scalar_gradient_x(air_pressure,dx,nlon,i,j,k)*inv_density )
				v_temp[i,j,k] += dt*( -u[i,j,k]*low_level.scalar_gradient_x(v,dx,nlon,i,j,k) - v[i,j,k]*low_level.scalar_gradient_y(v,dy,nlat,i,j,k) - coriolis[i]*u[i,j,k] - low_level.scalar_gradient_y(air_pressure,dy,nlat,i,j,k)*inv_density )
				w_temp[i,j,k] += -(air_pressure[i,j,k]-old_pressure[i,j,k])*inv_density*inv_dt_grav

	u += u_temp
	v += v_temp
	w = w_temp

	# approximate friction
	u *= 0.95
	v *= 0.95

	u[:,:,0] *= 0.8
	v[:,:,0] *= 0.8

	return u,v,w

def smoothing_3D(np.ndarray a,DTYPE_f smooth_parameter, DTYPE_f vert_smooth_parameter=0.5):
	cdef np.int_t nlat = a.shape[0]
	cdef np.int_t nlon = a.shape[1]
	cdef np.int_t nlevels = a.shape[2]
	smooth_parameter *= 0.5
	cdef np.ndarray test = np.fft.fftn(a)
	test[int(nlat*smooth_parameter):int(nlat*(1-smooth_parameter)),:,:] = 0
	test[:,int(nlon*smooth_parameter):int(nlon*(1-smooth_parameter)),:] = 0
	test[:,:,int(nlevels*vert_smooth_parameter):int(nlevels*(1-vert_smooth_parameter))] = 0
	return np.fft.ifftn(test).real

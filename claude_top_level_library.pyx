# claude_top_level_library

import claude_low_level_library as low_level
import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t DTYPE_f

# laplacian of scalar field a
cpdef laplacian_2D(np.ndarray a,np.ndarray dx,DTYPE_f dy):
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

cpdef laplacian_3D(np.ndarray a,np.ndarray dx,DTYPE_f dy,np.ndarray dz):
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
cpdef divergence_with_scalar(np.ndarray a,np.ndarray u,np.ndarray v,np.ndarray w,np.ndarray dx,DTYPE_f dy,np.ndarray pressure_levels):
	cdef np.ndarray au, av, aw
	au = a*u
	av = a*v
	aw = a*w

	cdef np.ndarray output = low_level.scalar_gradient_x_matrix(au,dx) + low_level.scalar_gradient_y_matrix(av,dy) + low_level.scalar_gradient_z_matrix(aw, pressure_levels)

	return output

cpdef radiation_calculation(np.ndarray temperature_world, np.ndarray potential_temperature, np.ndarray pressure_levels, np.ndarray heat_capacity_earth, np.ndarray albedo, DTYPE_f insolation, np.ndarray lat, np.ndarray lon, np.int_t t, np.int_t dt, DTYPE_f day, DTYPE_f year, DTYPE_f axial_tilt):
	# calculate change in temperature of ground and atmosphere due to radiative imbalance
	cdef np.int_t nlat,nlon,nlevels,k
	cdef DTYPE_f fl = 0.1
	cdef DTYPE_f inv_day = 1/(24*60*60)

	nlat = lat.shape[0]	
	nlon = lon.shape[0]
	nlevels = pressure_levels.shape[0]

	cdef np.ndarray temperature_atmos = low_level.theta_to_t(potential_temperature,pressure_levels)
	cdef np.ndarray sun_lat = low_level.surface_optical_depth_array(lat)
	cdef np.ndarray optical_depth = np.outer(sun_lat, (fl*(pressure_levels/pressure_levels[0]) + (1-fl)*(pressure_levels/pressure_levels[0])**4))

	# calculate upward longwave flux, bc is thermal radiation at surface
	cpdef np.ndarray upward_radiation = np.zeros((nlat,nlon,nlevels))
	upward_radiation[:,:,0] = low_level.thermal_radiation_matrix(temperature_world)
	for k in np.arange(1,nlevels):
		upward_radiation[:,:,k] = (upward_radiation[:,:,k-1] - (optical_depth[:,None,k]-optical_depth[:,None,k-1])*(low_level.thermal_radiation_matrix(temperature_atmos[:,:,k])))/(1 + optical_depth[:,None,k-1] - optical_depth[:,None,k])

	# calculate downward longwave flux, bc is zero at TOA (in model)
	cpdef np.ndarray downward_radiation = np.zeros((nlat,nlon,nlevels))
	downward_radiation[:,:,-1] = 0
	for k in np.arange(0,nlevels-1)[::-1]:
		downward_radiation[:,:,k] = (downward_radiation[:,:,k+1] - low_level.thermal_radiation_matrix(temperature_atmos[:,:,k])*(optical_depth[:,None,k+1]-optical_depth[:,None,k]))/(1 + optical_depth[:,None,k] - optical_depth[:,None,k+1])
	
	# gradient of difference provides heating at each level
	cpdef np.ndarray Q = np.zeros((nlat,nlon,nlevels))
	cpdef np.ndarray z_gradient = low_level.scalar_gradient_z_matrix(upward_radiation - downward_radiation, pressure_levels)
	cpdef np.ndarray solar_matrix = low_level.solar_matrix(75,lat,lon,t,day, year, axial_tilt)
	for k in np.arange(nlevels):
		Q[:,:,k] = -287*temperature_atmos[:,:,k]*z_gradient[:,:,k]/(1000*pressure_levels[None,None,k])
		# approximate SW heating of ozone
		if pressure_levels[k] < 400*100:
			Q[:,:,k] += solar_matrix*inv_day*(100/pressure_levels[k])

	temperature_atmos += Q*dt

	# update surface temperature with shortwave radiation flux
	temperature_world += dt*((1-albedo)*(low_level.solar_matrix(insolation,lat,lon,t, day, year, axial_tilt) + downward_radiation[:,:,0]) - upward_radiation[:,:,0])/heat_capacity_earth 

	return temperature_world, low_level.t_to_theta(temperature_atmos,pressure_levels)

cpdef velocity_calculation(np.ndarray u,np.ndarray v,np.ndarray w,np.ndarray pressure_levels,np.ndarray geopotential,np.ndarray potential_temperature,np.ndarray coriolis,DTYPE_f gravity,np.ndarray dx,DTYPE_f dy,DTYPE_f dt):

	# calculate acceleration of atmosphere using primitive equations on beta-plane
	cdef np.ndarray u_temp = dt*(-u*low_level.scalar_gradient_x_matrix(u, dx) - v*low_level.scalar_gradient_y_matrix(u, dy) - w*low_level.scalar_gradient_z_matrix(u, pressure_levels) + coriolis[:, None, None]*v - low_level.scalar_gradient_x_matrix(geopotential, dx) - 1E-4*u)
	cdef np.ndarray v_temp = dt*(-u*low_level.scalar_gradient_x_matrix(v, dx) - v*low_level.scalar_gradient_y_matrix(v, dy) - w*low_level.scalar_gradient_z_matrix(v, pressure_levels) - coriolis[:, None, None]*u - low_level.scalar_gradient_y_matrix(geopotential, dy) - 1E-4*v)
	
	u_temp[-2:,:,:] = 0
	v_temp[-2:,:,:] = 0
	u_temp[:2,:,:] = 0
	v_temp[:2:,:] = 0

	u += u_temp
	v += v_temp
	
	# approximate surface friction
	u[:,:,0] *= 0.8
	v[:,:,0] *= 0.8

	return u,v

cpdef w_calculation(np.ndarray u,np.ndarray v,np.ndarray w,np.ndarray pressure_levels,np.ndarray geopotential,np.ndarray potential_temperature,np.ndarray coriolis,DTYPE_f gravity,np.ndarray dx,DTYPE_f dy,DTYPE_f dt):
	cdef np.ndarray w_temp = np.zeros_like(u)
	cdef np.ndarray temperature_atmos = low_level.theta_to_t(potential_temperature,pressure_levels) 
	
	cdef np.int_t nlevels, k
	nlevels = len(pressure_levels)
	
	for k in np.arange(1,nlevels).tolist():
		w_temp[:,:,k] = w_temp[:,:,k-1] - (pressure_levels[k] - pressure_levels[k-1]) * pressure_levels[k] * gravity * ( low_level.scalar_gradient_x_matrix(u, dx)[:,:,k] + low_level.scalar_gradient_y_matrix(v, dy)[:,:,k] )/(287*temperature_atmos[:,:,k])
	w_temp[-2:,:,:] = 0
	w_temp[:2,:,:] = 0

	w += w_temp

	return w

cpdef smoothing_3D(np.ndarray a,DTYPE_f smooth_parameter, DTYPE_f vert_smooth_parameter=0.5):
	cdef np.int_t nlat = a.shape[0]
	cdef np.int_t nlon = a.shape[1]
	cdef np.int_t nlevels = a.shape[2]
	smooth_parameter *= 0.5
	cdef np.ndarray test = np.fft.fftn(a)
	test[int(nlat*smooth_parameter):int(nlat*(1-smooth_parameter)),:,:] = 0
	test[:,int(nlon*smooth_parameter):int(nlon*(1-smooth_parameter)),:] = 0
	test[:,:,int(nlevels*vert_smooth_parameter):int(nlevels*(1-vert_smooth_parameter))] = 0
	return np.fft.ifftn(test).real

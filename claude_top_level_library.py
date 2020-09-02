# claude_top_level_library

import claude_low_level_library as low_level
import numpy as np

# laplacian of scalar field a
def laplacian(a):
	output = np.zeros_like(a)
	if output.ndim == 2:
		for i in np.arange(1,nlat-1):
			for j in range(nlon):
				output[i,j] = (scalar_gradient_x_2D(a,i,(j+1)%nlon) - scalar_gradient_x_2D(a,i,(j-1)%nlon))/dx[i] + (scalar_gradient_y_2D(a,i+1,j) - scalar_gradient_y_2D(a,i-1,j))/dy
		return output
	if output.ndim == 3:
		for i in np.arange(1,nlat-1):
			for j in range(nlon):
				for k in range(nlevels-1):
					output[i,j,k] = (scalar_gradient_x(a,i,(j+1)%nlon,k) - scalar_gradient_x(a,i,(j-1)%nlon,k))/dx[i] + (scalar_gradient_y(a,i+1,j,k) - scalar_gradient_y(a,i-1,j,k))/dy + (scalar_gradient_z(a,i,j,k+1)-scalar_gradient_z(a,i,j,k-1))/(2*dz[k])
		return output

# divergence of (a*u) where a is a scalar field and u is the atmospheric velocity field
def divergence_with_scalar(a,u,v,dx,dy):
	output = np.zeros_like(a)
	nlat, nlon, nlevels = output.shape[:]

	au = a*u
	av = a*v

	for i in range(nlat):
		for j in range(nlon):
			for k in range(nlevels):
				output[i,j,k] = low_level.scalar_gradient_x(au,dx,nlon,i,j,k) + low_level.scalar_gradient_y(av,dy,nlat,i,j,k) #+ 0.1*scalar_gradient_z(a*w,i,j,k)
	return output

def radiation_calculation(temperature_world, temperature_atmos, air_pressure, air_density, heat_capacity_earth, albedo, insolation, lat, lon, heights, dz, t, dt, day, year, axial_tilt):
	# calculate change in temperature of ground and atmosphere due to radiative imbalance
	nlat, nlon, nlevels = temperature_atmos.shape[:]	

	upward_radiation = np.zeros(nlevels)
	downward_radiation = np.zeros(nlevels)
	optical_depth = np.zeros(nlevels)
	Q = np.zeros(nlevels)

	for i in range(nlat):
		for j in range(nlon):
			# calculate optical depth
			pressure_profile = air_pressure[i,j,:]
			density_profile = air_density[i,j,:]
			fl = 0.1
			optical_depth = low_level.surface_optical_depth(lat[i])*(fl*(pressure_profile/pressure_profile[0]) + (1-fl)*(pressure_profile/pressure_profile[0])**4)
			
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
				Q[k] = -low_level.scalar_gradient_z_1D(upward_radiation-downward_radiation,dz,0,0,k)/(1E3*density_profile[k])
				# make sure model does not have a higher top than 50km!!
				# approximate SW heating of ozone
				if heights[k] > 20E3:
					Q[k] += low_level.solar(5,lat[i],lon[j],t,day, year, axial_tilt)*((((heights[k]-20E3)/1E3)**2)/(30**2))/(24*60*60)

			temperature_atmos[i,j,:] += Q*dt

			# update surface temperature with shortwave radiation flux
			temperature_world[i,j] += dt*((1-albedo[i,j])*(low_level.solar(insolation,lat[i],lon[j],t, day, year, axial_tilt) + downward_radiation[0]) - upward_radiation[0])/heat_capacity_earth[i,j] 
	
	return temperature_world, temperature_atmos

def velocity_calculation(u,v,air_pressure,old_pressure,air_density,coriolis,gravity,dx,dy,dt):
	# introduce temporary arrays to update velocity in the atmosphere
	u_temp = np.zeros_like(u)
	v_temp = np.zeros_like(v)
	w_temp = np.zeros_like(u)

	nlat,nlon,nlevels = air_pressure.shape[:]

	# calculate acceleration of atmosphere using primitive equations on beta-plane
	for i in np.arange(1,nlat-1):
		for j in range(nlon):
			for k in range(nlevels):
				u_temp[i,j,k] += dt*( -u[i,j,k]*low_level.scalar_gradient_x(u,dx,nlon,i,j,k) - v[i,j,k]*low_level.scalar_gradient_y(u,dy,nlat,i,j,k) + coriolis[i]*v[i,j,k] - low_level.scalar_gradient_x(air_pressure,dx,nlon,i,j,k)/air_density[i,j,k] )
				v_temp[i,j,k] += dt*( -u[i,j,k]*low_level.scalar_gradient_x(v,dx,nlon,i,j,k) - v[i,j,k]*low_level.scalar_gradient_y(v,dy,nlat,i,j,k) - coriolis[i]*u[i,j,k] - low_level.scalar_gradient_y(air_pressure,dy,nlat,i,j,k)/air_density[i,j,k] )
				w_temp[i,j,k] += -(air_pressure[i,j,k]-old_pressure[i,j,k])/(dt*air_density[i,j,k]*gravity)

	u += u_temp
	v += v_temp
	w = w_temp

	# approximate friction
	u *= 0.95
	v *= 0.95

	return u,v,w
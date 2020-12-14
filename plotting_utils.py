import matplotlib.pyplot as plt
import claude_low_level_library_numba as low_level
import numpy as np
import time


def plotter_thread(q, plots, uvw, latlon, data, misc, flags, pole_indices, grid_values):

	lon_plot, lat_plot, heights_plot, lat_z_plot, nplots = plots
	u, v, w = uvw
	lat, lon = latlon
	potential_temperature, pressure_levels, atmosp_addition, temperature_world, t = data
	top, resolution, day = misc
	above, advection, diagnostic, plot, level_plots = flags

	pole_low_index_N, pole_low_index_S = pole_indices
	grid_x_values_N, grid_x_values_S, grid_y_values_N, grid_y_values_S = grid_values


	if plot:
		if not diagnostic:
			# set up plot
			f, ax = plt.subplots(2,figsize=(9,9))
			f.canvas.set_window_title('CLAuDE')
			test = ax[0].contourf(lon_plot, lat_plot, temperature_world, cmap='seismic')
			ax[0].streamplot(lon_plot, lat_plot, u[:,:,0], v[:,:,0], color='white',density=1)
			ax[1].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(low_level.theta_to_t(potential_temperature,pressure_levels),axis=1))[:top,:], cmap='seismic',levels=15)
			# ax[1].contour(heights_plot,lat_z_plot, np.transpose(np.mean(u,axis=1))[:top,:], colors='white',levels=20,linewidths=1,alpha=0.8)
			# ax[1].quiver(heights_plot, lat_z_plot, np.transpose(np.mean(v,axis=1))[:top,:],np.transpose(np.mean(10*w,axis=1))[:top,:],color='black')
			plt.subplots_adjust(left=0.1, right=0.75)
			ax[0].set_title('Surface temperature')
			ax[0].set_xlim(lon.min(),lon.max())
			ax[1].set_title('Atmosphere temperature')
			ax[1].set_xlim(lat.min(),lat.max())
			ax[1].set_ylim((pressure_levels.max()/100,pressure_levels[:top].min()/100))
			ax[1].set_yscale('log')		
			ax[1].set_ylabel('Pressure (hPa)')
			ax[1].set_xlabel('Latitude')
			cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
			f.colorbar(test, cax=cbar_ax)
			cbar_ax.set_title('Temperature (K)')
			f.suptitle( 'Time ' + str(round(t/day,2)) + ' days' )

			if level_plots:

				level_divisions = int(np.floor(nlevels/nplots))
				level_plots_levels = range(nlevels)[::level_divisions][::-1]

				g, bx = plt.subplots(nplots,figsize=(9,8),sharex=True)
				g.canvas.set_window_title('CLAuDE pressure levels')
				for k, z in zip(range(nplots), level_plots_levels):	
					z += 1
					bx[k].contourf(lon_plot, lat_plot, potential_temperature[:,:,z], cmap='seismic')
					bx[k].set_title(str(pressure_levels[z]/100)+' hPa')
					bx[k].set_ylabel('Latitude')
				bx[-1].set_xlabel('Longitude')
		else:
			# set up plot
			f, ax = plt.subplots(2,2,figsize=(9,9))
			f.canvas.set_window_title('CLAuDE')
			ax[0,0].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(u,axis=1))[:top,:], cmap='seismic')
			ax[0,0].set_title('u')
			ax[0,1].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(v,axis=1))[:top,:], cmap='seismic')
			ax[0,1].set_title('v')
			ax[1,0].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(w,axis=1))[:top,:], cmap='seismic')
			ax[1,0].set_title('w')
			ax[1,1].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(atmosp_addition,axis=1))[:top,:], cmap='seismic')
			ax[1,1].set_title('atmosp_addition')
			for axis in ax.ravel():
				axis.set_ylim((pressure_levels.max()/100,pressure_levels[:top].min()/100))
				axis.set_yscale('log')
			f.suptitle( 'Time ' + str(round(t/day,2)) + ' days' )
		
		plt.ion()
		plt.show()
		plt.pause(0.001)

		if not diagnostic:
			ax[0].cla()
			ax[1].cla()
			cbar_ax.cla()
			if level_plots:
				for k in range(nplots):
					bx[k].cla()		
		else:
			ax[0,0].cla()
			ax[0,1].cla()	
			ax[1,0].cla()
			ax[1,1].cla()

	if above:
		g,gx = plt.subplots(1,3, figsize=(15,5))
		plt.ion()
		plt.show()






	def update_plot(uvw, data, reprojections, velocity):
		

		u, v, w = uvw
		potential_temperature, pressure_levels, atmosp_addition, temperature_world, t, north_temperature_data, north_temperature_resample, north_polar_plane_temperature, south_temperature_data, south_temperature_resample, south_polar_plane_temperature = data

		# update plot
		if not diagnostic:

			ax[0].contourf(lon_plot, lat_plot, temperature_world, cmap='seismic',levels=15)
			if velocity:	
				ax[0].streamplot(lon_plot, lat_plot, u[:,:,0], v[:,:,0], color='white',density=0.75)
			ax[0].set_title('$\it{Ground} \quad \it{temperature}$')
			ax[0].set_xlim((lon.min(),lon.max()))
			ax[0].set_ylim((lat.min(),lat.max()))
			ax[0].set_ylabel('Latitude')
			ax[0].axhline(y=0,color='black',alpha=0.3)
			ax[0].set_xlabel('Longitude')

			test = ax[1].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(low_level.theta_to_t(potential_temperature,pressure_levels),axis=1))[:top,:], cmap='seismic',levels=15)
			# test = ax[1].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(potential_temperature,axis=1)), cmap='seismic',levels=15)
			# test = ax[1].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(v,axis=1))[:top,:], cmap='seismic',levels=15)
			if velocity:
				ax[1].contour(heights_plot,lat_z_plot, np.transpose(np.mean(u,axis=1))[:top,:], colors='white',levels=20,linewidths=1,alpha=0.8)
				ax[1].quiver(heights_plot, lat_z_plot, np.transpose(np.mean(v,axis=1))[:top,:],np.transpose(np.mean(10*w,axis=1))[:top,:],color='black')
			ax[1].set_title('$\it{Atmospheric} \quad \it{temperature}$')
			ax[1].set_xlim((-90,90))
			ax[1].set_ylim((pressure_levels.max()/100,pressure_levels[:top].min()/100))
			ax[1].set_ylabel('Pressure (hPa)')
			ax[1].set_xlabel('Latitude')
			ax[1].set_yscale('log')
			f.colorbar(test, cax=cbar_ax)
			cbar_ax.set_title('Temperature (K)')
			f.suptitle( 'Time ' + str(round(t/day,2)) + ' days' )
		
			if level_plots:
				quiver_padding = int(50/resolution)
				skip=(slice(None,None,2),slice(None,None,2))
				for k, z in zip(range(nplots), level_plots_levels):	
					z += 1
					bx[k].contourf(  lon_plot, lat_plot, potential_temperature[:,:,z], cmap='seismic',levels=15)
					bx[k].streamplot(lon_plot, lat_plot, u[:,:,z], v[:,:,z], color='white',density=1.5)
					bx[k].set_title(str(round(pressure_levels[z]/100))+' hPa')
					bx[k].set_ylabel('Latitude')
					bx[k].set_xlim((lon.min(),lon.max()))
					bx[k].set_ylim((lat.min(),lat.max()))				
				bx[-1].set_xlabel('Longitude')		
		else:
			ax[0,0].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(u,axis=1))[:top,:], cmap='seismic')
			ax[0,0].set_title('u')
			ax[0,1].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(v,axis=1))[:top,:], cmap='seismic')
			ax[0,1].set_title('v')
			ax[1,0].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(w,axis=1))[:top,:], cmap='seismic')
			ax[1,0].set_title('w')
			ax[1,1].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(atmosp_addition,axis=1))[:top,:], cmap='seismic')
			ax[1,1].set_title('atmosp_addition')
			for axis in ax.ravel():
				axis.set_ylim((pressure_levels.max()/100,pressure_levels[:top].min()/100))
				axis.set_yscale('log')
			f.suptitle( 'Time ' + str(round(t/day,2)) + ' days' )

		
		
		if above and velocity and advection:
			sample = -2
			gx[0].set_title('Original data')
			gx[1].set_title('Polar plane')
			gx[2].set_title('Reprojected data')

			lat_pole_low = lat[pole_low_index_N:]

			# gx[0].contourf(lon,lat[:pole_low_index_S],south_temperature_data[:,:,sample])
			# gx[1].contourf(grid_x_values_S,grid_y_values_S,south_polar_plane_temperature[:,:,sample])
			# gx[1].quiver(grid_x_values_S,grid_y_values_S,x_dot_S[:,:,sample],y_dot_S[:,:,sample])
			# gx[2].contourf(lon,lat[:pole_low_index_S],south_temperature_resample[:,:,sample])
			# gx[2].quiver(lon[::5],lat[:pole_low_index_S],reproj_u_S[:,::5,sample],reproj_v_S[:,::5,sample])

			gx[0].contourf(lon,lat_pole_low,north_temperature_data[:,:,sample])
			gx[1].contourf(grid_x_values_N,grid_y_values_N,north_polar_plane_temperature[:,:,sample])
			gx[1].quiver(grid_x_values_N,grid_y_values_N,x_dot_N[:,:,sample],y_dot_N[:,:,sample])
			gx[2].contourf(lon,lat_pole_low,north_temperature_resample[:,:,sample])
			gx[2].quiver(lon[::5],lat_pole_low,reproj_u_N[:,::5,sample],reproj_v_N[:,::5,sample])
		
		

	
	for uvw, data, reprojections, velocity in iter(q.get, 'STOP'):
		before_plot = time.time()
		print(f"[QUEUE] Len: {q.qsize()}")

		# clear plots
		# if plot or above:
			# plt.pause(0.01)

		if plot:
			if not diagnostic:
				ax[0].cla()
				ax[1].cla()
				
				if level_plots:
					for k in range(nplots):
						bx[k].cla()			
			else:
				ax[0,0].cla()
				ax[0,1].cla()
				ax[1,0].cla()
				ax[1,1].cla()
		if above:
			gx[0].cla()
			gx[1].cla()
			gx[2].cla()

		update_plot(uvw, data, reprojections, velocity);
		f.canvas.draw()
		f.canvas.flush_events()

		time_taken = float(round(time.time() - before_plot,3))
		print('Threaded Plotting: ',str(time_taken),'s')	

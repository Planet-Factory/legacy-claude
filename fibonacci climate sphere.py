import math, os, sys, math
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, spatial
from matplotlib.patches import Rectangle

def fibonacci_sphere(samples):

    points = []

    index = 0

    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):

        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        golden = phi * i  # golden angle increment

        x = math.cos(golden) * radius
        z = math.sin(golden) * radius

        theta = math.acos(z)
        varphi = np.arctan2(y,x) + np.pi

        if theta > 0.05*np.pi and theta < 0.95*np.pi:
            # points.append((x, y, z))
            points.append((varphi,theta))

    return points

radius = 6.4E6

def field_d_lat(field,lat,lon):
    return field(lat,lon,dtheta=1)[0]/radius
def field_d_lon(field,lat,lon):
    return field(lat,lon,dphi=1)[0]/(radius*np.sin(lat))

def get_temps(ls):
    temps = []
    for item in ls:
        temps.append(item.temp)
    return temps
def get_u(ls):
    output = []
    for item in ls:
        output.append(item.u)
    return output
def get_v(ls):
    output = []
    for item in ls:
        output.append(item.v)
    return output

class pixel:
    
    def __init__(self,lon,lat):
        self.lat = lat
        self.lon = lon 
        self.temp = 270 + 20*np.sin(self.lat)

        self.u = 0
        self.v = 0

        self.heat_capacity = 1E5
        self.albedo = 0.3
        
        self.f = 1E-5*np.cos(self.lat)

    def update_temp(self,dt,solar_constant,sun_lon):    
        self.temp += dt*(
            solar_constant*(1-self.albedo)*max(0,np.sin(self.lat))*max(0,np.sin(self.lon-sun_lon)) 
            - (5.67E-8)*(self.temp**4)
            )/self.heat_capacity

    def update_velocity(self,dt,temp,u,v):
        self.u -= dt*( self.u*field_d_lon(u,self.lat,self.lon) + self.v*field_d_lat(u,self.lat,self.lon) 
            + self.f*self.v + field_d_lon(temp,self.lat,self.lon) )
        self.v -= dt*( self.u*field_d_lon(v,self.lat,self.lon) + self.v*field_d_lat(v,self.lat,self.lon) 
            - self.f*self.u + field_d_lat(temp,self.lat,self.lon) )

    def advect(self,dt,temp,u,v):
        self.temp -= dt*( 
            self.temp*field_d_lon(u,self.lat,self.lon) + self.u*field_d_lon(temp,self.lat,self.lon) +
            self.temp*field_d_lat(v,self.lat,self.lon) + self.v*field_d_lat(temp,self.lat,self.lon) 
            )

#################

points = fibonacci_sphere(1500)
lons = []
lats = []
for point in points:
    lons.append(point[0])
    lats.append(point[1])

print('Fibonacci points calculated')

#################

atmosp = []
for point in points:
    atmosp.append(pixel(point[0],point[1]))

print('Atmosphere constructed')

#################

res = 75

dt = 18*60
day = 60*60*24
year = 365.25*day

lons_grid = np.linspace(0,2*np.pi,2*res)
lats_grid = np.linspace(0,np.pi,res)
lons_grid_gridded,lats_grid_gridded = np.meshgrid(lons_grid,lats_grid)

#################

temps_list = get_temps(atmosp)
u_list = get_u(atmosp)
v_list = get_v(atmosp)

temps = interpolate.SmoothSphereBivariateSpline(lats, lons, temps_list, s=4)
us = interpolate.SmoothSphereBivariateSpline(lats, lons, u_list, s=4)
vs = interpolate.SmoothSphereBivariateSpline(lats, lons, v_list, s=4)

def plotting():

    temps_list = get_temps(atmosp)
    u_list = get_u(atmosp)
    v_list = get_v(atmosp)

    atmosp_temp = temps(lats_grid,lons_grid)
    U = us(lats_grid,lons_grid)
    V = vs(lats_grid,lons_grid)

    quiver_resample = 4
    plt.pcolormesh(lons_grid_gridded,lats_grid_gridded,atmosp_temp)
    plt.gca().add_patch(Rectangle((0,0),2*np.pi,np.pi,linewidth=1,edgecolor='w',facecolor='none'))
    plt.quiver(lons_grid_gridded[::quiver_resample,::quiver_resample],lats_grid_gridded[::quiver_resample,::quiver_resample],U[::quiver_resample,::quiver_resample],V[::quiver_resample,::quiver_resample])
    plt.scatter(lons,lats,s=0.5,color='black')

    plt.xlim((0,2*np.pi))
    plt.ylim((0,np.pi))
    plt.title(str(len(points)+' points'))

    plt.pause(0.01)

    print('T: ',round(atmosp_temp.max()-273.15,1),' - ',round(atmosp_temp.min()-273.15,1),' C')
    print('U: ',round(U.max(),2),' - ',round(U.min(),2),' V: ',round(V.max(),2),' - ',round(V.min(),2))
    
    if np.isinf(atmosp_temp.max()):
        sys.exit()
    if np.isnan(U.max()):
        sys.exit()

plotting()
plt.ion()

sun_lon = 0

time = 0
while True:

    print('TIME: ',str(time/day).zfill(2), "days")

    plt.cla()

    ###

    temps_list = get_temps(atmosp)
    u_list = get_u(atmosp)
    v_list = get_v(atmosp)

    temps = interpolate.SmoothSphereBivariateSpline(lats, lons, temps_list,s=0.5)
    us = interpolate.SmoothSphereBivariateSpline(lats, lons, u_list,s=0.5)
    vs = interpolate.SmoothSphereBivariateSpline(lats, lons, v_list,s=0.5)
    
    # decrease weights of points near the poles?
    # vary smoothing parameter
    # different approach for velocities?

    for point in atmosp:
        point.update_temp(dt,1370,sun_lon)
        point.update_velocity(dt,temps,us,vs)
        point.advect(dt,temps,us,vs)

    ###

    plotting()

    ###

    sun_lon += dt*2*np.pi/day
    time += dt
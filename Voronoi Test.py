import matplotlib.pyplot as plt
import scipy
from scipy.spatial import SphericalVoronoi, geometric_slerp
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import math, os, sys, math

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
            points.append((x, y, z))
            # points.append((varphi,theta))

    return points

numPoints = 50 # More than 50 is very mind melty, this is a UNIT SPHERE
points = np.array(fibonacci_sphere(numPoints))

# points = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0],
#                     [0, 1, 0], [0, -1, 0], [-1, 0, 0]])

radius = 1
center = np.array([0, 0, 0])
points = points * radius
sv = SphericalVoronoi(points, radius, threshold=1e-6)

# sort vertices (optional, helpful for plotting)
sv.sort_vertices_of_regions()

t_vals = np.linspace(0, 1, 2000)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# plot the unit sphere for reference (optional)
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='y', alpha=0.1)
# plot generator points
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b')
# plot Voronoi vertices
ax.scatter(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2], c='g')
# for i, vertex in enumerate(sv.vertices):
#     ax.text(vertex[0], vertex[1], vertex[2], f"{i}")
# indicate Voronoi regions (as Euclidean polygons)
# regionN = 0
for region in sv.regions:
    n = len(region)
    for i in range(n):
        start = sv.vertices[region][i]
        end = sv.vertices[region][(i + 1) % n]
        result = geometric_slerp(start, end, t_vals)
        ax.plot(result[..., 0],
                result[..., 1],
                result[..., 2],
                c=f'k')

    
midPoints = np.array([
    geometric_slerp(sv.vertices[a], sv.vertices[b], 0.5)
    for region in sv.regions
    for (a, b) in zip(region, np.roll(region, 1))
])
ax.scatter(midPoints[..., 0], midPoints[..., 1], midPoints[..., 2], c='k')

ax.azim = 10
ax.elev = 40
_ = ax.set_xticks([])
_ = ax.set_yticks([])
_ = ax.set_zticks([])
fig.set_size_inches(4, 4)

# print(sv.regions)
plt.show()

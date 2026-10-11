# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 06:03:06 2026

@author: oscar
"""


# -*- coding: utf-8 -*-
"""
Created on Mon May 12 16:54:10 2025
@author: oscar
OBS!
Disclaimer: This python code was produced with the help of AI. AI was used for the
following applications:
1. To help the code use less memory (e.g converting to uint8)
2. It taught me the command that i could classify using xr.where,
3. It was used to select the colors of the colormap
"""
import rioxarray as rxr
import xarray as xr
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# _________________________________________ 1. Call data via paths_________________________________________
heightmap_path = r"C:/Users/oscar/GIS/final pro/heighter/Main_hojddata2_mosaik.tif" #height map (raster)
paths_path = r"C:/Users/oscar/GIS/final pro/pathdata/zl_riks.shp" #Walking paths (vector)

# _________________________________________ 2. Open/load datasets _________________________________________
#2a. Heightmap
heightmap = rxr.open_rasterio(heightmap_path, masked=True) #load heightmap with rioxarray
print(heightmap.dims) #check amount of dimensions (only want x,y)
heightmap = heightmap.squeeze(dim="band", drop=True) #drop band dimension
print(heightmap.dtype) #Ensure Datatype is float32 for optimal performance
print(heightmap.rio.crs) #Ensure heightmap has CRS EPSG:3006 (SWEREF 99 TM)

#2b. Walking paths
paths = gpd.read_file(paths_path)
assert paths.crs ==heightmap.rio.crs, "Coordinate systems of data do not match" #make sure both data sets have the same coordinate system

#_________________________________________ 3. Slope _________________________________________
#3a. Compute slope via xarray differentiate
dzdx = heightmap.differentiate(coord='x') #Computes slope in x direction using 1st derivative of heightmap
dzdy = heightmap.differentiate(coord='y') #Computes slope in y direction using 1st derivative of heightmap
slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy))) #Converts slope into degrees (to match reference Avaluator)

#3b. Classify slope
slope_class = xr.where( #creates xarray
    slope > 35, 2, # where every pixel with a slope above 35deg has a value 2
    xr.where((slope >= 30) & (slope <= 35), 1, 0) #where every pixel with a slope between 30 and 35 deg has a value 1, and where everything else has a value 0
).astype(np.uint8) #stores data as 8-bit unassigned integers, do save memory

# _________________________________________ 4. Curvature _________________________________________
#4a. Compute curvature
curv_xx = heightmap.differentiate(coord='x').differentiate(coord='x') #Compute curvature of x using 2nd derivative of height map
curv_yy = heightmap.differentiate(coord='y').differentiate(coord='y') #Compute curvature of y using 2nd derivative of height map
curvature = curv_xx + curv_yy #Laplacian formuka, shows total curvature as a scalar quantity at any given point.

#4b. Compute average curvature by neighbourhood
xres, yres = heightmap.rio.resolution() #Define resolution in both direction
neighbourhood = max(1, int(25 / xres)) #Defines size of region (in terms of pixels) over which the curvature will be calculated. neighbourhood of curvature are used rather than taking the curvature for each individual pixel. This helps smooth out the data to make it useful as a tool, otherwise the data would appear very granular across the map, making it useless.
curv_sm = curvature.rolling(x=neighbourhood, y=neighbourhood, center=True).mean() #computes average curvature over defined neighbourhood

#4c. Classify curvature
curv_class = xr.where(curv_sm > 0, 1, 0).astype(np.uint8)

# _________________________________________ 5. Detect cliffs _________________________________________
#5a. Define elevation difference parameter
elev_max = heightmap.rolling(x=3, y=3, center=True).max() #Define local maximum elevation in a 3x3 pixel neighbourhood
elev_min = heightmap.rolling(x=3, y=3, center=True).min() #Define local minimum elevation in a 3x3 pixel neighbourhood
elev_diff = elev_max - elev_min #Compute elevation difference

#Classify cliffs
cliffs = xr.where((slope > 75) & (elev_diff > 5), 1, 0).astype(np.uint8) #Cliffs have a risk rating of 1. This line defines cliffs as any 3x3 neighbourhood with an average slope of 75 deg as well as a elevation difference (drop) of at least 5 meters

#_________________________________________ 6. Combine risk classifications _________________________________________
avalanche_risk = slope_class + curv_class + cliffs #Sum up all risk classifications into one metric

# _________________________________________ 7. Define colormap _________________________________________
risk_colors = ['#FFFFFF', '#98FB98', '#FFD700', '#FF8C00', '#E60026'] #white, light green, yellow, orange, red
cmap = ListedColormap(risk_colors)
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], len(risk_colors)) #Attribute risk colors to their respective risk level (0,1,2,3,4)

# _________________________________________ 8. Plot map _________________________________________
#8a. Create figure
fig, ax = plt.subplots(figsize=(10, 8)) #Create a new figure and axes

#8b. Plot the risk map
left, bottom, right, top = heightmap.rio.bounds() # Retrieve the exact heightmap bounds for correct extent
im = avalanche_risk.plot.imshow( #plot image
    ax=ax, #define axes
    cmap=cmap, #define color map
    norm=norm, #define attributed risk levels
    extent=(left, right, bottom, top), #Define extent of axes
    add_colorbar=False #supress auto generated colorbar
)

#8c. Create custom colorbar
cbar = plt.colorbar( #produce colorbar
    im,
    ax=ax,
    boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], #Use same boundaries as cmap
    ticks=[0, 1, 2, 3, 4] #define the avalanche risk levels
)
cbar.set_ticklabels(['0', '1', '2', '3', '4']) #write labels
cbar.set_label('Combined Risk Score') #write title for color bar

# 8d. Define contours
min_elev = float(heightmap.min()) #Find max elevation of heightmap
max_elev = float(heightmap.max()) #Find min elevation of heightmap
interval = 100 #Chosen interval between contour lines
levels = np.arange( #construct array for contour levels
    start=np.floor(min_elev / interval) * interval, #multiply rounded down normalized min elevation, to ensure an integrer amount of intervals from min to max
    stop=np.ceil(max_elev / interval) * interval + interval, #multiply rounded up normalized max elevation, to ensure an integrer amount of intervals from min to max. add 1 more interval to ensure we draw all possible contours
    step=interval #define step size
)

#8e. Plot contour lines
cs = heightmap.plot.contour(
    x='x', y='y', #use same coordinates as heightmap
    levels=levels, #define levels
    colors='black',
    linewidths=0.5,
    alpha=0.5, #transparency
    ax=ax, #overlays on same axes
    add_labels=False #surpresses auto labelling
)
ax.clabel(cs, fmt='%d', fontsize=8) #contour labels

#8f. Overlay walking paths & legend
paths.plot(ax=ax, linewidth=1, edgecolor='black', label='Walking Paths') #plot the walking paths

#8g. Plot the legend
ax.legend(loc='upper right', title='Map Features') #plot legend with title

# 8h. Lock the axes to the heightmap extent
ax.set_xlim(left, right)
ax.set_ylim(bottom, top)

#8i. Map title and axes labels
ax.set_title('Avalanche Risk Map of Kebnekaise Area') #give map a title
ax.set_xlabel('Longitude') #x axis labels
ax.set_ylabel('Latitude') #y axis labels
plt.tight_layout() #to make sure no labels or ticks overlap
plt.show() #show plot

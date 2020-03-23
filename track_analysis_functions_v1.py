
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd

# Used for storing tiff movie series and images
import pims

#Used to analyze and extract features from images
from skimage import io
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.util import random_noise

# Used to analyze trajectories
import trackpy as tp
import math

# Used for motion analysis
from scipy.spatial import distance
from numpy import linalg as LA

#Used for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

import warnings
warnings.filterwarnings('ignore')
import pickle

from scipy.ndimage.morphology import distance_transform_edt
import math


# In[5]:


def identifyCellFromTrack(track_df, lab):
    cell_ids = np.unique(lab)
    cell_tracks = {}
    for i in range(len(cell_ids)):
        cell_tracks[i] = []
    
    for index,rows in track_df.iterrows():
        try:
            particle_coord_x = int(np.floor(rows['x']))
            particle_coord_y = int(np.floor(rows['y']))
            label_id = lab[particle_coord_y,particle_coord_x]
            cell_tracks[label_id].append(rows['particle'])
        except:
            pass
    
    all_tracks = []    
    tracks_to_remove = []
    
    for i in range(len(cell_ids)):
        cell_tracks[i] = list(set(cell_tracks[i]))
        all_tracks.extend(cell_tracks[i])
        if i==0:
            tracks_to_remove.extend(cell_tracks[i])
    
    dupes = set([x for x in all_tracks if all_tracks.count(x) > 1])
    tracks_to_remove.extend(dupes)
    tracks_to_remove.extend(np.setdiff1d(list(track_df.particle), all_tracks))
    tracks_to_remove = set(tracks_to_remove)
    new_df = track_df[~track_df['particle'].isin(tracks_to_remove)]
    new_df.index = range(len(new_df))
    new_df['cell_id'] = np.nan
    
    reverse_dict = {}
    for key,val in cell_tracks.items():
        for item in val:
            reverse_dict[item] = key

    for index,rows in new_df.iterrows():
        new_df.loc[index, 'cell_id'] = reverse_dict[rows['particle']]
        reverse_dict[rows['particle']]
    return new_df


# In[6]:


# Method: Generate feature dataframe that is empty
def generate_features_df(track_df):
    unique_tracks = np.array(track_df['particle'].unique())
    new_df = pd.DataFrame(columns=['ID'])
    for row_num, track_id in enumerate(unique_tracks):
        new_df.loc[row_num] = [track_id]
    new_df = new_df.set_index('ID')
    return new_df

# Add motions features to the features dataframe generated from method above
def calc_motion_features(track_df, traj_feature_df):
    new_df = pd.concat([traj_feature_df, 
                        pd.DataFrame(columns=['speed', 'speed_stdev', 'new_speed', 'new_speed_stdev', 'new_speed_range',
                                              'new_speed_min', 'new_speed_max','curvature', 'curv_stdev', 
                                              'net_displacement', 'new_displacement', 'path_length', 'persistance', 
                                              'new_pathlength', 'new_persistance'])])
    unique_tracks = np.array(track_df['particle'].unique())
    
    for row_num, track_id in enumerate(unique_tracks):
        temp_df = track_df.loc[track_df['particle'] == track_id]
        temp_df = temp_df.sort_values(by=['frame'])
        # Compute Vel X, Vel Y, and Speed
        vel_x = np.gradient(temp_df['x'])
        vel_y = np.gradient(temp_df['y'])
        velocity = np.transpose(np.array([vel_x, vel_y]))
        speed = [np.sqrt(dx**2 + vel_y[i]**2) for i,dx in enumerate(vel_x)]
        
        # Compute Normalized Speed
        new_vel_x = np.gradient(temp_df['new_x'])
        new_vel_y = np.gradient(temp_df['new_y'])
        framerate = np.array(temp_df['delta_t'])
        new_velocity = np.transpose(np.array([new_vel_x, new_vel_y]))
        new_speed = [np.sqrt(dx**2 + new_vel_y[i]**2) for i,dx in enumerate(new_vel_x)]
        new_speed = np.divide(new_speed, framerate)
                              
        # Compute the Acceleration Vector
        #acc_x = np.gradient(vel_x)
        #acc_y = np.gradient(vel_y)
        #acceleration = np.transpose(np.array([acc_x, acc_y]))

        # Compute Unit Norm and Unit Tangent Vectors 
        #unit_tangent = np.multiply(np.transpose(np.repeat([np.divide(1,speed)],2, axis=0)),velocity)

        #norm = np.transpose(np.array([np.gradient(unit_tangent[:,0]), np.gradient(unit_tangent[:,1])]))
        #norm_magnitude = [np.linalg.norm(vector) for vector in norm]
        #unit_norm = np.multiply(np.transpose(np.repeat([np.divide(1,norm_magnitude)],2, axis=0)),norm)

        # Compute Kurvature
        #curv = np.divide(norm_magnitude, speed)
        
        #Compute Curvature again
        ind_to_remove = np.logical_and(np.isnan(temp_df['x']), np.isnan(temp_df['y']))
        temp_x = temp_df['x'][~ind_to_remove]
        temp_y = temp_df['y'][~ind_to_remove]
        t = np.arange(0,len(temp_x))
    
        z_x = np.polyfit(t, temp_x, 2)
        z_y = np.polyfit(t, temp_y, 2)
        z_dx = np.polyder(z_x)
        z_ddx = np.polyder(z_dx)
        z_dy = np.polyder(z_y)
        z_ddy = np.polyder(z_dy)
        
        f_dx = np.poly1d(z_dx)
        f_ddx = np.poly1d(z_ddx)
        f_dy = np.poly1d(z_dy)
        f_ddy = np.poly1d(z_ddy)
        
        t2 = np.linspace(1, len(temp_x)-2, 10)

        dx = f_dx(t2)
        ddx = f_ddx(t2)
        dy = f_dy(t2)
        ddy = f_ddy(t2)
        
        curv = np.divide(np.sqrt((ddy*dx - ddx*dy)**2),np.power((dx**2 + dy**2),1.5))
        #curv = curv[1:-1]

        #dx = np.gradient(temp_df['x'])
        #dy = np.gradient(temp_df['y'])
        #ddx = np.gradient(dx)
        #ddy = np.gradient(dy)
        
        #curv = np.sqrt(np.divide(((ddy*dx - ddx*dy)**2),np.power((dx**2 + dy**2),1.5)))
        
        # Compute Net Displacement
        first_val = temp_df.iloc[0]
        last_val = temp_df.iloc[len(temp_df)-1]
        net_displacement = np.sqrt(np.power(last_val['x'] - first_val['x'],2) + np.power(last_val['y'] - first_val['y'],2))
        new_displacement = np.sqrt(np.power(last_val['new_x'] - first_val['new_x'],2) + np.power(last_val['new_y'] - first_val['new_y'],2))
        
        # Compute Total Path length
        dx_squared = np.power(np.diff(temp_df['x']),2)
        dy_squared = np.power(np.diff(temp_df['y']),2)
        path_length = sum(np.sqrt(dx_squared+dy_squared))

        #Compute New Path Length
        new_dx_squared = np.power(np.diff(temp_df['new_x']),2)
        new_dy_squared = np.power(np.diff(temp_df['new_y']),2)
        new_pathlength = sum(np.sqrt(new_dx_squared+new_dy_squared))

        # Compute Persistence
        persistance = np.divide(net_displacement, path_length)    
        new_persistance = np.divide(new_displacement, new_pathlength)
        # Filter outliers of Acceleration/Speed
        #speed_sorted = sorted(speed)
        #speed_sorted_filtered = speed_sorted[math.floor(len(speed)/5):len(speed)-math.floor(len(speed)/5)]

        #curv_sorted = sorted(curv)
        #curv_sorted_filtered = curv_sorted[math.floor(len(curv)/5):len(curv)-math.floor(len(curv)/5)]

        # Compute Average Acceleration/Speed
        avg_speed = np.average(speed)
        stdev_speed = np.std(speed)
        avg_curv = np.average(curv)
        stdev_curv = np.std(curv)
        new_avg_speed = np.average(new_speed)
        new_stdev_speed = np.std(new_speed)
                              
        # Add values to dataframe
        new_df.loc[track_id]['speed'] = avg_speed
        new_df.loc[track_id]['speed_stdev'] = stdev_speed
        new_df.loc[track_id]['new_speed'] = new_avg_speed
        new_df.loc[track_id]['new_speed_stdev'] = new_stdev_speed
        new_df.loc[track_id]['curvature'] = avg_curv
        new_df.loc[track_id]['curv_stdev'] = stdev_curv
        new_df.loc[track_id]['net_displacement'] = net_displacement
        new_df.loc[track_id]['path_length'] = path_length
        new_df.loc[track_id]['persistance'] = persistance
        new_df.loc[track_id]['new_displacement'] = new_displacement
        new_df.loc[track_id]['new_pathlength'] = new_pathlength
        new_df.loc[track_id]['new_persistance'] = new_persistance
        new_df.loc[track_id]['new_speed_range'] = np.max(new_speed) - np.min(new_speed)
        new_df.loc[track_id]['new_speed_min'] = np.min(new_speed)
        new_df.loc[track_id]['new_speed_max'] = np.max(new_speed)
    return new_df

# Add the particle mass feature to the features dataframe 
def calc_mass_features(track_df, traj_feature_df):
# Parse CSV file to find amplitude average + stdev
    unique_tracks = np.array(track_df['particle'].unique())
    new_df = pd.concat([traj_feature_df, 
                        pd.DataFrame(columns=['avg_mass', 'stdev_mass'])])

    for i, track_id in enumerate(unique_tracks):
        temp_df = track_df.loc[track_df['particle'] == track_id]

        mass = np.array(temp_df['mass'])
        mass_sorted = sorted(mass)
        mass_sorted_filtered = mass_sorted[math.floor(len(mass)/5):len(mass)-math.floor(len(mass)/5)]

        mass_avg = np.average(mass_sorted_filtered)
        mass_stdev = np.std(mass_sorted_filtered)

        new_df.loc[track_id]['avg_mass'] = mass_avg
        new_df.loc[track_id]['stdev_mass'] = mass_stdev
    return new_df


# In[7]:


#Function Filters the track matrix (ie. t1,t2) using features
def filter_track_from_feat(track_df, feature_df, feature_name, min_val=-np.inf, max_val=np.inf):
    tracks_to_remove = []
    for index,vals in feature_df.iterrows():
        if (vals[feature_name] < min_val or vals[feature_name]>max_val):
            tracks_to_remove.append(index)
    #print ("deleting following tracks:" , tracks_to_remove)
    new_df = track_df[~track_df['particle'].isin(tracks_to_remove)]
    return new_df    
    
#Function filteres the trajectory feature matrix using features
def filter_traj_from_feat(feature_df, feature_name, min_val=-np.inf, max_val=np.inf):
    tracks_to_remove = []
    for index,vals in feature_df.iterrows():
        if (vals[feature_name] < min_val or vals[feature_name]>max_val):
            tracks_to_remove.append(index)
    #print ("deleting following tracks:" , tracks_to_remove)
    new_df = feature_df[~feature_df.index.isin(tracks_to_remove)]
    return new_df


# In[8]:


def dist(x1,y1, x2,y2, x3,y3): # x3,y3 is the point
    px = x2-x1
    py = y2-y1
    something = px*px + py*py
    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(something)
    if u > 1:
        u = 1
    elif u < 0:
        u = 0
    x = x1 + u * px
    y = y1 + u * py
    dx = x - x3
    dy = y - y3
    dist = math.sqrt(dx*dx + dy*dy)
    return dist


# In[9]:


# Distance Functions
# Gets the distance between the axis and a specific track
def getDistanceFromAxis(track_df, track_id,  cell_ID=1, axis='major', label_img=None,):
    props = regionprops(label_img)[cell_ID-1]
    y1, x1 = props.centroid
    orientation = props.orientation
    x2 = np.nan
    y2 = np.nan
    if axis=='major':
        x2 = x1 + math.cos(orientation) * 0.5 * props.major_axis_length
        y2 = y1 - math.sin(orientation) * 0.5 * props.major_axis_length
    else:
        x2 = x1 - math.sin(orientation) * 0.5 * props.minor_axis_length
        y2 = y1 - math.cos(orientation) * 0.5 * props.minor_axis_length
        
    temp_df = track_df.loc[track_df['particle'] == track_id]
    distances = []
    for index,rows in temp_df.iterrows():
        x0 = rows['x']
        y0 = rows['y']

        num = np.absolute(np.multiply(y2-y1, x0) - np.multiply(x2-x1, y0)+ np.multiply(x2,y1) - np.multiply(y2,x1))
        den = np.sqrt(np.power(y2-y1,2) + np.power(x2-x1,2))
        distance = np.divide(num,den)
        distances.append(distance)
        
    return min(distances)


#Gets the signed distance between an axis and a specific cell (MIGHT NOT NEED THIS)
def getsignedDistanceFromAxis(track_df, track_id, label_img=None, cell_ID=1, axis='major'):
    props = regionprops(label_img)[cell_ID-1]
    y1, x1 = props.centroid
    orientation = props.orientation
    x2 = np.nan
    y2 = np.nan
    if axis=='major':
        x2 = x1 + math.cos(orientation) * 0.5 * props.major_axis_length
        y2 = y1 - math.sin(orientation) * 0.5 * props.major_axis_length
    else:
        x2 = x1 - math.sin(orientation) * 0.5 * props.minor_axis_length
        y2 = y1 - math.cos(orientation) * 0.5 * props.minor_axis_length
        
    temp_df = track_df.loc[track_df['particle'] == track_id]
    distances = []
    for index,rows in temp_df.iterrows():
        x0 = rows['x']
        y0 = rows['y']
        #print(index,x0,y0)

        num = np.multiply(y2-y1, x0) - np.multiply(x2-x1, y0)+ np.multiply(x2,y1) - np.multiply(y2,x1)
        den = np.sqrt(np.power(y2-y1,2) + np.power(x2-x1,2))
        distance = np.divide(num,den)
        distances.append(distance)
        
    return min(distances)

# Compute Distance of a track from the center of the cell
def getDistanceFromCenter(track_df, track_id, cell_ID=1, label_img=None):
    distances = [] 
    y0, x0 = regionprops(img_label)[cell_ID-1].centroid
    centroid = np.array([x0,y0])
    temp_df = track_df.loc[track_df['particle'] == track_id]
    for index,rows in temp_df.iterrows():
        object_coord = np.array([rows['x'],rows['y']])
        try:
            dst = distance.euclidean(object_coord, centroid)
        except:
            dst = np.inf
        distances.append(dst)
    return min(distances)

# Function Identifies nearby tracks given the track dataframe, a track of interest, and a search radius
def getDistanceFromTrack(track_df, track_id, poi_id):
    poi_df = track_df.loc[track_df['particle'] == poi_id]
    temp_df = track_df.loc[track_df['particle'] == track_id]
    distances = []
    for index,rows in temp_df.iterrows():
        object_coord = np.array([rows['new_x'],rows['new_y']])
        for index_p, rows_p in poi_df.iterrows():
            poi_coord = np.array([rows_p['new_x'], rows_p['new_y']])
            try:
                dst = distance.euclidean(object_coord, poi_coord)

            except:
                dst = np.inf
            
            distances.append(dst)            
    return min(distances)


# In[10]:


def getNearbyTrackstoObject(track_df, func, cell_ID=1, min_val = 0, max_val=30, **kwargs):
    nearby_tracks = []
    unique_tracks = np.array(track_df['particle'].unique())
    val = np.infty
    remove=False
    for track_id in unique_tracks:
        if 'axis' in kwargs:
            val = func(track_df, track_id, cell_ID, axis=kwargs['axis'], label_img=kwargs['label'])
        elif 'poi_id' in kwargs:
            val = func(track_df, track_id, poi_id=kwargs['poi_id'])
            remove=True
        else:
            val = func(track_df, track_id, cell_ID, label_img=kwargs['label'])
        if val <= max_val and val >= min_val:
            nearby_tracks.append(track_id)
    if remove == True:
        nearby_tracks.remove(kwargs['poi_id'])
    return nearby_tracks

def createTrackDict(track_df, min_search=10, max_search=200, step_size=30, *track_dict, **kwargs):
    return_dict = {}
    if track_dict:
        return_dict = track_dict
    
    if 'label' in kwargs:
        for i in range(min_search,max_search,step_size):
            print(i, end=" ")
            temp_tracks = getNearbyTrackstoObject(t3, max_val=i, **kwargs)
            return_dict[i] = temp_tracks
    else:    
        unique_tracks = np.array(track_df['particle'].unique())   
        for track_id in unique_tracks:
            return_dict[track_id] = {}
            print(track_id, end=" ")
            for i in range(min_search,max_search,step_size):
                temp_tracks = getNearbyTrackstoObject(t3, poi_id= track_id, max_val=i, **kwargs)
                return_dict[track_id][i] = temp_tracks
    return return_dict

def create_distance_matrix(track_df, unique_tracks):
    return_mat = np.zeros([len(unique_tracks), len(unique_tracks)])
    for i, track_id in enumerate(unique_tracks):
        for j, track_id_2 in enumerate(unique_tracks):
            if i == j:
                continue
            if return_mat[i,j] != 0:
                continue
            else:
                dst = getDistanceFromTrack(track_df, track_id_2, track_id)
                return_mat[i, j] = dst
                return_mat[j, i] = dst
    return return_mat

#dist_mat = create_distance_matrix(temp_cell_df, np.array(temp_cell_df['particle'].unique()))


# In[11]:


# Function computes the cosine distance between the track of interest and other tracks specified in an array
def compute_cosine_distance(track_df, poi_id, nearby_tracks):
    # Compute the Cosine Distance to find differences in orientation between all these tracks
    cosine_distances = []
    displacment_vector = []
    
    if isinstance(poi_id, (np.ndarray)):
        displacment_vector = poi_id
    else:
        poi_df = track_df.loc[track_df['particle'] == poi_id]
        #Compute Displacement vector of the particle of interest
        min_particle = poi_df.loc[poi_df['frame'].argmin()]
        max_particle = poi_df.loc[poi_df['frame'].argmax()]
        min_coord = np.array([min_particle['x'], min_particle['y']])
        max_coord = np.array([max_particle['x'], max_particle['y']])
        displacment_vector = np.subtract(max_coord, min_coord)        
    
    for particle_id in nearby_tracks:
        temp_df = track_df.loc[track_df['particle'] == particle_id]
        min_particle = temp_df.loc[temp_df['frame'].argmin()]
        max_particle = temp_df.loc[temp_df['frame'].argmax()]
        min_coord = np.array([min_particle['x'], min_particle['y']])
        max_coord = np.array([max_particle['x'], max_particle['y']])
        disp_vector_2 = np.subtract(max_coord, min_coord)

        cos_dist = np.divide(np.dot(displacment_vector, disp_vector_2), np.multiply(LA.norm(displacment_vector), LA.norm(disp_vector_2)))
        cosine_distances.append([cos_dist, particle_id])
    return cosine_distances

def getOrientationFromAxis(track_df, track_id, cell_ID=1, label_img=None, axis='major'):
    props = regionprops(label_img)[cell_ID-1]
    y0, x0 = props.centroid
    orientation = props.orientation
    
    axis_vec = np.nan
    if axis=='major':
        x1 = x0 + math.cos(orientation) * 0.5 * props.major_axis_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.major_axis_length
        axis_vec = np.array([x1-x0, y1-y0])
    else:
        x2 = x0 - math.sin(orientation) * 0.5 * props.minor_axis_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.minor_axis_length
        axis_vec = np.array([x2-x0, y2-y0])

    cosine_distance = compute_cosine_distance(nearby_tracks=[track_id], track_df=track_df, poi_id=axis_vec)
    return cosine_distance[0][0]

def getOrientationTracksAxis(track_df, min_val=0, max_val=1, cell_ID=1, label_img=None, axis='major'):
    nearby_tracks = []
    unique_tracks = np.array(track_df['particle'].unique())
    for track_id in unique_tracks:
        val = getOrientationFromAxis(track_df, track_id, cell_ID, label_img, axis='major')
        if val <= max_val and val >= min_val:
            nearby_tracks.append(track_id)
    return nearby_tracks


# In[ ]:





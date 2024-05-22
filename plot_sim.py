import matplotlib.animation
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_opening, generate_binary_structure
from scipy.io import savemat
from scipy import ndimage
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.filters import threshold_otsu
from tqdm import tqdm
import torch
import h5py
import os
from skimage.transform import resize

def make_maze_map(file,N_universe=1000,N_trials=10):
    #Ensure PyTorch is using GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()


    leaf_image = imread(file, as_gray=True)

    if leaf_image.ndim == 3:
        leaf_image = rgb2gray(leaf_image)
    threshold = threshold_otsu(leaf_image)
    binary_mask = leaf_image > threshold
    binary_mask = binary_opening(binary_mask, structure=generate_binary_structure(2, 2))

    pad_height = (N_universe - binary_mask.shape[0])
    pad_width = (N_universe - binary_mask.shape[1])

    # Adjust padding for even distribution
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # Apply padding
    maze_map = np.pad(binary_mask, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=1)

    # Ensure the map size is exactly N_universe x N_universe, handling any potential off-by-one errors
    if maze_map.shape[0] > N_universe:
        maze_map = maze_map[:N_universe, :]
    if maze_map.shape[1] > N_universe:
        maze_map = maze_map[:, :N_universe]
    # Convert maze_map to a PyTorch tensor and repeat it along a new axis for N_trials
    maze_map = np.tile(maze_map[..., np.newaxis], N_trials)
    return maze_map

def make_maze_map_procedural(N_universe=1000,N_trials=10):
    #Ensure PyTorch is using GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    maze_map_out = torch.zeros((N_universe, N_universe, N_trials), device=device)
    
    for i in range(N_trials):
        file = f'leaf_database/0006_{i+1:04d}.JPG'


        leaf_image = imread(file)
        if leaf_image.ndim == 3:
            leaf_image = leaf_image[:,:,1] # green only
        
        #downsample
        height, width = leaf_image.shape
        if height > width:
            new_height = N_universe
            new_width = int((new_height / height) * width)
        else:
            new_width = N_universe
            new_height = int((new_width / width) * height)
        downsampled_image = resize(leaf_image, (new_height, new_width), anti_aliasing=True)

        #make mask
        threshold = threshold_otsu(downsampled_image)
        binary_mask = downsampled_image > threshold
        binary_mask = binary_opening(binary_mask, structure=generate_binary_structure(2, 2))
        if binary_mask[round(binary_mask.shape[0]/2),round(binary_mask.shape[1]/2)] == 1:
            binary_mask = -1*binary_mask+1
        pad_height = (N_universe - binary_mask.shape[0])
        pad_width = (N_universe - binary_mask.shape[1])

        # Adjust padding for even distribution
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # Apply padding
        maze_map = np.pad(binary_mask, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=1)

        # Ensure the map size is exactly N_universe x N_universe, handling any potential off-by-one errors
        if maze_map.shape[0] > N_universe:
            maze_map = maze_map[:N_universe, :]
        if maze_map.shape[1] > N_universe:
            maze_map = maze_map[:, :N_universe]
        # Convert maze_map to a PyTorch tensor and repeat it along a new axis for N_trials
        maze_map_out[:,:,i] = torch.tensor(maze_map, device=device) #B fill with torch
    return maze_map_out




def subject_animation(sensor_recording,angle_recording,step_intent_recording,\
                      step_action_recording,position_recording,maze_map,trial_index = 0,animation_filename='sim.gif',T_used = -1):
    
    #angle_recording = np.cumsum(angle_recording, axis=0)
    if T_used != -1:
        T = T_used
    else:
        T = sensor_recording.shape[1]
    subject_radius_discrete = 30
    touch_radius_discrete = 35
    N_sensors = sensor_recording.shape[0]
    print(maze_map.shape)
    maze_map = maze_map.cpu().detach().numpy()
    N_universe = 1000
    plt.rcParams["animation.html"] = "jshtml"
    plt.rcParams['figure.dpi'] = 150  
    plt.ioff()
    fig, ax = plt.subplots()


    def animate(t,angle=0):
        pos = position_recording[:, t, trial_index]
        angle = angle_recording[t, trial_index]
        end_pos_x = pos[0] + subject_radius_discrete * np.cos(angle)
        end_pos_y = pos[1] + subject_radius_discrete * np.sin(angle)
        plt.cla()
        #plt.xlim(0,10)
        ax.imshow(maze_map[:, :, trial_index], cmap='gray') 
        circle = plt.Circle((pos[0], pos[1]), subject_radius_discrete, color='red', fill=False)
        ax.add_artist(circle)
        circle2 = plt.Circle((pos[0], pos[1]), touch_radius_discrete, color='crimson', fill=False)
        ax.add_artist(circle2)
        ax.plot([pos[0], end_pos_x], [pos[1], end_pos_y], color='cyan')  # Line from center to perimeter
        ax.set_aspect('equal', adjustable='datalim')

        #Avatar representation
        for i in range(N_sensors):
            angle_range = 360 / N_sensors
            color = plt.cm.viridis(sensor_recording[i, t, trial_index])  # Map the sensor value to a color
            wedge = Wedge(center=(N_universe*1.3, N_universe*0.5), r=touch_radius_discrete*4, 
                        theta1=i*angle_range, theta2=(i+1)*angle_range, 
                        color=color)
            ax.add_patch(wedge)
        text_angle = np.radians(0)  # Midpoint of wedge in radians
        text_x = N_universe*1.3 - touch_radius_discrete* (4+2)   # Positioning text slightly inside the wedge
        text_y = N_universe*0.5
        ax.text(text_x, text_y, f'Front', ha='center', va='center', color='blue', fontsize=8)
        ax.axis('off')



    anim = matplotlib.animation.FuncAnimation(fig, animate, frames=tqdm(range(T), desc='Animating'))
    writergif = matplotlib.animation.PillowWriter(fps=20) 
    anim.save(animation_filename, writer=writergif)


# def subject_path(sensor_recording,angle_recording,step_intent_recording,\
#                       step_action_recording,position_recording,maze_map,trial_index = 0,figure_filename='path_x.jpg',T_used = -1):
    
#     angle_recording = np.cumsum(angle_recording, axis=0)
#     if T_used != -1:
#         T = T_used
#     else:
#         T = sensor_recording.shape[1]
#     subject_radius_discrete = 30
#     touch_radius_discrete = 35
#     N_sensors = sensor_recording.shape[0]
#     print(maze_map.shape)
#     maze_map = maze_map.cpu().detach().numpy()
#     N_universe = 1000

#     fig, ax = plt.subplots()

#     path = position_recording[:, :T, trial_index]
#     touched_points = position_recording[:, :T, trial_index]
#     touched_points[:,:] = np.nan
#     predicted_points[:,:] = np.nan


#     for i in range(T):
#         pos = position_recording[:, t, trial_index]
#         angle = angle_recording[t, trial_index]
#         end_pos_x = pos[0] + subject_radius_discrete * np.cos(angle)
#         end_pos_y = pos[1] + subject_radius_discrete * np.sin(angle)
        
#         #plt.xlim(0,10)
#         ax.imshow(maze_map[:, :, trial_index], cmap='gray') 
#         circle = plt.Circle((pos[0], pos[1]), subject_radius_discrete, color='red', fill=False)
#         ax.add_artist(circle)
#         circle2 = plt.Circle((pos[0], pos[1]), touch_radius_discrete, color=[1,0.3,0.3], fill=False)
#         ax.add_artist(circle2)
#         ax.plot([pos[0], end_pos_x], [pos[1], end_pos_y], color='blue')  # Line from center to perimeter
#         ax.set_aspect('equal', adjustable='datalim')

#         #Avatar representation
#         for i in range(N_sensors):
#             angle_range = 360 / N_sensors
#             color = plt.cm.viridis(sensor_recording[i, t, trial_index])  # Map the sensor value to a color
#             wedge = Wedge(center=(N_universe*1.3, N_universe*0.5), r=touch_radius_discrete*4, 
#                         theta1=i*angle_range, theta2=(i+1)*angle_range, 
#                         color=color)
#             ax.add_patch(wedge)
#         text_angle = np.radians(0)  # Midpoint of wedge in radians
#         text_x = N_universe*1.3 - touch_radius_discrete* (4+2)   # Positioning text slightly inside the wedge
#         text_y = N_universe*0.5
#         ax.text(text_x, text_y, f'Front', ha='center', va='center', color='blue', fontsize=8)



#     anim = matplotlib.animation.FuncAnimation(fig, animate, frames=tqdm(range(T), desc='Animating'))
#     writergif = matplotlib.animation.PillowWriter(fps=20) 
#     fig.save(figure_filename)




    
def pull_from_file(file):
    with np.load(file) as data:
        sensor_recording = data['sensor_recording']
        angle_recording = data['angle_recording']
        step_intent_recording = data['step_intent_recording']
        step_action_recording = data['step_action_recording']
        position_recording = data['position_recording']
        orientation_recording = data['orientation_recording']

    return angle_recording, step_intent_recording,\
          step_action_recording, sensor_recording, position_recording,orientation_recording


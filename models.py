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
from skimage.transform import resize
def fill_circle_in_grid(x, y, radius, x_grid, y_grid):
    r = radius**2
    return ((x_grid - x)**2 + (y_grid - y)**2 <= r).int()

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

def world_sim(T=250,N_trials=100,input_file = 'leaf.jpg',output_name = 'simulation_data'):

    # Ensure PyTorch is using GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # Initialize simulation parameters
    #T = 50
    #N_trials = 10
    leaf_image = imread(input_file, as_gray=True)
    N_universe = 1000

    # Subject + touch definition
    subject_radius_discrete = 30
    touch_width_discrete = 5
    touch_radius_discrete = subject_radius_discrete + touch_width_discrete

    # Number of sensors and their angular edges
    N_sensors = 64
    sensor_edges = np.linspace(-np.pi, np.pi, N_sensors + 1)
    # Convert sensor edges to torch tensor
    sensor_edges = torch.from_numpy(sensor_edges).float().to(device)
    # sensor_edges = sensor_edges.unsqueeze(1).repeat(1, N_trials)

    # Universe/Object definition using meshgrid
    x_grid_universe_np, y_grid_universe_np = np.meshgrid(np.arange(1, N_universe + 1), np.arange(1, N_universe + 1))
    x_grid_universe = torch.from_numpy(np.tile(x_grid_universe_np[..., np.newaxis], N_trials)).float().to(device)
    y_grid_universe = torch.from_numpy(np.tile(y_grid_universe_np[..., np.newaxis], N_trials)).float().to(device)

    layer_numb = torch.arange(N_trials,device=device).view(1, 1, N_trials).expand(N_universe, N_universe, N_trials)
    layer_numb_flat = layer_numb.view(N_universe**2, N_trials)
    # Image processing for maze map
    # if leaf_image.ndim == 3:
    #     leaf_image = rgb2gray(leaf_image)
    # threshold = threshold_otsu(leaf_image)
    # binary_mask = leaf_image > threshold#angles
    # binary_mask = binary_opening(binary_mask, structure=generate_binary_structure(2, 2))

    # pad_height = (N_universe - binary_mask.shape[0])
    # pad_width = (N_universe - binary_mask.shape[1])

    # # Adjust padding for even distribution
    # pad_top = pad_height // 2
    # pad_bottom = pad_height - pad_top
    # pad_left = pad_width // 2
    # pad_right = pad_width - pad_left

    # # Apply padding
    # maze_map = np.pad(binary_mask, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=1)

    # # Ensure the map size is exactly N_universe x N_universe, handling any potential off-by-one errors
    # if maze_map.shape[0] > N_universe:
    #     maze_map = maze_map[:N_universe, :]
    # if maze_map.shape[1] > N_universe:
    #     maze_map = maze_map[:, :N_universe]
    # # Convert maze_map to a PyTorch tensor and repeat it along a new axis for N_trials
    # maze_map = torch.from_numpy(np.tile(maze_map[..., np.newaxis], N_trials)).float().to(device)
    maze_map = make_maze_map_procedural(N_universe=N_universe,N_trials=N_trials)


    # Simulation variables, converted to torch tensors and moved to GPU
    sensor_recording = torch.zeros((N_sensors, T, N_trials), device=device)
    angle_recording = torch.zeros((T, N_trials), device=device)
    orientation_recording = torch.zeros((T, N_trials), device=device)
    step_intent_recording = torch.zeros((T, N_trials), device=device)
    step_action_recording = torch.zeros((T, N_trials), device=device)
    position_recording = torch.zeros((2, T, N_trials), device=device)

    # Get valid subject starting location
    x_subject_discrete = 500 * torch.ones(N_trials, device=device)
    y_subject_discrete = 500 * torch.ones(N_trials, device=device)
    x_subject_old  = 500 * torch.ones(N_trials, device=device)
    y_subject_old  = 500 * torch.ones(N_trials, device=device)
    ang = torch.zeros(N_trials, device=device)#torch.rand(N_trials, device=device)*2*np.pi - np.pi
    intersection = torch.zeros(N_trials, dtype=torch.bool)



    # Simulation loop
    for t in tqdm(range(T), desc='Simulation'):
        # intent
        mag = torch.round(5 * torch.rand(N_trials, device=device))
        ang_intent = (torch.rand(N_trials, device=device) - 0.5)
        #ang_intent = torch.randn(N_trials, device=device) * 0.25
        ang += ang_intent
        orientation_recording[t, :] = ang
        angle_recording[t, :] = ang_intent
        step_intent_recording[t, :] = mag

        # save old positions
        x_subject_old = x_subject_discrete.clone()
        y_subject_old = y_subject_discrete.clone()
        a = x_subject_old[:]

        # provisional step
        x_subject_discrete += torch.round(mag * torch.cos(ang))
        y_subject_discrete += torch.round(mag * torch.sin(ang))

        # Update masks
        subject_mask = fill_circle_in_grid(x_subject_discrete, y_subject_discrete, subject_radius_discrete, x_grid_universe, y_grid_universe)
        skin_mask = fill_circle_in_grid(x_subject_discrete, y_subject_discrete, touch_radius_discrete, x_grid_universe, y_grid_universe)
        skin_mask = torch.sub(skin_mask,subject_mask)
        layer_numb_flat = layer_numb.view(N_universe**2, N_trials)

        # intention -> action
        intersection = torch.mul(subject_mask, maze_map).sum(dim=(0, 1))
        intersection = intersection.gt(0).int() 
        intersection_flipped = -1*intersection +1
        x_subject_old = torch.mul(x_subject_old , intersection)
        y_subject_old = torch.mul(y_subject_old , intersection)
        x_subject_discrete = torch.mul(x_subject_discrete, intersection_flipped)
        y_subject_discrete = torch.mul(y_subject_discrete, intersection_flipped)
        x_subject_discrete = torch.add(x_subject_discrete, x_subject_old)
        y_subject_discrete = torch.add(y_subject_discrete, y_subject_old)

        #x_subject_discrete= torch.add(torch.mul(x_subject_old , intersection),torch.mul(x_subject_discrete, intersection_flipped))
        #y_subject_discrete = torch.add(torch.mul(y_subject_old , intersection) + torch.mul(y_subject_discrete, intersection_flipped))
        mag = torch.mul(mag, intersection_flipped)
        # Update masks
        subject_mask = fill_circle_in_grid(x_subject_discrete, y_subject_discrete, subject_radius_discrete, x_grid_universe, y_grid_universe)
        skin_mask = fill_circle_in_grid(x_subject_discrete, y_subject_discrete, touch_radius_discrete, x_grid_universe, y_grid_universe)
        skin_mask = torch.sub(skin_mask,subject_mask)

        # Update positions
        position_recording[0, t, :] = x_subject_discrete
        position_recording[1, t, :] = y_subject_discrete
        step_action_recording[t, :] = mag

        # Calculate touch information
        touch_mask = skin_mask * maze_map
        x_diff = x_grid_universe - x_subject_discrete
        y_diff = y_grid_universe - y_subject_discrete
        angles = (torch.atan2(y_diff, x_diff) - ang)*touch_mask
        angles = torch.atan2(torch.sin(angles), torch.cos(angles))
        distances = torch.sqrt(x_diff**2 + y_diff**2)
        touch = ((touch_radius_discrete - subject_radius_discrete) - (distances - subject_radius_discrete)) * touch_mask
        touch = touch / abs(touch_radius_discrete - subject_radius_discrete)


        sensor_matrix = torch.bucketize(angles, sensor_edges, right=False) - 1
        depth_flat= touch.view(N_universe**2, N_trials)
        angle_flat = sensor_matrix.view(N_universe**2, N_trials)
        spaced_angle_depth_flat = angle_flat*10 + depth_flat
        spaced_angle_depth_flat, indices = torch.sort(spaced_angle_depth_flat, dim=0, descending=True)
        sorted_angle_big = torch.gather(angle_flat*10, 0, indices)
        spaced_angle_depth_flat = spaced_angle_depth_flat-sorted_angle_big
        sorted_layer_numb = torch.gather(layer_numb_flat , 0, indices)
        
        sorted_angle = torch.gather(angle_flat, 0, indices)
    

        # Shift data to create two versions: one shifted up and one shifted down
        data_shifted_up = torch.roll(spaced_angle_depth_flat, shifts=1, dims=0)
        data_shifted_down = torch.roll(spaced_angle_depth_flat, shifts=-1, dims=0)

        # Ensure that the first and last elements do not compare incorrectly
        data_shifted_up[0, :] = float('-inf')  # Make the first row of the upward shifted matrix -inf
        data_shifted_down[-1, :] = float('-inf')  # Make the last row of the downward shifted matrix -inf

        # Find local maxima
        is_local_max = (spaced_angle_depth_flat > data_shifted_up) & (spaced_angle_depth_flat > data_shifted_down)
        #print(spaced_angle_depth_flat[is_local_max], sorted_angle[is_local_max], sorted_layer_numb[is_local_max])
        sensor_recording[sorted_angle[is_local_max], t, sorted_layer_numb[is_local_max]] = spaced_angle_depth_flat[is_local_max]
        # for sensor in range(N_sensors):
        #     relevant_angles = sensor_matrix==sensor
        #     sensor_data = torch.max((touch * relevant_angles).flatten(start_dim=0, end_dim=1), dim=0)[0]
        #     sensor_recording[sensor, t, :] = sensor_data


    back_to_cpu = True
    if back_to_cpu == True:
        sensor_recording = sensor_recording.cpu()
        angle_recording = angle_recording.cpu()
        orientation_recording = orientation_recording.cpu()
        step_intent_recording = step_intent_recording.cpu()
        step_action_recording = step_action_recording.cpu()
        position_recording = position_recording.cpu()
        maze_map = maze_map.cpu()

    data = {
        'sensor_recording': sensor_recording,
        'angle_recording': angle_recording,
        'step_intent_recording': step_intent_recording,
        'step_action_recording': step_action_recording,
        'position_recording': position_recording,
        'orientation_recording': orientation_recording
    }
    #savemat(output_name+'.mat', data, do_compression=True, format='5')
    np.savez_compressed(output_name + '.npz', **data)


    
import torch
import torch.nn as nn
import numpy as np
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using', DEVICE)

class TemporalPrediction (nn.Module):
    def __init__ (self, hidden_units, L1, device,N_sensors=64, T_test = 100,include_motor=True):
        super(TemporalPrediction, self).__init__() #inherit nn.module properties
        if include_motor ==True:
            N_in = N_sensors+4
            N_out = N_sensors+1
        else:
            N_in = N_sensors
            N_out = N_sensors
        self.L1 = L1
        self.device = device
        
        self.hidden = nn.Linear(in_features=(N_in)*(T_test-1), out_features=hidden_units)
        self.sigmoid = nn.Sigmoid()#nn.Softplus()#
        self.fc = nn.Linear(in_features=hidden_units, out_features=N_out)
        

    def forward (self, inputs):
        return self.fc(self.sigmoid(self.hidden(inputs)))

    def L1_regularisation (self):
        weights = torch.empty(0, device=self.device)
        for name, params in self.named_parameters():
            if 'weight' in name:
                weights = torch.cat((weights, params.flatten()), 0)
        return self.L1*weights.abs().sum()

    def loss_fn (self, x, y):  
        loss_l1 = self.L1_regularisation()
        loss_mse = nn.functional.mse_loss(x, y)
        loss = loss_mse + loss_l1
        return loss, loss_mse, loss_l1


class SimpleRNN(nn.Module):
    def __init__(self, hidden_size, num_layers,N_sensors,L1,DEVICE=DEVICE,include_motor=True):
        super(SimpleRNN, self).__init__()
        if include_motor ==True:
            N_in = N_sensors+4
            N_out = N_sensors+1
        else:
            N_in = N_sensors
            N_out = N_sensors
        self.device = DEVICE
        self.L1 = L1
        self.rnn = nn.RNN(N_in, hidden_size, num_layers)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(in_features=hidden_size, out_features=N_out)

    def forward(self, input_data, DEVICE=DEVICE, hidden=None, standard= True):
        if standard==True:
            if hidden is None:
                batch_size = input_data.size(1)
                hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=DEVICE)
                hidden_rec = torch.zeros(self.num_layers, batch_size, input_data.size(0), self.hidden_size, device=DEVICE)
            output, hidden = self.rnn(input_data, hidden)
            output = self.sigmoid(output)  # Apply sigmoid to the last RNN output
            output = self.fc(output[-1])  # Apply the fully connected layer to the last output
            return output, hidden_rec
        else:
            batch_size = input_data.size(1)
            seq_len = input_data.size(0)

            if hidden is None:
                hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=DEVICE)
            hidden_rec = torch.zeros(seq_len, self.num_layers, batch_size, self.hidden_size, device=DEVICE)
            outputs = []
            current_input = input_data
            current_hidden = hidden
            for t in range(seq_len):
                output, current_hidden = self.rnn(current_input[t:t+1], current_hidden)
                outputs.append(self.sigmoid(self.fc(output)))
                hidden_rec[t] = current_hidden
            final_output = torch.cat(outputs, dim=0)
            return final_output, hidden_rec
        
    def L1_regularisation (self):
        weights = torch.empty(0, device=self.device)
        for name, params in self.named_parameters():
            if 'weight' in name:
                weights = torch.cat((weights, params.flatten()), 0)
        return self.L1*weights.abs().sum()

    def loss_fn(self, x, y):
        loss_l1 = self.L1_regularisation()
        loss_mse = nn.functional.mse_loss(x, y)
        loss = loss_mse + loss_l1
        return loss, loss_mse, loss_l1
def pull_from_file(file,ignore_or = False):
    with np.load(file) as data:
        sensor_recording = data['sensor_recording']
        angle_recording = data['angle_recording']
        step_intent_recording = data['step_intent_recording']
        step_action_recording = data['step_action_recording']
        position_recording = data['position_recording']
        angle_left_recording = np.maximum(angle_recording,0)
        angle_right_recording = np.maximum(-angle_recording,0)
        if ignore_or == False:
            orientation_recording = data['orientation_recording']
            return angle_right_recording, angle_left_recording, step_intent_recording,\
                step_action_recording, sensor_recording, position_recording,orientation_recording
        else:
            return angle_right_recording, angle_left_recording, step_intent_recording,\
                step_action_recording, sensor_recording, position_recording
def extract_subsequences(data, T_test,sensor=False,single_world = False):
    """
    Extracts all subsequences of length T_test from the data array.
    data: numpy array of shape (trials, time_points, ...) where ... can be additional dimensions.
    T_test: length of the subsequences to extract.
    """
    subsequences = []
    extra_d = 0
    if sensor ==True:
        extra_d = 1
    if single_world ==False:
        num_trials = data.shape[1+extra_d]
    else: 
        num_trials = 1
    num_time_points = data.shape[0+extra_d]
    # Calculate number of possible subsequences per trial
    num_subsequences = num_time_points - T_test + 1

    # Loop over each trial and extract subsequences
    for trial in range(num_trials):
        for start in range(num_subsequences):
            end = start + T_test
            if sensor == True:
                subsequence = data[:,start:end,trial]
            else:
                subsequence = data[start:end,trial]
            subsequences.append(subsequence)
    #print(len(subsequences))
    # Reshape to have shape (num_trials * num_subsequences, T_test, ...)
    subsequences = np.array(subsequences)

    if sensor==True:
        subsequences = np.transpose(subsequences,  (0, 2, 1))
        return subsequences
    else:
        subsequences = np.expand_dims(subsequences, -1)
        return subsequences
    

def frame_selection (x,y,include_motor=True,N_sensors = 6,batch_size = 1000):
    shuffle_indices = np.random.permutation(np.arange(x.shape[0]))
    #    Apply the same shuffle to both arrays
    x = x[shuffle_indices]
    y = y[shuffle_indices]

    x = torch.Tensor(x).to(DEVICE)
    y = torch.Tensor(y).to(DEVICE)    
    if include_motor == True:
        for batch_n in range(0, x.shape[0], batch_size):
            clip_past_frames = x[batch_n:batch_n+batch_size, :-1,:]
            clip_next_frame  = y[batch_n:batch_n+batch_size, -1,:]
            clip_past_frames = clip_past_frames.contiguous().view(clip_past_frames.shape[0], -1)
            yield (clip_past_frames, clip_next_frame)
    else:
        for batch_n in range(0, x.shape[0], batch_size):
            clip_past_frames = x[batch_n:batch_n+batch_size, :-1,:(N_sensors)]
            clip_next_frame  = y[batch_n:batch_n+batch_size, -1,:(N_sensors)]
            clip_past_frames = clip_past_frames.contiguous().view(clip_past_frames.shape[0], -1)
            yield (clip_past_frames, clip_next_frame)    
def frame_selection_RNN (x,y,include_motor=True,N_sensors = 64,batch_size = 1000):
    shuffle_indices = np.random.permutation(np.arange(x.shape[0]))
    #    Apply the same shuffle to both arrays
    x = x[shuffle_indices]
    y = y[shuffle_indices]

    x = torch.Tensor(x).to(DEVICE)
    y = torch.Tensor(y).to(DEVICE)    
    if include_motor == True:
        for batch_n in range(0, x.shape[0], batch_size):
            clip_past_frames = x[batch_n:batch_n+batch_size, :-1,:]
            clip_next_frame  = y[batch_n:batch_n+batch_size, -1,:]
            clip_past_frames = torch.transpose(clip_past_frames, 0,1)
            yield (clip_past_frames, clip_next_frame)
    else:
        for batch_n in range(0, x.shape[0], batch_size):
            clip_past_frames = x[batch_n:batch_n+batch_size, :-1,:(N_sensors)]
            clip_next_frame  = y[batch_n:batch_n+batch_size, -1,:(N_sensors)]
            clip_past_frames = torch.transpose(clip_past_frames, 0,1)
            yield (clip_past_frames, clip_next_frame)    
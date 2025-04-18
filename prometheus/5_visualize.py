import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Define the model architecture (must be the same as the saved model)
class ConvLSTMCell(nn.Module):
    # ... (Implementation same as in the training script)
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,  # 4 for the 4 gates (i, f, c, o)
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # Concatenate along channel axis
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_c, cc_o = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        c_next = f * c_cur + i * torch.tanh(cc_c)
        o = torch.sigmoid(cc_o)
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    # ... (Implementation same as in the training script)
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            hidden_state = hidden_state
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class SimpleConvLSTM(nn.Module):
    # ... (Implementation same as in the training script)
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size, num_layers, num_conv_layers=1):
        super(SimpleConvLSTM, self).__init__()

        # Initial convolutional layers
        conv_layers = []
        conv_layers.append(nn.Conv2d(input_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2))
        conv_layers.append(nn.ReLU())
        for _ in range(num_conv_layers - 1):
            conv_layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2))
            conv_layers.append(nn.ReLU())
        self.conv_layers = nn.Sequential(*conv_layers)

        # ConvLSTM layers
        self.convlstm = ConvLSTM(
            input_dim=hidden_dim,
            hidden_dim=[hidden_dim] * num_layers,
            kernel_size=(kernel_size, kernel_size),
            num_layers=num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )

        # Output convolutional layer to match the desired output shape (1 time step)
        self.output_conv = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, x):
        # x shape: (batch_size, time_steps, channels, height, width)

        # Process each time step with the initial convolutional layers
        batch_size, time_steps, channels, height, width = x.size()
        x = x.view(batch_size * time_steps, channels, height, width)
        x = self.conv_layers(x)
        x = x.view(batch_size, time_steps, -1, height, width)

        # Pass the processed sequence to ConvLSTM
        x, _ = self.convlstm(x)

        # Select only the last time step's output
        x = x[0][:, -1, :, :, :]  # Take the output from the last layer and the last time step

        # Reshape for the output convolutional layer
        x = x.view(batch_size, -1, height, width)
        x = self.output_conv(x)

        return x

# Instantiate the model
input_dim = 1
hidden_dim = 64
output_dim = 1
kernel_size = 3
num_layers = 2
model = SimpleConvLSTM(input_dim, hidden_dim, output_dim, kernel_size, num_layers)

# Load the saved weights
model.load_state_dict(torch.load('best_model.pth'))
model.eval()  # Set the model to evaluation mode

# Load the test data
# Assuming you have your data loaded as NumPy arrays:
# labels_np: (4200, 7, 5, 5)
# features_np: (4200, 7, 5, 5, 5)

# Prepare Data
labels_tensor = torch.load('../model_data/labels_all.pt', weights_only=False).float()
features_tensor = torch.load('../model_data/features_all.pt', weights_only=False).float()

X = features_tensor[:, :4, :, :, :]
y = labels_tensor[:, 4, :, :].unsqueeze(1)  # Target: 5th time step

# Split Data - Use the same random state as in training!
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a DataLoader for the test set
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # No need to shuffle for visualization

# Function to visualize predictions
def visualize_predictions(model, data_loader, num_samples=4):
    model.eval()
    
    # Filter samples to ensure at least one fire event in the ground truth
    valid_samples = []
    with torch.no_grad():
      for features, labels in data_loader:
          mask = labels.sum(dim=(1, 2, 3)) > 0  # Check if there's at least one fire pixel
          valid_features = features[mask]
          valid_labels = labels[mask]

          if valid_features.size(0) > 0:
              outputs = model(valid_features)
              predicted = (torch.sigmoid(outputs)).float()
              valid_samples.extend(list(zip(valid_features, valid_labels, predicted)))
          
          if len(valid_samples) >= num_samples:
              break

    # Visualize the filtered samples
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 2 * num_samples))
    for i in range(num_samples):
        features, ground_truth, prediction = valid_samples[i]

        ground_truth_map = ground_truth[0].cpu().numpy()  # Remove channel dimension
        prediction_map = prediction[0].cpu().numpy()  # Remove channel dimension

        # Ground Truth (Discrete)
        axes[i, 0].imshow(ground_truth_map, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        axes[i, 0].set_title('Ground Truth')
        axes[i, 0].set_xticks([])  # Remove axis ticks
        axes[i, 0].set_yticks([])

        # Prediction (Discrete)
        axes[i, 1].imshow(prediction_map, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        axes[i, 1].set_title('Prediction')
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

    plt.tight_layout()
    plt.show()

# Visualize some sample predictions
visualize_predictions(model, test_loader, num_samples=8)
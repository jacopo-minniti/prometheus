import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class ConvLSTMCell(nn.Module):
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
    

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


labels_tensor = torch.load('../model_data/labels_all.pt', weights_only=False).float()
features_tensor = torch.load('../model_data/features_all.pt', weights_only=False).float()

X = features_tensor[:, :3, :, :, :]
y = labels_tensor[:, 3, :, :].unsqueeze(1)  # Target: 4th time step

# 2. Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Remove extra dimension from y_train and y_test
# y_train = y_train.unsqueeze(1)  # Add a channel dimension
# y_test = y_test.unsqueeze(1)    # Add a channel dimension

# 3. Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 4. Model, Loss, and Optimizer
input_dim = 1  # Number of feature channels
hidden_dim = 64
output_dim = 1
kernel_size = 3
num_layers = 2
model = SimpleConvLSTM(input_dim, hidden_dim, output_dim, kernel_size, num_layers)

criterion = FocalLoss(alpha=0.25, gamma=2)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)

# 5. Training Loop
num_epochs = 10
best_val_loss = float('inf')  # Initialize best validation loss for saving the best model

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Validation (assuming you have a validation set)
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for val_features, val_labels in test_loader:  # Use test_loader for simplicity
            val_outputs = model(val_features)
            val_loss += criterion(val_outputs, val_labels).item()
        val_loss /= len(test_loader)

        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Model saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")

# Load the Best Model
model.load_state_dict(torch.load('best_model.pth'))

# Sample Prediction and Visualization
model.eval()
with torch.no_grad():
    # Get a sample batch from the test set
    features, labels = next(iter(test_loader))
    features, labels = features, labels
    outputs = model(features)
    print(len(outputs))
    predicted = (torch.sigmoid(outputs > 0.5)).float()
    # Choose a sample from the batch to visualize
    sample_idx = 0
    for i in range(3):
        prediction_map = predicted[i, 0, :, :].cpu().numpy()  # Predicted map
        ground_truth_map = labels[i, 0, :, :].cpu().numpy()  # Ground truth map

        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Ground Truth
        im1 = axes[0].imshow(ground_truth_map, cmap='hot', interpolation='nearest')
        axes[0].set_title('Ground Truth')
        fig.colorbar(im1, ax=axes[0])

        # Prediction
        im2 = axes[1].imshow(prediction_map, cmap='hot', interpolation='nearest')
        axes[1].set_title('Prediction')
        fig.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        plt.show()

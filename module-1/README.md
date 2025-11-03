# Module 1: Introduction to PyTorch

This module introduces the PyTorch deep learning framework, covering its dynamic computation graph, basic building blocks like neurons and layers, and the typical workflow for developing machine learning models.

Early frameworks required defining the entire model structure upfront and couldn't use normal Python control flow. PyTorch addressed this issue.
￼
A single neuron is a linear equation with weight (W) and bias (b).
￼
Even with multiple inputs a single neuron will still be a linear equation with weights corresponding to each parameter and a bias value.
￼
1. Gather raw data - Get raw dataset for ingestion 
2. Data prep - Clean the dataset to fix any errors, missing values etc, Transforming the data to different formats, engineering new features (like converting an address to a distance) etc.
3. Modeling 
4. Training the model with training set 
5. Evaluation of model using test set
6. Deployment

```python
import torch                 # core functionality of pytorch
import torch.nn as nn.       # components for building neural networks
import torch.optim as optim  # tools fr training those networks
distances = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
times = torch.tensor([[6.96], [12.11], [16.77], [22.21]], dtype=torch.float32)
# Define the model
model = nn.Sequential(nn.Linear(1, 1))
# Define the loss function and the optimizer
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

Activation Function
Adding activation function to the output of a neuron converts the linear equation to a non-linear one. This enables the neurons to learn patterns which are non-linear.
￼
E.g. `nn.ReLU()`

```python
model = nn.Sequential(nn.Linear(1, 1), nn.ReLU())
```

Adding a ReLU layer in the model as shown above.
￼
First linear layer takes 1 input and gives 5 outputs
ReLU layer just does the transform and the last linear layer then taken the 5 inputs and produces the final 1 output.

Code: https://github.com/akrivalabs/pytorch-course/blob/main/module-1/C1M1_Assignment.ipynb

```python
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Load the dataset from the CSV file
file_path = './data_with_features.csv'
data_df = pd.read_csv(file_path)

def rush_hour_feature(hours_tensor, weekends_tensor):
    """
    Engineers a new binary feature indicating if a delivery is in a weekday rush hour.

    Args:
        hours_tensor (torch.Tensor): A tensor of delivery times of day.
        weekends_tensor (torch.Tensor): A tensor indicating if a delivery is on a weekend.

    Returns:
        torch.Tensor: A tensor of 0s and 1s indicating weekday rush hour.
    """

    ### START CODE HERE ###
    
    # Define rush hour and weekday conditions
    is_morning_rush = (hours_tensor >= 8.0) & (hours_tensor < 10.0)
    is_evening_rush = (hours_tensor >= 16.0) & (hours_tensor < 19.0)
    is_weekday = weekends_tensor == 0

    # Combine the conditions to create the final rush hour mask
    is_rush_hour_mask = is_weekday & (is_morning_rush | is_evening_rush)

    ### END CODE HERE ###

    # Convert the boolean mask to a float tensor to use as a numerical feature
    return is_rush_hour_mask.float()


def prepare_data(df):
    """
    Converts a pandas DataFrame into prepared PyTorch tensors for modeling.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing the raw delivery data.

    Returns:
        prepared_features (torch.Tensor): The final 2D feature tensor for the model.
        prepared_targets (torch.Tensor): The final 2D target tensor.
        results_dict (dict): A dictionary of intermediate tensors for testing purposes.
    """

    # Extract the data from the DataFrame as a NumPy array
    # (There's no direct torch.from_dataframe(), so we use .values to get a NumPy array first)
    all_values = df.values

    ### START CODE HERE ###

    # Convert all the values from the DataFrame into a single PyTorch tensor
    full_tensor = torch.tensor(all_values, dtype=torch.float32)

    # Use tensor slicing to separate out each raw column
    raw_distances = full_tensor[:, 0]
    raw_hours = full_tensor[:, 1]
    raw_weekends = full_tensor[:, 2]
    raw_targets = full_tensor[:, 3]

    # Call your rush_hour_feature() function to engineer the new feature
    is_rush_hour_feature = rush_hour_feature(raw_hours, raw_weekends)

    # Use the .unsqueeze(1) method to reshape the four 1D feature tensors into 2D column vectors
    distances_col = raw_distances.unsqueeze(1)
    hours_col = raw_hours.unsqueeze(1)
    weekends_col = raw_weekends.unsqueeze(1)
    rush_hour_col = is_rush_hour_feature.unsqueeze(1)

    ### END CODE HERE ###

    # Normalize the continuous feature columns (distance and time)
    dist_mean, dist_std = distances_col.mean(), distances_col.std()
    hours_mean, hours_std = hours_col.mean(), hours_col.std()
 
    distances_norm = (distances_col - dist_mean) / dist_std
    hours_norm = (hours_col - hours_mean) / hours_std

    # Combine all prepared 2D features into a single tensor
    prepared_features = torch.cat([
        distances_norm,
        hours_norm,
        weekends_col,
        rush_hour_col
    ], dim=1) # dim=1 concatenates them column-wise, stacking features side by side

    # Prepare targets by ensuring they are the correct shape
    prepared_targets = raw_targets.unsqueeze(1)
    
    # Dictionary for Testing Purposes
    results_dict = {
        'full_tensor': full_tensor,
        'raw_distances': raw_distances,
        'raw_hours': raw_hours,
        'raw_weekends': raw_weekends,
        'raw_targets': raw_targets,
        'distances_col': distances_col,
        'hours_col': hours_col,
        'weekends_col': weekends_col,
        'rush_hour_col': rush_hour_col
    }
    

    return prepared_features, prepared_targets, results_dict

# Process the entire DataFrame to get the final feature and target tensors.
features, targets, _ = prepare_data(data_df)

def init_model():
    """
    Initializes the neural network model, optimizer, and loss function.

    Returns:
        model (nn.Sequential): The initialized PyTorch sequential model.
        optimizer (torch.optim.Optimizer): The initialized optimizer for training.
        loss_function: The initialized loss function.
    """

    # Set the random seed for reproducibility of results (DON'T MANIPULATE IT)
    torch.manual_seed(41)

    ### START CODE HERE ###

    # Define the model architecture using nn.Sequential
    model = nn.Sequential(
        # Input layer (Linear): 4 input features, 64 output features
        nn.Linear(4, 64),
        # First ReLU activation function
        nn.ReLU(),
        # Hidden layer (Linear): 64 inputs, 32 outputs
        nn.Linear(64, 32),
        # Second ReLU activation function
        nn.ReLU(),
        # Output layer (Linear): 32 inputs, 1 output (the prediction)
        nn.Linear(32, 1)
    ) 
    
    # Define the optimizer (Stochastic Gradient Descent)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Define the loss function (Mean Squared Error for regression)
    loss_function = nn.MSELoss()

    ### END CODE HERE ###

    return model, optimizer, loss_function

model, optimizer, loss_function = init_model()

def train_model(features, targets, epochs, verbose=True):
    """
    Trains the model using the provided data for a number of epochs.
    
    Args:
        features (torch.Tensor): The input features for training.
        targets (torch.Tensor): The target values for training.
        epochs (int): The number of training epochs.
        verbose (bool): If True, prints training progress. Defaults to True.
        
    Returns:
        model (nn.Sequential): The trained model.
        losses (list): A list of loss values recorded every 5000 epochs.
    """
    
    # Initialize a list to store the loss
    losses = []
    
    ### START CODE HERE ###
    
    # Initialize the model, optimizer, and loss function using `init_model`
    model, optimizer, loss_function = init_model()

    # Loop through the specified number of epochs
    for epoch in range(epochs):
        
        # Forward pass: Make predictions
        outputs = model(features)

        # Calculate the loss
        loss = loss_function(outputs, targets)

        # Zero the gradients
        optimizer.zero_grad()

        # Backward pass: Compute gradients
        loss.backward()

        # Update the model's parameters
        optimizer.step()
    
    ### END CODE HERE ### 

        # Every 5000 epochs, record the loss and print the progress
        if (epoch + 1) % 5000 == 0:
            losses.append(loss.item())
            if verbose:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    return model, losses

# Training loop
model, loss = train_model(features, targets, 30000)

# Disable gradient calculation for efficient predictions
with torch.no_grad():
    # Perform a forward pass to get model predictions
    predicted_outputs = model(features)

# Change the values below to get an estimate for a different delivery
# Set distance for the delivery in miles
distance_miles = 10 
# Set time of day in 24-hour format (e.g., 9.5 for 9:30 AM)
time_of_day_hours = 15
# Use True/False or 1/0 to indicate if it's a weekend
is_weekend = 1

# Convert the raw inputs into a 2D tensor for the model
raw_input_tensor = torch.tensor([[distance_miles, time_of_day_hours, is_weekend]], dtype=torch.float32)
helper_utils.prediction(model, data_df, raw_input_tensor, rush_hour_feature)
```

```bash

+------------------------------------------+-----------------------+
|                         Model Prediction                         |
+------------------------------------------+-----------------------+
| Time of the Week                         | Weekend               |
| Distance                                 | 10.0 miles            |
| Time                                     | 15:00                 |
| Is this considered a rush hour period?   | No                    |
+------------------------------------------+-----------------------+
| Estimated Delivery Time                  | 27.50 minutes         |
+------------------------------------------+-----------------------+
```

Note: https://github.com/akrivalabs/pytorch-course/blob/main/module-1/helper_utils.py#L517 has the prediction() helper_utils function defined.

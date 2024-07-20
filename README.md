Logistic Regression with Mini-Batch Gradient Descent using PyTorch
This repository contains a project that demonstrates how to implement Logistic Regression using Mini-Batch Gradient Descent in PyTorch. The objective is to train a logistic regression model on a sample dataset, visualize the loss surface, and understand the effect of different learning rates and batch sizes on model training.

Table of Contents
Introduction
Dependencies
Dataset
Model
Training
Visualization
Results
Usage
Conclusion
Introduction
Logistic Regression is a simple yet powerful classification algorithm used for binary classification problems. In this project, we implement logistic regression using PyTorch and train it using Mini-Batch Gradient Descent. We also visualize the loss surface and the training process to understand the model's behavior.

Dependencies
Python 3.x
numpy
matplotlib
PyTorch
torchvision
torchaudio
Install the required packages using:
pip install torch torchvision torchaudio numpy matplotlib

Dataset
We use a synthetic dataset with one feature (x) and a binary target variable (y). The dataset is defined in the Data class, which inherits from torch.utils.data.Dataset. The x values range from -1 to 1, and the y values are set to 1 for x > 0.2 and 0 otherwise.

Model
The logistic regression model is defined in the logistic_regression class, which inherits from torch.nn.Module. The model consists of a single linear layer followed by a sigmoid activation function.

Training
The model is trained using Mini-Batch Gradient Descent with the following steps:

Initialize the model, loss criterion (Binary Cross-Entropy Loss), and optimizer (Stochastic Gradient Descent).
Set the batch size and learning rate.
Train the model for a specified number of epochs, updating the model parameters based on the loss and gradients computed for each batch.
Visualize the loss surface and parameter updates during training.
Visualization
The plot_error_surfaces class is used to visualize the loss surface and the parameter space during training. It provides 3D and contour plots of the loss surface, as well as data space plots showing the model's decision boundary.

Results
After training, the model's weights and bias are printed, and the accuracy on the training data is computed. The cost vs. iteration graph is plotted to show the convergence of the training process.

Usage
To train the model with a learning rate of 0.01, 120 epochs, and a batch size of 1, run the following code:
# Create the plot_error_surfaces object
get_surface = plot_error_surfaces(15, 13, data_set[:][0], data_set[:][1], 30)

# Train the Model
model = logistic_regression(1)
criterion = nn.BCELoss()
trainloader = DataLoader(dataset=data_set, batch_size=1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
epochs = 120
loss_values = []

for epoch in range(epochs):
    for x, y in trainloader:
        yhat = model(x)
        loss = criterion(yhat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        get_surface.set_para_loss(model, loss.tolist())
        loss_values.append(loss)
    if epoch % 20 == 0:
        get_surface.plot_ps()

# Print final weights and bias
w = model.state_dict()['linear.weight'].data[0]
b = model.state_dict()['linear.bias'].data[0]
print("w = ", w, "b = ", b)

# Compute accuracy
yhat = model(data_set.x)
yhat = torch.round(yhat)
correct = 0
for prediction, actual in zip(yhat, data_set.y):
    if prediction == actual:
        correct += 1
print("Accuracy: ", correct / len(data_set) * 100, "%")

# Plot Cost vs Iteration
LOSS_BGD1 = [i.item() for i in loss_values]
plt.plot(LOSS_BGD1)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()


This project provides a hands-on example of training a logistic regression model using Mini-Batch Gradient Descent in PyTorch. By visualizing the loss surface and the training process, we gain insights into the behavior of the model and the optimization process. This approach can be extended to more complex models and datasets for practical applications in machine learning and deep learning.

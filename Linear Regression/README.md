## Deciding Initialization
- **Motivation:** Getting all things needed for computation in place.
- Parameters for computation:
  - Weights
  - Bias
  - learning_rate
  - n_iters
  - batch_size

## Loading Dataset
- **Motivation:** Getting the dataset in place
- Checks made:
  - X & y should be compatible shapes
  - X & y should be numpy arrays -> else convert them to numpy  
  - Handling empty X & y arrays

## Initialization
- **Motivation:** Init the weights and bias while giving the user the choice of initialization method
- Methods of initialization:
  - Zero : Basic initialization that can be used for baseline or for debugging. It would work fine, due to the convex nature of linear regression as an algorithm.
  - Random : Would provide a non-biased starting point for linear regression
  - Small-random: Ideal for normalized, and scaled datasets. Can help prevent large initial values that could potentially slow down convergence or destabilize training.

## Dimensions of the different parameters
- **Motivation:** This subheading is purely for solving my confusion regarding the dimensions of the X, y, W, B matrices.
- X.shape = (n_samples, n_features)
- y.shape = (n_samples, n_outputs)
- W.shape = (n_outputs, n_features)
- B.shape = (1, n_outputs)
- When we do the linear regression operation by the following:  
  
  &nbsp;&nbsp;&nbsp;&nbsp;$h(x) = W \cdot X^T + B$  

  the transformation that happens will be as follows:

  = (n_outputs, n_features) * (batch_size, n_features)^T +  (n_outputs, 1) <br>
  = (n_outputs, n_features) * (n_features, batch_size) +  (n_outputs, 1) <br>
  = (n_outputs, batch_size) + (n_outputs, 1) <br>
  = (n_outputs, batch_size)

## Loss Functions
- **Mean Squared Error** - Used for penalizing larger mistakes heavily but this loss function is, sensitive to outliers.
- **Mean Absolute Error** - Treats all errors equally regardless of the magnitude and is more robust at handling outliers.
- **Root Mean Squared Error** - Similar to the MSE but expressed in the same units as the target variable.

## Updating Weights and Biases
- Based on the chosen loss function, the weights and biases are updated. <br>
    &nbsp;&nbsp;&nbsp;&nbsp;W = W - learning_rate * ∂J/∂W <br>
 &nbsp;&nbsp;&nbsp;&nbsp;B = B - learning_rate * ∂J/∂B <br>

## Gradient Descent
- Determined based on batch_size
  -   batch_size == 1 : Stochastic Gradient Descent
  -   1 < batch_size < n_samples : Mini-batch Gradient Descent
  -   batch_size == n_samples : Batch Gradient Descent
 
## Fit Method 
- Brings all the elements together and monitors the loss value for each epoch


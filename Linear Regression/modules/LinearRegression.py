import numpy as np
import random

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000, batch_size = None, init_method='zero', loss_function = 'mse' ):
      self.learning_rate = learning_rate
      self.n_iters = n_iters
      self.W = None
      self.B = None
      self.init_method = init_method
      self.batch_size = batch_size
      self.loss_function = loss_function

    def load_dataset(self, X, y):
        '''
        X: (m, n) -> m = no. of samples, n = no. of features
        y: (m, 1) -> m = no. of samples
        '''
        try:
          if not isinstance(X, np.ndarray) or not  isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrrays")
          if X.size == 0 or y.size == 0:
            raise ValueError("X and y cannot be empty")
          if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible dimensions")

          if len(y.shape)== 1:
            y = y.reshape(-1, 1)

          self.X = X
          self.y = y
          self.n_samples, self.n_features = self.X.shape
          self.n_outputs = y.shape[1]

          if not self.batch_size:
            self.batch_size = self.X.shape[0]

        except Exception as e:
          print(f"Error in load_dataset: {str(e)}")


    def initialize_parameters(self):
      if not hasattr(self, 'X') or not hasattr(self, 'y'):
        raise ValueError("X and y must be loaded before initializing parameters")

      if not self.X.any() or not self.y.any():
        raise ValueError("X and y must be loaded before initializing parameters")

      if self.init_method == 'zero':
        self.W = np.zeros((self.n_outputs, self.n_features))
        self.B = np.zeros((self.n_outputs,1))
      elif self.init_method == 'random':
        self.W = np.random.rand(self.n_outputs, self.n_features)
        self.B = np.random.rand(self.n_outputs,1)
      elif self.init_method == 'small_random':
        self.W = np.random.rand(self.n_outputs, self.n_features)* 0.01
        self.B = np.random.rand(self.n_outputs,1)*0.01
      else:
        raise ValueError(f"Unsupported initialization method: {self.init_method}")


    def predict(self, X):
      '''
      X: (m, batch_size) -> m = no. of samples, n = no. of features
      '''
      # print("self.W.shape: ", self.W.shape)
      # print("X.T.shape:", X.T.shape) 
      # print("self.B.shape: ",self.B.shape )
      return np.dot(X, self.W.T) + self.B

    def compute_loss(self, y, y_pred, loss_function=None):
      try:
        if y.shape != y_pred.shape:
          raise ValueError(f"y.shape : {y.shape} and y_pred.shape: {y_pred.shape} have incompatible dimensions.")
        
        if self.loss_function == 'mse' or (loss_function and loss_function == 'mse'):
          return self._mse(y, y_pred)
        elif self.loss_function == 'mae' or (loss_function and loss_function == 'mae'):
          return self._mae(y, y_pred)
        elif self.loss_function == 'rmse' or (loss_function and loss_function == 'rmse'):
          return self._rmse(y, y_pred)
        else:
          raise ValueError(f"Unsupported Loss Function: {self.loss_function}")
      except Exception as e:
          print(f"Error in computing loss: {str(e)}")

    def _mse(self, y, y_pred):
      return np.mean((y-y_pred)**2)

    def _mae(self, y, y_pred):
      return np.mean(np.abs(y-y_pred))

    def _rmse(self, y, y_pred):
      return np.sqrt(np.mean(y-y_pred)**2)
  
    def _r2(self, y, y_pred):
      try: 
        y= np.asarray(y)
        y_pred = np.asarray(y_pred)
        if y.size == 0 or y_pred.size ==0:
          raise ValueError("y or y_pred input arrays cannot be empty.") 
        if y.shape != y_pred.shape:
          raise ValueError(f"y.shape : {y.shape} and y_pred.shape: {y_pred.shape} have incompatible dimensions.")
        
        y_mean = np.mean(y) 
        total_sum_sq = np.sum((y - y_mean)**2) 
        if total_sum_sq == 0:
          return 1.0 if np.allclose(y, y_pred) else 0.0
        
        residual_sum_sq = np.sum((y-y_pred)**2) 
        r2 = 1 - (residual_sum_sq/total_sum_sq) 
        return r2 
      
      except Exception as e:
        print(f"Error in computing r2: {str(e)}")

 

    def compute_gradients(self, X, y, y_pred):
      dW = None
      dB = None
      try:
        if not self.loss_function:
          raise ValueError("Loss function not specified")
        if y.shape != y_pred.shape:
          raise ValueError(f"y.shape : {y.shape} and y_pred.shape: {y_pred.shape} have incompatible dimensions.")
        if self.loss_function == 'mse':
     
          dW = (2/self.batch_size) * np.dot(X.T, (y_pred - y))
          dB = (2/self.batch_size) * np.sum(y_pred - y, axis=0, keepdims=True)
          
        elif self.loss_function == 'mae':
          dW = (1/self.batch_size) * np.dot(X.T, np.sign(y_pred - y))
          dB = (1/self.batch_size) * np.sum(np.sign(y_pred-y), axis = 0, keepdims = True) 

        #elif self.loss_function == 'rmse':
        else:
          raise ValueError(f"Unsupported Loss Function: {self.loss_function}")
      except Exception as e:
        print(f"Error in computing gradients: {str(e)}")
        raise 
      return dW, dB


    def update_parameters(self, dW, dB):
      try:
        if not isinstance(dW, np.ndarray) or not  isinstance(dB, np.ndarray):
          raise TypeError("dW and dB must be numpy arrrays")
        if dW.size == 0 or dB.size == 0:
          raise ValueError("dW and dB cannot be None")
       
        self.W -= self.learning_rate * dW.T
        self.B -= self.learning_rate * dB.T

      except Exception as e:
        print(f"Error in updating parameters: {str(e)}")


    def batch_generator(self):
      try:

        if self.batch_size is None or self.batch_size >= self.n_samples:
          raise ValueError(f"{self.batch_size} is invalid.")

        indices = np.arange(self.n_samples)
        np.random.shuffle(indices)
        for start in range(0, self.n_samples, self.batch_size):
          end = min(start+self.batch_size, self.n_samples)
          batch_indices = indices[start:end]
          yield self.X[batch_indices], self.y[batch_indices]

      except Exception as e:
        print(f"Error in batch_generator: {str(e)}")


    def fit(self, 
            batch_size = None, 
            init_method = None, 
            loss_function = None, 
            shuffle = False, 
            early_stopping = True
            ):
      if batch_size:
        self.batch_size = batch_size
      if init_method:
        self.init_method = init_method
      if loss_function:
        self.loss_function = loss_function
      if not self.init_method:
        self.init_method = 'zero'
      self.initialize_parameters()

      if not self.batch_size:
        self.batch_size = self.X.shape[0]

      epoch_loss = 0
      num_batches = 0
      loss = []

      for epoch in range(self.n_iters):
        for batch_X, batch_y in self.batch_generator():
          # print('X.shape:', batch_X.shape)
          # print('y.shape:', batch_y.shape)


          y_pred = self.predict(batch_X)
          batch_loss = self.compute_loss(batch_y, y_pred)
          #print(f"Batch Loss: {batch_loss}")

          #Update Weights
          dW, dB = self.compute_gradients(batch_X, batch_y, y_pred)
          if not isinstance(dW, np.ndarray) or not  isinstance(dB, np.ndarray):
            raise TypeError("dW and dB must be numpy arrrays")
          if dW.size == 0 or dB.size == 0:
            raise ValueError("dW and dB cannot be None")
          else: 
            self.update_parameters(dW, dB)

          epoch_loss += batch_loss
          num_batches +=1

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        loss.append(avg_epoch_loss)
        if self.n_iters > 0 and epoch % (self.n_iters//10) == 0:
          print(f"Epoch {epoch+1}/{self.n_iters}, Loss: {avg_epoch_loss}")

        

      return self.W, self.B, loss


    def evaluate(self, X_test, y_test, loss_function = None, pred = False):
      y_pred = self.predict(X_test) 
      loss = self.compute_loss(y_test, y_pred, loss_function = loss_function) 
      if pred: 
        return loss, y_pred 
      return loss 
    



      

# Usage example:
# lr = LinearRegression()
# lr.load_dataset(X, y)
# weights, bias, final_cost = lr.fit()
# mse, r2 = lr.evaluate(X_test, y_test)
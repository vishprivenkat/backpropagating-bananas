import numpy as np
import math 
import matplotlib.pyplot as plt 
import argparse 
import pandas as pd 

def load_data(file_path, ip_col, op_col, sgd_flag=False, mb_flag=False, batch_size=1, randomize=False):
  """Load the entire dataset from filename""" 
  try: 
    if not sgd_flag and not mb_flag: 
      #Perform Batch Gradient Descent and load entire dataset 
      dataset = pd.read_csv(file_path) 
      yield dataset[[ip_col, op_col]]
    else:
      chunks = pd.read_csv(file_path, chunksize = batch_size) 
      for chunk in chunks:
        if randomize:
          chunk = chunk.sample(frac=1).reset_index(drop=True) 
        yield chunk[[ip_col, op_col]] 
    return dataset[[ip_col, op_col]] 
    
  except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found") 
  except pd.errors.EmptyDataError:
    print(f"Error: The file '{file_path}' is empty") 
  except KeyError: 
    print(f"Error: One or more columns '{ip_col}', '{op_col}' were not found in the CSV file") 
  except pd.errors.ParserError:
    print(f"Error: The file '{file_path}' could not be parsed") 
  except Exception as e: 
    print(f"An unexpected error occured: {e}") 


def gradient_descent(data_gen, mb_flag=False, sgd_flag=False, lr=0.01, epochs=100): 
  """Perform Gradient Descent""" 
  w = 0 
  b = 0 
  loss = []  #Need Loss to be handled like a dataframe 
  for epoch in range(epochs): 
    dw = 0 
    db = 0 
    
  

def main(args):
    print("==== Received arguments ====")
    print(f"File path: {args.fp}")
    print(f"Input column: {args.ip}")
    print(f"Output column: {args.op}")
    print(f"Stochastic Gradient Descent Flag: {args.sgd}")
    print(f"Mini-batch Gradient Descent Flag: {args.mb}")
    print(f"Batch size: {args.batch_size}")
    print("=============================\n") 

    print("Loading Data")
    data_generator = load_data(args.fp, args.ip, args.op, 
                               sgd_flag=args.sgd, 
                               mb_flag=args.mb, 
                               batch_size=args.batch_size if args.mb else 1, 
                               randomize=True if (args.sgd or args.mb) else False)




if __name__ == "__main__": 
  parser = argparse.ArgumentParser(description = "Gradient Descent Script") 

  #Adding Arguments 
  parser.add_argument('-fp', type=str, required=True, help="Path to the CSV file for gradient descent") 
  parser.add_argument('-ip', type=str, required=True, help="name of the input column") 
  parser.add_argument('-op', type=str, required=True, help="name of the output column") 
  parser.add_argument('-sgd', type=bool, default=False, help="Flag for executing Stochastic Gradient Descent") 
  parser.add_argument('-mb', type=bool, default=False, help="Flag for executing Mini-Batch Gradient Descent") 
  parser.add_argument('-batch_size', type=int, default=8, help="Batch Size for Mini-Batch Gradient Descent") 
  

  args = parser.parse_args() 
  main(args) 
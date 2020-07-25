# Fuzzy_Systems_Classification
TSK models that are using the hybrid method for training.
For the task 1 use [Haberman's dataset](https://archive.ics.uci.edu/ml/datasets/Haberman's+Survival).
For the task 2 use [Epileptic Seizure Recognition Data Set](https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition)
## Task 1  Model 1
- Use Subtractive Clustering 
- Class Independent for clusterInfluenceRange = 0.7
- Change the output function to constant
- Train the TSK model with hybrid method (Backpropagation and Least Squares Method)
- Evaluate the model
  - Error matrix
  - Producer’s accuracy – User’s accuracy
  - Overall accuracy
  - K
## Task 1  Model 2
- Use Subtractive Clustering 
- Class Independent for clusterInfluenceRange = 0.9
- Change the output function to constant
- Train the TSK model with hybrid method (Backpropagation and Least Squares Method)
- Evaluate the model
  - Error matrix
  - Producer’s accuracy – User’s accuracy
  - Overall accuracy
  - K
## Task 1  Model 3
- Use Subtractive Clustering 
- Class Dependent for clusterInfluenceRange = 0.7
- Change the output function to constant
- Train the TSK model with hybrid method (Backpropagation and Least Squares Method)
- Evaluate the model
  - Error matrix
  - Producer’s accuracy – User’s accuracy
  - Overall accuracy
  - K
## Task 1  Model 4
- Use Subtractive Clustering 
- Class Dependent for clusterInfluenceRange = 0.9
- Change the output function to constant
- Train the TSK model with hybrid method (Backpropagation and Least Squares Method)
- Evaluate the model
  - Error matrix
  - Producer’s accuracy – User’s accuracy
  - Overall accuracy
  - K
## Task 2
  - Use Subtractive Clustering 
  - Create a grid search for the best number of features and values of radii
  - Use relieff for feature selection
  - Compare metrics between models to find the best parameters

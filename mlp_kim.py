import os
import sys
import scipy
import numpy as np
import sklearn
import sklearn.neural_network as nn
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nntorch

def mlp_kim():

    # define files to import from Matlab
    Xname=r'C:\Users\sabatini\Documents\currtens\tensor.mat'
    yname=r'C:\Users\sabatini\Documents\currtens\allLabels.mat'
    timename=r'C:\Users\sabatini\Documents\currtens\timepoints_for_tensor.mat'
    holdout_frac_smallClass=0.2 # fraction of data to hold out for testing but only least represented class
    holdout_frac=0.5 # fraction of data to hold out for testing
    L2_alpha=0.001
    #NNrank=0.0005 #0.0125 #0.0005 #0.25 #0.75 # rank of the neural network
    Nneurons=2 #32 # number of neurons in the hidden layer
    takeMoreData=2 # take this fraction of data from the classes with more trials
    nn_solver='adam' # 'lbfgs' or 'adam'
    maxIte=60000 # maximum number of iterations
    lr=0.001 # learning rate for sklearn
    learning_rate_for_torch=0.0001 # learning rate for torch
        
    # read data from files
    X=scipy.io.loadmat(Xname)
    y=scipy.io.loadmat(yname)
    timepoints=scipy.io.loadmat(timename)
    X = X['tensor']
    y = y['allLabels']
    timepoints = timepoints['timepoints_for_tensor']
    timepoints=np.squeeze(timepoints)
    time_step = np.median(np.diff(timepoints, 1, 0))
    print(timepoints)
    print(time_step)
    print(X.shape)
    print(y.shape)

    # Fill all nans with zeros
    X = np.nan_to_num(X)
    y = np.nan_to_num(y)

    # print(X)

    # Cross-validation
    # Split the data into training and test sets (30% held out for testing)
    # Find the number of trials of each class
    n0 = np.sum(y==0)
    n1 = np.sum(y==1)
    n2 = np.sum(y==2)
    n3 = np.sum(y==3)
    # Find the minimum number of trials of any class
    nmin = np.min([n0,n1,n2,n3])
    # Find the minimum in [n0,n1,n2,n3]
    nminloc = np.argmin([n0,n1,n2,n3])
    print('nminloc: ', nminloc)
    print('Class with fewest trials: ', nminloc)
    # Get holdout_frac_smallClass % of the minimum number of trials
    ntest = int(holdout_frac_smallClass*nmin)
    print('Number of test trials: ', ntest)
    print('Number of training trials: ', nmin-ntest)
    # Find which trials of y are class 0
    # Get random nmin-ntest trials of class 0
    if nminloc==0:
        idx0 = np.random.choice(np.where(y==0)[0], nmin-ntest, replace=True)
    else:
        idx0 = np.random.choice(np.where(y==0)[0], int(takeMoreData*(nmin-ntest)), replace=False)
    # Get random nmin-ntest trials of class 1
    if nminloc==1:
        idx1 = np.random.choice(np.where(y==1)[0], nmin-ntest, replace=True)
    else:
        idx1 = np.random.choice(np.where(y==1)[0], int(takeMoreData*(nmin-ntest)), replace=False)
    # Get random nmin-ntest trials of class 2
    if nminloc==2:
        idx2 = np.random.choice(np.where(y==2)[0], nmin-ntest, replace=True)
    else:
        idx2 = np.random.choice(np.where(y==2)[0], int(takeMoreData*(nmin-ntest)), replace=False)
    # Get random nmin-ntest trials of class 3
    if nminloc==3:
        idx3 = np.random.choice(np.where(y==3)[0], nmin-ntest, replace=True)
    else:
        idx3 = np.random.choice(np.where(y==3)[0], int(takeMoreData*(nmin-ntest)), replace=False)
    print('which are 0: ', np.where(y==0)[0])
    print('idx0: ', idx0)
    # Get the indices of the training trials
    train_idx = np.concatenate((idx0,idx1,idx2,idx3)) 
    print('Training trials: ', train_idx)
    # Test trials are the remaining trials
    test_idx = np.setdiff1d(np.arange(0,len(y)), train_idx)
    print('Test trials: ', test_idx)
    
    # Create the training and test sets
    X_train = X[:,:,train_idx]
    X_test = X[:,:,test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Make sure the labels are 1D arrays
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    #n_neurons=NNrank*X_train.shape[0]*X_train.shape[1] # number of neurons in the hidden layer
    n_neurons=Nneurons
    # Floor of n_neurons
    n_neurons = int(n_neurons)
    print('Number of neurons in the hidden layer: ', n_neurons)

    # Reshape the data into 2D arrays
    X_train = np.reshape(X_train, (X_train.shape[0]*X_train.shape[1], X_train.shape[2]))
    X_test = np.reshape(X_test, (X_test.shape[0]*X_test.shape[1], X_test.shape[2]))
    # Trials should be first dimension and features should be second dimension
    X_train = X_train.T
    X_test = X_test.T

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    # Build and train a multilayer perceptron classifier with one hidden layer
    # with n_neurons neurons 
    clf = nn.MLPClassifier(solver=nn_solver, alpha=L2_alpha, hidden_layer_sizes=(n_neurons,), max_iter=maxIte, learning_rate_init=lr, learning_rate='constant')
    clf.fit(X_train, y_train)

    # Print loss over iterations
    if nn_solver=='adam':
        plt.plot(clf.loss_curve_)
        plt.show()

    # Predict the labels of the test data
    y_pred = clf.predict(X_test)

    # Compute the classification accuracy on the test data
    accuracy = clf.score(X_test, y_test)
    print('Accuracy: ', accuracy)

    # Compute accuracy on time- and neuron-shuffled data
    X_test_shuffled = np.zeros(X_test.shape)
    for i in range(X_test.shape[0]):
        X_test_shuffled[i,:] = np.random.permutation(X_test[i,:])
    y_pred_shuffled = clf.predict(X_test_shuffled)
    accuracy_shuffled = clf.score(X_test_shuffled, y_test)
    print('Accuracy on time- and neuron-shuffled data: ', accuracy_shuffled)

    # Compute the classification accuracy on the trial-shuffled data
    y_test_shuffled = np.random.permutation(y_test)
    y_pred_shuffled = clf.predict(X_test)
    accuracy_shuffled = clf.score(X_test, y_test_shuffled)
    print('Accuracy on trial-shuffled data: ', accuracy_shuffled)

    # Compute the confusion matrix
    cm = np.zeros((4,4))
    for i in range(len(y_test)):
        cm[y_test[i],y_pred[i]] += 1
    print('Confusion matrix: ', cm)
    # Normalize first column of confusion matrix by its sum
    # Iterate over all rows of confusion matrix
    for i in range(cm.shape[0]):
        cm[i,:] = cm[i,:]/np.sum(cm[i,:])
    print('Normalized confusion matrix: ', cm)

    # Display the confusion matrix as a heatmap
    sns.heatmap(cm, cmap='jet')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.show()

    
    # Now do the same thing with pytorch
    # Cross-validation
    # Divide into training and test sets with holdout_frac of the data in the test set
    # and the rest in the training set
    # Reshape the data into 2D arrays
    X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
    # Make sure the labels are 1D arrays
    y = np.squeeze(y)
    # Trials should be first dimension and features should be second dimension
    X = X.T
    y = y.T
    print(X.shape)
    print(y.shape)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=holdout_frac) #, stratify=y)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    backup_y_train = y_train

    # Make a multilayer perceptron classifier with one hidden layer using pytorch
    # Define the neural network
    model=nntorch.Sequential(
        nntorch.Linear(X_train.shape[1], n_neurons),
        nntorch.ReLU(),
        nntorch.Linear(n_neurons, 4),
        nntorch.Softmax(dim=1)
    )
    # Compute weights for each class
    print(y_train)
    class_w = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    print(class_w)

    # Define the loss function
    loss_fn = nntorch.CrossEntropyLoss(weight=torch.from_numpy(class_w).float())
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_for_torch, weight_decay=L2_alpha)
    # Convert the data to torch tensors
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    # Train the model
    loss_torch = []
    for t in range(maxIte):
        # Forward pass
        y_pred = model(X_train)
        # Compute and print loss
        loss = loss_fn(y_pred, y_train)
        if t % 1000 == 0:
            print(t, loss.item())
        # Zero the gradients before running the backward pass.
        optimizer.zero_grad()
        # Backward pass
        loss.backward()
        # Update the weights
        optimizer.step()
        # Save the loss
        loss_torch.append(loss.item())
    # Plot the loss over iterations
    plt.plot(loss_torch)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()
    # Predict the labels of the test data
    y_pred = model(torch.from_numpy(X_test).float())
    print(np.argmax(y_pred.detach().numpy(), axis=1))
    print(y_test)
    # Compute the classification accuracy on the test data
    accuracy = np.mean(np.argmax(y_pred.detach().numpy(), axis=1) == y_test)
    print('Accuracy: ', accuracy)
    # Compute accuracy on trial-shuffled data
    y_pred_shuffled = model(torch.from_numpy(X_test[np.random.permutation(X_test.shape[0]),:]).float())
    accuracy_shuffled = np.mean(np.argmax(y_pred_shuffled.detach().numpy(), axis=1) == y_test)
    print('Accuracy on trial-shuffled data: ', accuracy_shuffled)
    # Compute accuracy on time- and neuron-shuffled data
    X_test_shuffled = np.zeros(X_test.shape)
    for i in range(X_test.shape[0]):
        X_test_shuffled[i,:] = np.random.permutation(X_test[i,:])
    y_pred_shuffled = model(torch.from_numpy(X_test_shuffled).float())
    accuracy_shuffled = np.mean(np.argmax(y_pred_shuffled.detach().numpy(), axis=1) == y_test)
    print('Accuracy on time- and neuron-shuffled data: ', accuracy_shuffled)
    # Compute the confusion matrix
    cm = np.zeros((4,4))
    for i in range(len(y_test)):
        cm[y_test[i],np.argmax(y_pred.detach().numpy(), axis=1)[i]] += 1
    print('Confusion matrix: ', cm)
    # Normalize first column of confusion matrix by its sum
    # Iterate over all rows of confusion matrix
    for i in range(cm.shape[0]):
        cm[i,:] = cm[i,:]/np.sum(cm[i,:])
    print('Normalized confusion matrix: ', cm)
    # Display the confusion matrix as a heatmap
    sns.heatmap(cm, cmap='jet')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.show()


    # Now train network on the trial label-shuffled data
    # Shuffle the labels of the training data
    y_train = backup_y_train
    y_train_shuffled = np.random.permutation(y_train)
    # Define the neural network
    model=nntorch.Sequential(
        nntorch.Linear(X_train.shape[1], n_neurons),
        nntorch.ReLU(),
        nntorch.Linear(n_neurons, 4),
        nntorch.Softmax(dim=1)
    )
    # Compute weights for each class
    print(y_train_shuffled)
    class_w = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train_shuffled), y=y_train_shuffled)
    print(class_w)
    # Define the loss function
    loss_fn = nntorch.CrossEntropyLoss(weight=torch.from_numpy(class_w).float())
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_for_torch, weight_decay=L2_alpha)
    # Convert the data to torch tensors
    #X_train = torch.from_numpy(X_train).float()
    y_train_shuffled = torch.from_numpy(y_train_shuffled).long()
    # Train the model
    loss_torch = []
    for t in range(maxIte):
        # Forward pass
        y_pred = model(X_train)
        # Compute and print loss
        loss = loss_fn(y_pred, y_train_shuffled)
        if t % 1000 == 0:
            print(t, loss.item())
        # Zero the gradients before running the backward pass.
        optimizer.zero_grad()
        # Backward pass
        loss.backward()
        # Update the weights
        optimizer.step()
        # Save the loss
        loss_torch.append(loss.item())
    # Plot the loss over iterations
    plt.plot(loss_torch)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()
    # Predict the labels of the test data
    y_pred = model(torch.from_numpy(X_test).float())
    # Compute the classification accuracy on the test data
    accuracy = np.mean(np.argmax(y_pred.detach().numpy(), axis=1) == y_test)
    print('Accuracy for network trained on trial shuffle: ', accuracy)


# # Make a neural network perceptron with one hidden layer using jax
# # Define the neural network
# def init_random_params(scale, layer_sizes, rng=jax.random.PRNGKey(0)):
#     """Build a list of (weights, biases) tuples,
#     one for each layer in layer_sizes, initialized randomly."""
#     keys = jax.random.split(rng, len(layer_sizes))
#     return [jax.random.normal(key, (m, n), scale) for key, (m, n) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:]))]
    
# def relu(x):
#     return np.maximum(0, x)
    
# def predict(params, inputs):
#     activations = inputs
#     for w, b in params[:-1]:
#         outputs = np.dot(activations, w) + b
#         activations = relu(outputs)
#     final_w, final_b = params[-1]
#     return np.dot(activations, final_w) + final_b

# # Define the loss function
# def loss(params, batch):
#     inputs, targets = batch
#     preds = predict(params, inputs)
#     return np.mean((preds - targets)**2)

# # Define the accuracy function
# def accuracy(params, batch):
#     inputs, targets = batch
#     target_class = np.argmax(targets, axis=1)
#     predicted_class = np.argmax(predict(params, inputs), axis=1)
#     return np.mean(predicted_class == target_class)

# # Define the update function
# def update(params, batch, step_size):
#     grads = jax.grad(loss)(params, batch)
#     return [(w - step_size * dw, b - step_size * db)
#             for (w, b), (dw, db) in zip(params, grads)]

# # Define the training loop
# def train(params, train_data, test_data, num_epochs, batch_size, step_size):
#     num_train = train_data[0].shape[0]
#     num_complete_batches, leftover = divmod(num_train, batch_size)
#     num_batches = num_complete_batches + bool(leftover)
#     print('num_batches: ', num_batches)
#     for epoch in range(num_epochs):
#         perm = np.random.permutation(num_train)
#         for i in range(num_batches):
#             batch_idx = perm[i*batch_size:(i+1)*batch_size]
#             params = update(params, (train_data[0][batch_idx], train_data[1][batch_idx]), step_size)
#         train_acc = accuracy(params, train_data)
#         test_acc = accuracy(params, test_data)
#         print("Epoch {} in progress".format(epoch))
#         print("Training set accuracy {}".format(train_acc))
#         print("Test set accuracy {}".format(test_acc))
#     return params





mlp_kim()
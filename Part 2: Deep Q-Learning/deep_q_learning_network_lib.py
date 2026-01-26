# %%
import numpy as np
from pathlib import Path

# %% [markdown]
# For deep q learning, there are a couple of differences in back prop we have to set up, but overall it's very similar to a typical NN setup

# %% [markdown]
# Helper Functions Set Up:
# 
#     - ReLU
# 
#     - softmax
# 
#     - derivative of ReLU
# 
#     - identity
#     
#     - derivative of identity

# %%
def ReLU(Z):
    """Applies ReLU"""
    return np.maximum(0, Z)

def softmax(Z):
    """Applies softmax"""
    return np.exp(Z)/np.sum(np.exp(Z))

def deriv_ReLU(Z):
    """Diffirentiates ReLU"""
    return Z > 0

def identity(Z):
    """Does nothing"""
    return Z

def deriv_identity(Z):
    """Returns 1 like Z"""
    return np.ones_like(Z)

# %% [markdown]
# NN functionality:
# 
#     - init params
# 
#     - forward prop
# 
#     - back prop
# 
#     - updating params
# 
#     - training step
    
# %%
def init_params(nn_structure:list):
    """This function sets up initial parameters W1, b1, W2, b2, ... following the given structure"""

    #weights and biases will contain np matrices of the weights and biases for each layer such that
    #weight[n] = n'th layer weights, etc
    weights = []
    biases = []
    
    for layer_size_index in range(1, len(nn_structure)):
        
        #current layer starting at 1st
        current_layer_size = nn_structure[layer_size_index]
        #previous layer start at 0th (input)
        previous_layer_size = nn_structure[layer_size_index - 1]

        W = np.random.rand(current_layer_size, previous_layer_size) - 0.5
        b = np.random.rand(current_layer_size, 1) - 0.5

        weights.append(W)
        biases.append(b)

    return weights, biases


def forward_propogate(weights, biases ,input_layer, functions):
    """forward propogates the NN using inserted params & functions, given the input layer"""

    s = input_layer.copy() #just for easier notation
    nn_length = len(weights)

    #will be eventually returned
    forward_propogation_params_A = [s]
    forward_propogation_params_Z = []
    
    #forward propogates
    for i in range(nn_length):
        #Note that A_0 = s
        #Z_n = W_n @ A_(n-1) + b_n
        Z = weights[i] @ forward_propogation_params_A[-1] + biases[i]
        
        #A_n = activation function(Z_n)
        A = functions[i](Z)

        forward_propogation_params_Z.append(Z)
        forward_propogation_params_A.append(A)

    
    
    return forward_propogation_params_A, forward_propogation_params_Z


def back_propogate(forward_propogation_params_A, forward_propogation_params_Z, weights, actions, targets, functions_deriv):
    """
    weights: list of weight matrices
    forward_propogation_params_A: list of all A matrices from forward_propagation
    forward_propogation_params_Z: list of all Z matrices from forward_propagation
    actions: array of action indices taken (batch_size,)
    targets: array of target y-values (batch_size,)
    functions_deriv: list of derivative functions for each layer
    """
    #note this is written with a decent amount of help from AI
    #I udnerstand most of this tho, but some of the matrix calculus is confusing
    #AI is unmatched sometimes, it would take a human a long long long time to figure this out


    m = actions.shape[0] #size of the batch

    #gradient storate will later be returned
    dW_list = []
    db_list = []

    #grab output layer
    final_A = forward_propogation_params_A[-1]

    #set everything else to 0, except for the decision which was made
    dZ = np.zeros_like(final_A)
    batch_indices = np.arange(m)
    predictions = final_A[actions, batch_indices]
    dZ[actions, batch_indices] = predictions - targets

    #backpropogate through the layers until we get to the input layer
    for i in range(len(weights) -1, -1, -1):
        #a bunch of matrix calculus I am semi familiar with, but want to understand better
        A_prev = forward_propogation_params_A[i] 

        dW = 1/m * dZ @ A_prev.T
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)

        dW_list.insert(0, dW)
        db_list.insert(0, db)


        if i > 0:
            dZ = (weights[i].T @ dZ) * functions_deriv[i - 1](forward_propogation_params_Z[i - 1])


    return dW_list, db_list

def update_params(weights, biases, back_propogration_weights, back_propogration_biases, alpha):
    """updates all the params in the neural netwrok"""

    nn_length = len(weights)

    
    #update everything
    for i in range(nn_length):
        #split for notation
        W = weights[i]
        b = biases[i]

        dW = back_propogration_weights[i]
        db = back_propogration_biases[i]

        #print(f'weight W: {W.shape}')
        #print(f'delta weight dW: {dW.shape}')
        #update the biases
        W -= alpha * dW
        b -= alpha * db

    return weights, biases #Since arrays are pointers, I can just return the orignial list of arrays

def train_step(main_model_weights, main_model_biases, target_model_weights, target_model_biases, batch, functions, function_derivs, gamma, alpha):
    """
    completes a training step on a batch
    """
    #this function is also written with the help of AI

    #unpack the batch
    s, a, r, s_next, done = batch

    #get Targets
    forward_propogation_params_A_next, _ = forward_propogate(target_model_weights, target_model_biases, s_next, functions)
    q_next_max = np.max(forward_propogation_params_A_next[-1], axis=0)

    #Calculate targets using bellman equation:
    targets = r + (gamma * q_next_max * ( 1 - done)) #so for everything it's just gonna be r + gamme * q_next_max other then for the very last layer

    #Forward pass through the main network
    #This is required for backprop
    forward_propogation_params_A, forward_propogation_params_Z = forward_propogate(main_model_weights, main_model_biases, s, functions)

    #Backpropogate
    back_propogration_weights, back_propogration_biases = back_propogate(forward_propogation_params_A, forward_propogation_params_Z, main_model_weights, a, targets, function_derivs)

    
    #Update weights
    main_model_weights, main_model_biases = update_params(main_model_weights, main_model_biases, back_propogration_weights, back_propogration_biases, alpha)

    return main_model_weights, main_model_biases



def update_target_model(main_model_weights, main_model_biases):
    """updates the target model to have the main model weights"""
    target_model_weights = [w.copy() for w in main_model_weights]
    target_model_biases = [b.copy() for b in main_model_biases]

    return target_model_weights, target_model_biases


def save_model(weights, biases, model_name, location=''):
    """saves the model to a .npy file"""
    path = Path(location + '/' + model_name) 
    path.mkdir(parents=True, exist_ok=True) # makes the directory to store the path in

    np.save(f"{location + '/' + model_name}/weights.npy", weights) #stores weights
    np.save(f"{location + '/' + model_name}/biases.npy", biases) #stores biases

def load_model(location, model_name):
    """loads the parameters"""
    weights = np.load(f"{location + '/' + model_name}/weights.npy")
    biases = np.load("{location + '/' + model_name}/biases.npy")


def pick_action_softmax(output_values, tau):
    """picks a random action using the boltzman function"""

    probability_weights = np.power(np.e, - tau * output_values)  #generates probability weights using softmax(values)
    pick = np.random.choice(range(len(probability_weights)), p=probability_weights/probability_weights.sum()) #picks random move using weights
    return pick
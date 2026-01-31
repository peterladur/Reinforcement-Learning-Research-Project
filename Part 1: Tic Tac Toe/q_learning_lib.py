# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
from typing import Tuple
from pprint import pprint
import json

# %% [markdown]
# **Part 0: Preparation**<br>
# Building a validity checker<br>
# Building a determining function<br>

# %% [markdown]
# <h3>012</h3>
# <h3>345</h3>
# <h3>678</h3>

# %%
def check_validity(state: str) -> bool: 
    """checks if a state is valid"""

    x_number = state.count('x')
    o_number = state.count('o')
    empty_number = state.count('_')
    #There should be at most 1 more x then o
    if (x_number < o_number) or (x_number - 1 > o_number):
        return False
    
    if (x_number + o_number + empty_number) != 9:
        return False

    #There can at most be one win
    x_win = False
    o_win = False

    #check diagonals
    if ((state[0] + state[4] + state[8]) == "ooo") or ((state[6] + state[4] + state[2]) == "ooo"):
        o_win = True
    elif ((state[0] + state[4] + state[8]) == "xxx") or ((state[6] + state[4] + state[2]) == "xxx"):
        x_win = True  


    #Check rows and cols
    for i in range(3):
        if (state[3 * i] + state[3 * i + 1] + state[3 * i + 2] == "xxx") or ((state[i] + state[3 + i] + state[6 + i] == "xxx")):
            x_win = True
        if (state[3 * i] + state[3 * i + 1] + state[3 * i + 2] == "ooo") or ((state[i] + state[3 + i] + state[6 + i] == "ooo")):
            o_win = True

    #if two players won, that's impossible
    if x_win and o_win:
        return False
    if x_win and (x_number == o_number):
        return False
    if o_win and (x_number > o_number):
        return False
    
    return True

def check_result(state: str) -> int:
    """checks what is the result of this state return game result (-1, 0, 1) or 2 if the game is still going"""
    #check diagonals
    if ((state[0] + state[4] + state[8]) == "ooo") or ((state[6] + state[4] + state[2]) == "ooo"):
        return -1
    elif ((state[0] + state[4] + state[8]) == "xxx") or ((state[6] + state[4] + state[2]) == "xxx"):
        return 1
    #Check rows and cols
    for i in range(3):
        if (state[3 * i] + state[3 * i + 1] + state[3 * i + 2] == "xxx") or ((state[i] + state[3 + i] + state[6 + i] == "xxx")):
            return 1
        if (state[3 * i] + state[3 * i + 1] + state[3 * i + 2] == "ooo") or ((state[i] + state[3 + i] + state[6 + i] == "ooo")):
            return -1
    
    if state.count('_') > 0:
        return 2

    return 0

def display_state(state: str):
    """displays the state"""
    print(state[0] + state[1] + state[2])
    print(state[3] + state[4] + state[5])
    print(state[6] + state[7] + state[8])

# %%
"""
___  __x  _ox  _ox  _ox  _ox _ox _ox
___  ___  ___  _x_  _xo  _xo _xo _xo
___  ___  ___  ___  ___  _x  _xo xxo 
"""

# %% [markdown]
# Step 1:
# Import Data Set
# 
# Step 2:
# Create a dead empty Q-Table
# 
# Step 3:
# Create helper functions
# 
# Step 4:
# 'Play The Game' Function
# 
# Step 5:
# Training Function
# 
# Step 6:
# Visualising Results
# 
# Step 7:
# Result analysis

# %%
#Step 1 & 2:

def init_Q_Table(filename="data_exports/list_of_all_possible_states.csv"):
    """Creates an empty Q_Table as a dictionary with numpy arrays of zeros"""
    
    Q_Table_read = pd.read_csv("data_exports/list_of_all_possible_states.csv") #read the data set using pandas



    #Create an empty Q_Table
    Q_Table = dict()

    for state in Q_Table_read['state']:
        Q_Table[state] = np.zeros(9)

    return Q_Table


# %%
#Import Perfect Q_Table
def change_to_numpy(actions):
    """Changes a normal list to a numpy array"""
    actions = np.array(actions, dtype=float)
    return np.array(actions)

def import_perfect_Q_Table(filename="data_exports/perfect_Q_Table.json"):
    """"Imports the perfect Q_Table that was generated using a different scirpt"""

    with open(filename, "r", encoding="utf-8") as f:
        perfect_Q_Table = json.load(f)


    perfect_Q_Table = {state: change_to_numpy(action) for state, action in perfect_Q_Table.items()}

    return perfect_Q_Table

def export_Q_Table(Q_Table, filename='export_Q_Table.json'):
    """Exports the Q_Table as a .json file"""
    Q_Table_copy = dict()
    Q_Table_copy = {i: Q_Table[i].tolist() for i in Q_Table}
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(Q_Table, f, indent=2)

# %%
#Helper Functions
def return_valid_moves(state: str):
    """returns an array where 
    1 => possible to make a move
    0 => square is taken"""
    possible_moves = np.zeros(9)
    for i, move in enumerate(state):
        if move == '_':
            possible_moves[i] = 1
    
    return possible_moves

def pick_learning_move(Q_Table: dict, state:str, tau=np.e, player='x') -> int:
    """picks a random move from a list of possible moves
    uses the softmax function as weights
    """
    
    valid_moves = return_valid_moves(state)

    player_multiplier = 1 #x aims maximise the score
    if player == 'o':
        player_multiplier = -1 #o aims to minimise the score

    weights = np.power(tau, player_multiplier * Q_Table[state]) * valid_moves #generates weights using softmax(values)
    pick = np.random.choice(range(9), p=weights/weights.sum()) #picks random move using weights

    return pick

def pick_random_move(state: str) -> int:
    """This functions picks any random move in the game"""

    weights = return_valid_moves(state) #string x___o__x_ would give weights [0, 1, 1, 1, 0, 1, 1, 0, 1]
    pick = np.random.choice(range(9), p=weights/weights.sum()) 
    return pick


def pick_perfect_move(perfect_Q_Table: dict, state: str, player='x', random_perfect_move:bool = True) -> int:
    """This picks a random best possible move"""

    q_row = perfect_Q_Table[state].copy() #this is used to make sure that na illigal move is not picked
    for i, char in enumerate(state):
        if char != '_':
            q_row[i] = np.nan

    desired_move = np.nanmax(q_row) #x aims to pick the move with the highest Q-value from all avaliable moves

    if player == 'o': #o aims to pick the move with the lowest Q-value from all avaliable moves
        desired_move = np.nanmin(q_row)

    possible_indices = [] #picking a random perfect move

    for i, move in enumerate(q_row): #go through each move
        if move == desired_move:
            possible_indices.append(i) #add it to the list

            if random_perfect_move: #if we just want the first perfect move, return it
                return i

    return random.choice(possible_indices) 


def update_board(state:str, action, player='x') -> str:
    """returns a string which is the updated board, after a move(action) has been played"""
    state_list = list(state)
    state_list[action] = player
    state = ''.join(state_list)
    return state

def pick_maximum(Q_Table: dict, state: str):
    """finds the what is the best possible result you can achieve"""
    maximum = -1
    valid_moves = return_valid_moves(state)

    for i in range(9):
        if valid_moves[i] == 1:

            if Q_Table[state][i] > maximum:
                maximum = max(maximum, Q_Table[state][i])

    
    return maximum

def pick_minimum(Q_Table: dict, state: str):
    """finds the what is the worst possible result you can achieve"""
    minimum = 1
    valid_moves = return_valid_moves(state)

    for i in range(9):
        if valid_moves[i] == 1:

            if Q_Table[state][i] < minimum:
                minimum = min(minimum, Q_Table[state][i])
    
    return minimum


def learn_from_queue(Q_Table, queue, alpha=0.1, player='x'):
    """Updates the Q_Table based on the past games
    queue is a list which contains the game and the result
    game is a list which contains the states and the actions take
    """


    for game, result in queue: #goes through ever game in the queue

        for move_number, (state, action) in enumerate(game): #go through each state and action taken in the game

            if (move_number + 1) == len(game): #check if the game is terminated this turn
                Q_Table[state][action] += alpha * (result - Q_Table[state][action]) #If the game is terminated, the give immidiate reward as a resut

            else: #if the game is not terminated this turn
                #apply the formula:
                #   Q(state, action) = Q(state, action) + alpha [r + gamma * max/min Q(future state, future action) - Q(state, action)]
                #   alpha = learning rate
                #   gamma = discount rate
                #   maximum future reward for x, or minimum future reward for o
                next_state = game[move_number + 1][0] #next state 
                if player == 'x':
                    Q_Table[state][action] += alpha * (pick_maximum(Q_Table, next_state) - Q_Table[state][action]) # x aims to have most positive winning chances
                if player == 'o':
                    Q_Table[state][action] += alpha * (pick_minimum(Q_Table, next_state) - Q_Table[state][action]) # o aims to have most negative entries
    return Q_Table

def display_counter(counter):
    """displays the statistics over the last 100 games"""
    total = sum(counter) / 100
    print(f"{counter[0]/total:.2f}%   {counter[1]/total:.2f}%   {counter[2]/total:.2f}%")



# %%

def plot_training_results(data, total_batches, batch_size):
    """
    Plots training results. The X-axis starts exactly at 0 and ends at total_batches * batch_size.
    """
    # 1. Ensure data is a 2D array of floats
    arr = np.array(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Data is not 2D. Did you pass the Q-Table instead of the results?")

    # 2. Normalize to 100%
    row_sums = arr.sum(axis=1)
    scale = np.divide(100.0, row_sums, out=np.zeros(len(row_sums), dtype=float), where=(row_sums!=0))
    arr = (arr.T * scale).T

    # 3. Create X-axis scaling starting at 0
    total_games = total_batches * batch_size
    num_data_points = len(arr)
    
    # Changed: starts at 0 instead of (total_games / num_data_points)
    x_axis = np.linspace(0, total_games, num_data_points)

    # 4. Plot
    plt.figure(figsize=(10, 6))
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    labels = ["O win", "Draw", "X win"]
    
    plt.stackplot(x_axis, arr[:, 0], arr[:, 1], arr[:, 2], 
                  labels=labels, colors=colors, alpha=0.85)

    plt.title(f"Training Progress (Total Games: {total_games:,})", fontsize=14)
    plt.xlabel("Total Games Played", fontsize=12)
    plt.ylabel("Distribution (%)", fontsize=12)
    
    # Ensure the axis tightly hugs the data
    plt.xlim(0, total_games)
    plt.ylim(0, 100)
    plt.legend(loc='upper right', frameon=True, facecolor='white')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()

# %%
#Functions related to playing the game

def play_the_game_learning(Q_Table, tau=np.e, player='x', perfect_opponent=False, perfect_Q_Table=dict()):
    """Playes the game and returns a queue of tuples"""
    queue = [] #Used for training
    state = "_________" #original state
    move_number = 0
    result = 2 #currently the game is being played
    player_index = 0 #x has a player index of 0, o has a player index of 1 
    oppostion = 'o'


    if player == 'o':
        player_index = 1
        oppostion = 'x'

    while result == 2: #While the game is still going
        if (move_number % 2) == player_index: #if it's the turn of the learning player
            action = pick_learning_move(Q_Table, state, tau, player) #play a learning move
            queue.append((state, action)) #record the state and the action take
            state = update_board(state, int(action), player) #update board

        else: #if  it's the turn of the computer player
            if perfect_opponent: #perfect opponent players perfect move
                action = pick_perfect_move(perfect_Q_Table, state, oppostion)
            else: #random opponent plays random move
                action = pick_random_move(state)
            state = update_board(state, action, oppostion) #Update board

        move_number += 1 

        result = check_result(state) #check if the result is now terminal
    return queue, result

def play_the_game_random():
    """Playes the randomly from both sides game and returns the results of the game"""
    state = "_________"
    move_number = 0
    result = 2


    while result == 2: #While the game is still going
        if (move_number % 2) == 0: #x
            action = pick_random_move(state) #plays a random move
            state = update_board(state, action, 'x') 

        else: #o
            action = pick_random_move(state) #plays a random move
            state = update_board(state, action, 'o')

        move_number += 1

        result = check_result(state) #checks if the result is terminal
    return result

def play_the_game_two_Q_Tables(Q_Table_X, Q_Table_O, strategy='perfect', tau=np.e):
    """plays one game using either 'perfect' or 'softmax' strategy for each player"""

    state = "_________"
    move_number = 0
    result = 2

    while result == 2: #While the game is still going

        if (move_number % 2) == 0: #x
            if strategy == 'perfect':
                action = pick_perfect_move(Q_Table_X, state, 'x', True) #plays a perfect move
            else:
                action = pick_learning_move(Q_Table_X, state, tau, 'x') #plays a softmax move
            state = update_board(state, action, 'x') 

        else: #o
            if strategy == 'perfect':
                action = pick_perfect_move(Q_Table_O, state, 'o', True) #plays a perfect move
            else:
                action = pick_learning_move(Q_Table_O, state, tau, 'o') #plays a softmax move
            state = update_board(state, action, 'o')

        move_number += 1

        result = check_result(state) #checks if the result is terminal

    return result

def play_the_game_Q_Table_vs_policy(Q_Table, player, strategy, perfect_Q_Table):
    """plays the game using best moves from Q_Table from one opponent and moves from a random/perfect policy for the other opponent"""

    state = "_________" #original state
    move_number = 0
    result = 2 #currently the game is being played
    player_index = 0 #x has a player index of 0, o has a player index of 1 
    oppostion = 'o'


    if player == 'o':
        player_index = 1
        oppostion = 'x'

    while result == 2: #While the game is still going
        if (move_number % 2) == player_index: #if it's the turn of the player using Q_Table
            action = pick_perfect_move(Q_Table, state, player)
            state = update_board(state, int(action), player) #update board

        else: #if  it's the turn of the computer player
            if strategy == 'perfect': #perfect opponent players perfect move
                action = pick_perfect_move(perfect_Q_Table, state, oppostion)
            else: #random opponent plays random move
                action = pick_random_move(state)
            state = update_board(state, action, oppostion) #Update board

        move_number += 1 

        result = check_result(state) #check if the result is now terminal

    return result


def calculate_tau(turn) -> float:
    """calculates tau
    tau can change to prioritise exploitation over exploration"""
    
    tau = 1 + 2 * turn/100
    
    return float(tau)

def calculate_alpha(turn) -> float:
    """calculates alpha
    Alpha can be lowered to decrease the learning rate eventually"""
    
    alpha = 0.1#max(0.0001, 0.1 * np.exp(-0.0001 * turn))
    return float(alpha)


# %%

BATCH_SIZE = 10
NUMBER_OF_BATCHES = 10000

def perform_training(player, opponent_type='perfect',number_of_batches=NUMBER_OF_BATCHES, batch_size=BATCH_SIZE, display_training=True,
                      alpha_func=calculate_alpha, tau_func=calculate_tau, result_frequency=50):
    """trains a certain player on a certain strategy
    
    returns the trained Q-Table and the wins distribution as a np.matrix"""


    if display_training:
        print('o win   draw    x win')
    Q_Table = init_Q_Table()

    counter_final_values = [] #list of how well the bot is performing
    counter = [0, 0, 0]

    if opponent_type == 'perfect':
        perfect_Q_Table = import_perfect_Q_Table()
    

    for batch_number in range(0, number_of_batches): #playes this many batches
        game_queue = [] #queue for the batch

        tau = tau_func(batch_number * batch_size) #decides on tau



        for game_number in range(batch_size): #plays through each batch
            
            if opponent_type == 'perfect':
                queue, result = play_the_game_learning(Q_Table,tau,  player, True, perfect_Q_Table) #plays the games against optimal opponent
            if opponent_type == 'random':
                queue, result = play_the_game_learning(Q_Table,tau,  player, False) #plays the games against random opponent               

            counter[result + 1] += 1 #updates the result counter

            game_queue.append((queue, result)) #adds the game to the qeue

        alpha = alpha_func(batch_number * batch_size)#calculates the learning rate


        Q_Table = learn_from_queue(Q_Table, game_queue, alpha, player) #learn (updates the Q_Table)

        if batch_number % result_frequency == 0: #every hundred values, displays score
            counter_final_values.append(counter)

            if display_training:
                print(int(batch_number))
                display_counter(counter)
            counter = [0, 0, 0]

    return Q_Table, np.array(counter_final_values)



def Q_Table_match(Q_Table_X, Q_Table_O, number_of_games=1000, strategy='perfect', tau=np.e):
    """Lets two Q-Table play a match against each other and returns results
    
    strategy is either:
    'perfect' meaning the largest Q-Table value is used, 
    'softmax' meaning moves are picked with weighted unequal probability
    """

    counter = np.zeros(3) #this will be returned as the score

    for game in range(number_of_games):

        result = play_the_game_two_Q_Tables(Q_Table_X, Q_Table_O, strategy, tau) #plays the game

        counter[1 + result] += 1 #adds one to the counter

    return counter


def Q_Table_vs_policy_match(Q_Table, player, number_of_games=1000, strategy='perfect'):
    """Let's a Q-Table play against a certain policy (Q_table of x trainied on perfect, vs random O, etc) and returns the results"""

    counter = np.zeros(3) #this will be returned as the score
    perfect_Q_Table = import_perfect_Q_Table()


    for game in range(number_of_games):

        result = play_the_game_Q_Table_vs_policy(Q_Table, player, strategy, perfect_Q_Table)

        counter[1 + result] += 1

    return counter

def play_random_match(number_of_games=1000):
    """"plays a match where both players move randomly and returns the results"""
    counter = np.zeros(3) #this will be returned as the score

    for game in range(number_of_games):

        result = play_the_game_random()

        counter[1 + result] += 1

    return counter











#------------- Monte Carlo -------------------

def learn_from_queue_MC(Q_Table, queue, alpha=0.1):
    """
    Updates the Q_Table using Monte Carlo Policy Evaluation.
    Unlike Q-Learning, this does not bootstrap (look at future Q-values).
    It moves the Q-value of every state-action pair in the game toward the final result.
    """
    for game, result in queue:
        for state, action in game:
            # MC Update Rule: Q(s,a) = Q(s,a) + alpha * (Final_Result - Q(s,a))
            Q_Table[state][action] += alpha * (result - Q_Table[state][action])
            
    return Q_Table

def perform_training_MC(player, opponent_type='random', number_of_batches=NUMBER_OF_BATCHES, 
                        batch_size=BATCH_SIZE, display_training=True,
                        alpha_func=calculate_alpha, tau_func=calculate_tau, result_frequency=50):
    """
    Trains a player using Monte Carlo learning.
    """
    if display_training:
        print(f'Training {player} via Monte Carlo...')
        print('o win   draw    x win')
        
    Q_Table = init_Q_Table()
    counter_final_values = []
    counter = [0, 0, 0]

    if opponent_type == 'perfect':
        perfect_Q_Table = import_perfect_Q_Table()

    for batch_number in range(1, number_of_batches):
        game_queue = []
        tau = tau_func(batch_number * batch_size)

        for game_number in range(batch_size):
            if opponent_type == 'perfect':
                queue, result = play_the_game_learning(Q_Table, tau, player, True, perfect_Q_Table)
            else:
                queue, result = play_the_game_learning(Q_Table, tau, player, False)

            counter[result + 1] += 1
            game_queue.append((queue, result))

        alpha = alpha_func(batch_number * batch_size)
        
        # Call the Monte Carlo learning function
        Q_Table = learn_from_queue_MC(Q_Table, game_queue, alpha)

        if batch_number % result_frequency == 0:
            counter_final_values.append(counter)
            if display_training:
                print(int(batch_number))
                display_counter(counter)
            counter = [0, 0, 0]

    return Q_Table, np.array(counter_final_values)



def init_Visits_Table(filename="data_exports/list_of_all_possible_states.csv"):
    """Creates a table to track how many times each state-action pair has been visited."""
    Q_Table_read = pd.read_csv(filename)
    Visits_Table = dict()
    for state in Q_Table_read['state']:
        Visits_Table[state] = np.zeros(9)
    return Visits_Table


def learn_from_queue_MC_incremental(Q_Table, Visits_Table, queue):
    """
    Updates the Q_Table using the incremental mean formula from the provided code.
    Q(s,a) = Q(s,a) + (1/N(s,a)) * (Reward - Q(s,a))
    """
    for game, result in queue:
        for state, action in game:
            # Increment the visit counter for this specific state-action pair
            Visits_Table[state][action] += 1
            n = Visits_Table[state][action]
            
            # The incremental mean update rule
            Q_Table[state][action] += (result - Q_Table[state][action]) / n
            
    return Q_Table, Visits_Table


def perform_training_MC_incremental(player, opponent_type='random', number_of_batches=NUMBER_OF_BATCHES, 
                                   batch_size=BATCH_SIZE, display_training=True,
                                   tau_func=calculate_tau, result_frequency=50):
    """
    Trains a player using the Monte Carlo Incremental Mean strategy.
    """
    if display_training:
        print(f'Training {player} via Incremental Monte Carlo...')
        print('o win   draw    x win')
        
    Q_Table = init_Q_Table()
    Visits_Table = init_Visits_Table() # Track visits for 1/N update
    
    counter_final_values = []
    counter = [0, 0, 0]

    if opponent_type == 'perfect':
        perfect_Q_Table = import_perfect_Q_Table()

    for batch_number in range(1, number_of_batches):
        game_queue = []
        tau = tau_func(batch_number * batch_size)

        for game_number in range(batch_size):
            if opponent_type == 'perfect':
                queue, result = play_the_game_learning(Q_Table, tau, player, True, perfect_Q_Table)
            else:
                queue, result = play_the_game_learning(Q_Table, tau, player, False)

            counter[result + 1] += 1
            game_queue.append((queue, result))
        
        # Use the incremental update rule from the other code
        Q_Table, Visits_Table = learn_from_queue_MC_incremental(Q_Table, Visits_Table, game_queue)

        if batch_number % result_frequency == 0:
            counter_final_values.append(counter.copy())
            if display_training:
                print(int(batch_number))
                display_counter(counter)
            counter = [0, 0, 0]

    return Q_Table, np.array(counter_final_values)
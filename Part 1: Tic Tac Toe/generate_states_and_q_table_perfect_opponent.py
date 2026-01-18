from itertools import permutations
import time
import pprint
import json

def find_maximum(entries:list):
    """finds the maximum of a list ignoring null values"""
    if entries == None:
        return -1

    maximum = -1
    for entry in entries:
        if entry != None:
            maximum = max(maximum, entry)

    return maximum

def find_minimum(entries:list):
    """finds the minimum of a list ignoring null values"""
    if entries == None:
        return 1
    
    minimum = 1
    for entry in entries:
        if entry != None:
            minimum = min(minimum, entry)

    return minimum


def check_state(state:str):
    """Checks the given state

    Output:
    None: Invalid State
    -1: Win for O
    0: Drawn
    1: Win for X
    2: Undetermined
    """
    
    x_number = state.count('x')
    o_number = state.count('o')
    
    #check if the number of x, o is correct
    if (x_number < o_number) or (x_number - o_number) > 1:
        return None
    
    
    #check if at most 1 player has won
    x_win = False
    o_win = False
    
    diag1 = state[0] + state[4] + state[8] #/
    diag2 = state[2] + state[4] + state[6] #\
    
    if (diag1 == 'xxx') or (diag2 == 'xxx'):
        x_win = True   
    if (diag1 == 'ooo') or (diag2 == 'ooo'):
        o_win = True

    
    for i in range(3):
        row = state[0 + 3*i] + state[1 + 3*i] + state[2 + 3*i] #checks each row
        col = state[0 + i] + state[3 + i] + state[6 + i] #checks each column
        if (row == 'xxx') or (col == 'xxx'):
            x_win = True
        if (row == 'ooo') or (col == 'ooo'):
            o_win = True
    
    if (o_win and (x_number > o_number)): #O can only win provided x_number == o_number
        return None

    if(x_win and (x_number == o_number)): #X can only win provided x_number == o_number + 1
        return None

    if (x_win and o_win): #both players cannot win
        return None

    if (x_win):
        return 1

    if (o_win):
        return -1 
    
    if (x_number + o_number) == 9:
        return 0
    
    return 2


def display_state(state: str, message=""):
    """displays state as a 3x3"""
    print(message) #Useful for debugging
    print(state[:3])
    print(state[3:6])
    print(state[6:])


def generate_all_possible_states():
    """generates all possible states"""


    Q_Table = dict()

    """part 1 generates all the final states"""
    digits = ['x'] * 5 + ['o'] * 4

    # Generate all unique permutations
    unique_combinations = set(permutations(digits))
    unique_combinations =  {''.join(x) for x in unique_combinations}

    for combination in unique_combinations: #checks each possible permutation for validity
        if check_state(combination) != None:
            Q_Table[combination] = [check_state(combination)] * 9 #if the state is valid, it adds it to the Q-Table (check_state == result of a completely finished game)

    possible_states = dict() #possible states, stores moves 0-9 mapped to a list of all possible states for that number of moves
    possible_states[9] = unique_combinations
    

    """part 2 will remove an x or a o, step by step to generate a Q-Table

    "parent combination" is the original combination from which a symbol is removed """
    for turn in range(8, -1, -1):
        possible_states[turn] = set() 

        #the symbol removed depends on which turn it is
        symbol = "x"
        if (turn % 2 == 1):
            symbol = "o"


        for combination in possible_states[turn + 1]: #goes through each "parente combination"
            list_combination = list(combination) 
            for i, char in enumerate(list_combination): #for each x/o in the combination, it removes it, and adds the possibility to the list
                if char == symbol:
                    list_combination[i] = "_"


                    str_combination = ''.join(list_combination)
                    possible_states[turn].add(str_combination)

                    #if the state is physically possible, generates an entry in the Q Table for it
                    result = check_state(str_combination)
                    if result != None: #if state is valid 
                        Q_Table.setdefault(str_combination, [None] * 9)
                        if result != 2: #if the game state is a determined state
                            Q_Table[str_combination] = [result] * 9 #give it the determined value

                        else:
                            validity = check_state(combination) #we want to see what is the result of the original combination
                            
                            if validity != 2: #If the original state is terminal, then the q value for that entry is the result
                                Q_Table[str_combination][i] = validity 
                            else: #If the original combination is not valid then we 
                                if symbol == 'x': #if X was removed, we want to see, how good was the original position for O 
                                    Q_Table[str_combination][i] = find_minimum(Q_Table.setdefault(combination, [])) #We pick the lowest possible Q-value
                                if symbol == 'o': #if O was removed, it is the opposite
                                    Q_Table[str_combination][i] = find_maximum(Q_Table.setdefault(combination, [])) #We pick the lowest possible Q-value


                    list_combination[i] = symbol #since, we have removed the symbol earlier, we now add it back
    
    for turn in possible_states: #makes sure each state is a valid one
        possible_states[turn] = {x for x in possible_states[turn] if (check_state(x) != None)}

    return possible_states, Q_Table #return all the values

def print_and_write_statistics(possible_states, Q_Table, csv_filename="list_of_all_possible_states.csv", json_filename="perfect_Q_Table.json"):
    """prints the statistics for the Q-Table
    
    The game state distribution (o-wins, draws, x-wins, undetermined)
    The total possible number of game states

    Writes the data

    All possible game states (e.g. "xx_xx_xooo_") to a .csv file
    The Q-Table for the perfect opponent to a .json file
    """    
    
    x_wins = 0
    o_wins = 0
    draws = 0
    undetermined = 0



    all_states = []

    for turn in possible_states: #goes through each state
        for position in possible_states[turn]: #records it's result
            result = check_state(position)
            if result == 1:
                x_wins +=1
            if result == -1:
                o_wins += 1
            if result == 0:
                draws += 1
            if result == 2:
                undetermined += 1
            all_states.append(position) #adds it to a final list


    #print statistics
    print(f"Total possible positions: {x_wins + o_wins + draws + undetermined}")
    print(f"x wins: {x_wins}")
    print(f"o wins: {o_wins}")
    print(f"Draws: {draws}")
    print(f"Undetermined: {undetermined}")


    #write .csv and .json file
    write_set_to_file(all_states, csv_filename)
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(Q_Table, f, indent=2)


def write_set_to_file(my_set, filename):
    """
    Writes each element of a set to a text file, with each entry on a new line.
    """
    try:
        with open(filename, 'w') as f:
            f.write('state' + '\n')
            for item in my_set:
                f.write(str(item) + '\n')
        print(f"Set elements successfully written to '{filename}'.")
    except IOError as e:
        print(f"Error writing to file '{filename}': {e}")


def main():

    TOTAL_LENGTH = 0
    t1 = time.time()
    possible_states, Q_Table = generate_all_possible_states()
    print_and_write_statistics(possible_states, Q_Table)
    t2 = time.time()
    print(t2 - t1)

main()
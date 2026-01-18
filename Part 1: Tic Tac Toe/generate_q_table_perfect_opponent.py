from itertools import permutations
import time
import pprint
import json

def find_maximum(entries):

    if entries == None:
        return -1

    maximum = -1
    for entry in entries:
        if entry != None:
            maximum = max(maximum, entry)

    return maximum

def find_minimum(entries):

    if entries == None:
        return 1
    
    minimum = 1
    for entry in entries:
        if entry != None:
            minimum = min(minimum, entry)

    return minimum


def check_if_valid(state:str):
    """checks if a given state is possible"""

    """check if the number x, o is correct"""
    x_number = state.count('x')
    o_number = state.count('o')
    
    if (x_number < o_number) or (x_number - o_number) > 1:
        return None
    
    
    """check if the there if there is more then one win on the board"""
    x_win = False
    o_win = False
    
    diag1 = state[0] + state[4] + state[8] #/
    diag2 = state[2] + state[4] + state[6] #\
    
    if (diag1 == 'xxx') or (diag2 == 'xxx'):
        x_win = True   
    if (diag1 == 'ooo') or (diag2 == 'ooo'):
        o_win = True

    
    for i in range(3):
        row = state[0 + 3*i] + state[1 + 3*i] + state[2 + 3*i]
        col = state[0 + i] + state[3 + i] + state[6 + i]
        if (row == 'xxx') or (col == 'xxx'):
            x_win = True
        if (row == 'ooo') or (col == 'ooo'):
            o_win = True
    
    if (o_win and (x_number > o_number)):
        return None

    if(x_win and (x_number == o_number)):
        return None

    if (x_win and o_win):
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
    print(message)
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

    for combination in unique_combinations:
        if check_if_valid(combination) != None:
            Q_Table[combination] = [check_if_valid(combination)] * 9

    possible_states = dict()
    possible_states[9] = unique_combinations
    
    #for combination in possible_states[9]:
        #Q_Table.setdefault(combination, [check_if_valid(combination)] * 9)


    """part 2 will remove an x or a o, step by step"""
    for turn in range(8, -1, -1):
        #print(turn)
        possible_states[turn] = set()

        #depends on which turn it is
        symbol = "x"
        if (turn % 2 == 1):
            symbol = "o"

        for combination in possible_states[turn + 1]: #goes through each parent combination
            list_combination = list(combination) 
            for i, char in enumerate(list_combination): #for each x/o in the combination, it removes it, and adds the possibility to the list
                if char == symbol:
                    list_combination[i] = "_"


                    str_combination = ''.join(list_combination)
                    possible_states[turn].add(str_combination)

                    #if the state is physically possible, generates an entry in the Q Table for it
                    result = check_if_valid(str_combination)
                    if result != None: #if state is valid 
                        Q_Table.setdefault(str_combination, [None] * 9)
                        if result != 2: #if the game state is a determined state
                            Q_Table[str_combination] = [result] * 9 #give it the determined value

                        else:
                            validity = check_if_valid(combination) #we want to see what is the result of the original combination
                            
                            if validity in {-1, 1, 0}:
                                Q_Table[str_combination][i] = validity
                            else:
                                if symbol == 'x':
                                    Q_Table[str_combination][i] = find_minimum(Q_Table.setdefault(combination, []))
                                if symbol == 'o':
                                    Q_Table[str_combination][i] = find_maximum(Q_Table.setdefault(combination, []))


                    list_combination[i] = symbol

                    
    x_wins = 0
    o_wins = 0
    draws = 0
    undetermined = 0

    for turn in possible_states:
        possible_states[turn] = {x for x in possible_states[turn] if (check_if_valid(x) != None)}

    all_states = []
    for turn in possible_states:
        for position in possible_states[turn]:
            result = check_if_valid(position)
            if result == 1:
                x_wins +=1
            if result == -1:
                o_wins += 1
            if result == 0:
                draws += 1
            if result == 2:
                undetermined += 1
            all_states.append(position)


    print(len(all_states))
    
    write_set_to_file(all_states, "Peter_Ladur_list_of_all_possible_states.txt")

    print(f"x wins: {x_wins}")
    print(f"o wins: {o_wins}")
    print(f"Draws: {draws}")
    print(f"Undetermined: {undetermined}")
    print(f"Total possible positions: {x_wins + o_wins + draws + undetermined}")


    pprint.pprint(Q_Table)

    with open("perfect_Q_Table.json", "w", encoding="utf-8") as f:
        json.dump(Q_Table, f, indent=2)


def write_set_to_file(my_set, filename):
    """
    Writes each element of a set to a text file, with each entry on a new line.

    Args:
        my_set: The set to write to the file.
        filename: The name of the file to write to.
    """
    try:
        with open(filename, 'w') as f:
            for item in my_set:
                f.write(str(item) + '\n')
        print(f"Set elements successfully written to '{filename}'.")
    except IOError as e:
        print(f"Error writing to file '{filename}': {e}")


def main():

    TOTAL_LENGTH = 0
    t1 = time.time()
    generate_all_possible_states()
    t2 = time.time()
    print(t2 - t1)

main()
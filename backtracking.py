"""
Backtracking is an algorithmic technique for generating all or some solutions
of a computational problem. Examples include:
    generating variations / combinations / permutations of a set of items;
    generating subsets of a set;
    completing a Sudoku puzzle;
    placing 8 queens on a chess board so that they do not attack each other.
"""

def dfs_backtrack(output, is_solution, add_to_output, children, candidate=""):
    """
        output: container solutions are added to
        is_solution: boolean function, true if candidate is a solution
        add_to_output: function that adds the input candidate to the output container
        children: function that returns child nodes/variant of the given node/candidate
    """
    if is_solution(candidate):
        add_to_output(candidate, output)
    else:
        for child_candidate in children(candidate):
            dfs_backtrack(output, is_solution, add_to_output, children, child_candidate)

def binary_numbers(desired_length, starting_candidate=""):
    def is_solution(candidate):
        return len(candidate) == desired_length
    def add_to_output(candidate, output):
        output.append("0b" + candidate)
    def children(candidate):
        return [candidate + "0", candidate + "1"]
    solutions = []
    dfs_backtrack(solutions, is_solution, add_to_output, children, starting_candidate)
    return solutions

def decimal_numbers(desired_length, starting_candidate=""):
    def is_solution(candidate):
        return len(candidate) == desired_length
    def add_to_output(candidate, output):
        output.append(candidate)
    def children(candidate):
        return [candidate + str(x) for x in range(10)]
    solutions = []
    dfs_backtrack(solutions, is_solution, add_to_output, children, starting_candidate)
    return solutions

def hexadecimal_numbers(desired_length, starting_candidate=""):
    def is_solution(candidate):
        return len(candidate) == desired_length
    def add_to_output(candidate, output):
        output.append("0x" + candidate)
    def children(candidate):
        symbols = [str(x) for x in range(10)]
        symbols += ['a','b','c','d','e','f']      
        return [candidate + str(x) for x in symbols]
    solutions = []
    dfs_backtrack(solutions, is_solution, add_to_output, children, starting_candidate)
    return solutions

def variations(desired_length, symbols=['a','b','c'], starting_candidate=""):
    """
    Variations is a selection of any number of each item and in any order
        n=size of set, k=desired_length
        V = n**k
    """
    def is_solution(candidate):
        return len(candidate) == desired_length
    def add_to_output(candidate, output):
        output.append(candidate)
    def children(candidate):     
        return [candidate + str(x) for x in symbols]
    solutions = []
    dfs_backtrack(solutions, is_solution, add_to_output, children, starting_candidate)
    return solutions

def permutations(desired_length, symbols=['a','b','c'], starting_candidate=""):
    """
    Permutation is the arrangement of items in which order matters
        n=size of set, k=desired_length
        P = n! / (n-k)!
    """
    def is_solution(candidate):
        return len(candidate) == desired_length
    def add_to_output(candidate, output):
        output.append(candidate)
    def children(candidate):     
        return [candidate + str(x) for x in symbols if str(x) not in candidate]
    solutions = []
    dfs_backtrack(solutions, is_solution, add_to_output, children, starting_candidate)
    return solutions

def combinations(desired_length, symbols=['a','b','c'], starting_candidate=""):
    """
    Combinations is the selection of items in which order does not matter
        n=size of set, k=desired_length
        C = (n! / (n-k)!) / k!
    """
    def is_solution(candidate):
        return len(candidate) == desired_length
    def add_to_output(candidate, output):
        output.append(candidate)
    def children(candidate):
        output = []
        for x in symbols:
            if str(x) not in candidate:
                child_candidate = candidate + str(x)
                child_set = set(child_candidate)
                if all([child_set != set(x) for x in solutions]):
                    output.append(child_candidate)
        return output
    solutions = []
    dfs_backtrack(solutions, is_solution, add_to_output, children, starting_candidate)
    return solutions

def subsets(desired_length, symbols=['a','b','c'], starting_candidate=""):
    """
    Generates all subsets of a given set, similar to combinations but adds to output as unique candidates are generated
        S = 2**n - 1
    """
    def is_solution(candidate):
        return len(candidate) == desired_length
    def add_to_output(candidate, output):
        if candidate not in output:
            output.append(candidate)
    def children(candidate):
        children = []
        for x in symbols:
            if str(x) not in candidate:
                child_candidate = candidate + str(x)
                child_set = set(child_candidate)
                if all([child_set != set(x) for x in solutions]):
                    children.append(child_candidate)
                    add_to_output(child_candidate, solutions)
        return children
    solutions = []
    dfs_backtrack(solutions, is_solution, add_to_output, children, starting_candidate)
    return solutions

def sudoko(table):
    """
    Brute force sudoku solver, accepts any square nxn table
        note that the solution doesn't adhear to the additional rule that internal squares contain 1-n
    """
    def is_solution(candidate):
        for i in range(len(candidate)):# ith row
            for j in range(len(candidate[0])):# jth col
                element = candidate[i][j]
                if element == None:
                    return False
        return True           
    def add_to_output(candidate, output):
        output.append(candidate)
    def children(candidate):
        children = []
        skip = False
        # traverse to the next free slot
        for i in range(len(candidate)):# ith row
            for j in range(len(candidate[0])):# jth col              
                if candidate[i][j] == None:# found free slot
                    skip = True
                    break
            if skip:
                break
        row = candidate[i]
        col = [candidate[x][j] for x in range(len(candidate))]
        for n in range(1,len(candidate)+1):# for each possible number that can be inserted
            # prune / ensure that n can be inserted here
            if n in row or n in col:
                continue
            # if so, create candidate child
            child = [[candidate[i][j] for j in range(len(candidate[0]))] for i in range(len(candidate))]
            # insert number into the free slot
            child[i][j] = n# i,j points to the free slot
            # add the child candidate to the children of the orignial candidate
            children.append(child)
        return children
    
    solutions = []
    dfs_backtrack(solutions, is_solution, add_to_output, children, table)
    return solutions

def n_queens():
    return

def main():
    def factorial(n):
        if n <= 0:
            return 1
        return n * factorial(n-1)

    # binary, decimal, hexadecimal numbers
    binary_digits = 4
    binaries = binary_numbers(binary_digits)
    print(int(binaries[-1],2) == 2**binary_digits-1, len(binaries), binaries[:5], "...", binaries[-5:])

    decimal_digits = 2
    decimals = decimal_numbers(decimal_digits)
    print(int(decimals[-1],10) == 10**decimal_digits-1,len(decimals), decimals[:5], "...", decimals[-5:])
    
    hexadecimal_digits = 2
    hexadecimals = hexadecimal_numbers(hexadecimal_digits)
    print(int(hexadecimals[-1],16) == 16**hexadecimal_digits-1, len(hexadecimals), hexadecimals[:5], "...", hexadecimals[-5:])

    # variations, combinations, permutations
    length = 3
    symbols = ['a','b','c','d']
    print("\nsymbols:", symbols, "set_length:", length)

    symbols_variations = variations(length, symbols)
    print(len(symbols_variations) == len(symbols)**length, len(symbols_variations), "variations:",
          symbols_variations[:5], "...", symbols_variations[-5:])
    
    symbols_permutations = permutations(length, symbols)
    print(len(symbols_permutations) ==
          factorial(len(symbols))/factorial(len(symbols)-length), len(symbols_permutations), "permutations:", symbols_permutations)

    symbols_combinations = combinations(length, symbols)
    print(len(symbols_combinations) ==
          (factorial(len(symbols))/factorial(len(symbols)-length))/factorial(length), len(symbols_combinations),
          "combinations:", symbols_combinations)

    # binary,decimal,hexadecimal numbers generated using variations function
    b = ['0b' + x for x in variations(2,['0','1'])]
    d = [x for x in variations(2,[i for i in range(10)])]
    h = ['0x' + x for x in variations(2,[i for i in range(10)] + ['a','b','c','d','e','f'])]
    if b != binary_numbers(2) or d != decimal_numbers(2) or h != hexadecimal_numbers(2):
        print("error")
        
    print(2**len(symbols)-1 == len(subsets(len(symbols), symbols)), 2**len(symbols)-1, "subsets:", subsets(len(symbols), symbols))

    # testing various unique sudoku solutions (without the iternal square rule)
    def print_table(table):
        print("-"*len(table[0])*5)
        for i in range(len(table)):
            row = ""
            for j in range(len(table[0])):
                element = str(table[i][j])
                if len(element) < 4:
                    element = " " + element + " "*(3-len(element))
                row += element + "|"
            print(row)
        print("-"*len(table[0])*5)

    sudoko_table = [
        [None,None,None],
        [None,None,None],
        [None,None,None]
        ]
    solution = [
        [1   ,2   ,3   ],
        [2   ,3   ,1   ],
        [3   ,1   ,2   ]
        ]
    print("\nMultiple solution Sudoku 3x3",solution in sudoko(sudoko_table))
    print_table(sudoko_table)
    for table in sudoko(sudoko_table):
        print_table(table)
 
    sudoko_table = [
        [1   ,None,3   ],
        [2   ,None,1   ],
        [None,1   ,None]
        ]
    solution = [
        [1   ,2   ,3   ],
        [2   ,3   ,1   ],
        [3   ,1   ,2   ]
        ]
    print("\nSudoku 3x3",sudoko(sudoko_table)[0]==solution and len(sudoko(sudoko_table))==1)
    print_table(sudoko_table)
    print_table(sudoko(sudoko_table)[0])
    
    sudoko_table = [
        [1   ,None,3   ,4   ],
        [2   ,None,1   ,None],
        [None,1   ,None,None],
        [None,None,None,None]
        ]
    solution = [
        [1   ,2   ,3   ,4   ],
        [2   ,4   ,1   ,3   ],
        [3   ,1   ,4   ,2   ],
        [4   ,3   ,2   ,1   ]
        ]
    print("\nSudoku 4x4",sudoko(sudoko_table)[0]==solution and len(sudoko(sudoko_table))==1)
    print_table(sudoko_table)
    print_table(sudoko(sudoko_table)[0])

    sudoko_table = [
        [1   ,5   ,3   ,4   ,None],
        [2   ,None,1   ,None,None],
        [None,1   ,None,None,3   ],
        [None,None,None,None,None],
        [None,4   ,None,None,1   ]
        ]
    solution = [
        [1   ,5   ,3   ,4   ,2   ],
        [2   ,3   ,1   ,5   ,4   ],
        [4   ,1   ,5   ,2   ,3   ],
        [3   ,2   ,4   ,1   ,5   ],
        [5   ,4   ,2   ,3   ,1   ]
        ]
    print("\nSudoku 5x5",sudoko(sudoko_table)[0]==solution and len(sudoko(sudoko_table))==1)
    print_table(sudoko_table)
    print_table(sudoko(sudoko_table)[0])

    sudoko_table = [
        [1   ,None,None,4   ,None,None],
        [2   ,None,1   ,None,None,3   ],
        [None,1   ,None,None,3   ,None],
        [None,None,6   ,None,None,None],
        [None,4   ,None,None,1   ,6   ],
        [6   ,None,2   ,None,None,None]
        ]
    solution = [
        [1   ,2   ,3   ,4   ,6   ,5   ],
        [2   ,6   ,1   ,5   ,4   ,3   ],
        [5   ,1   ,4   ,6   ,3   ,2   ],
        [4   ,5   ,6   ,3   ,2   ,1   ],
        [3   ,4   ,5   ,2   ,1   ,6   ],
        [6   ,3   ,2   ,1   ,5   ,4   ]
        ]
    print("\nSudoku 6x6",sudoko(sudoko_table)[0]==solution and len(sudoko(sudoko_table))==1)
    print_table(sudoko_table)
    print_table(sudoko(sudoko_table)[0])

    sudoko_table = [
        [None,None,3   ,None,None,None,None,8   ,None],
        [2   ,None,None,5   ,6   ,None,8   ,9   ,None],
        [None,4   ,5   ,None,7   ,None,9   ,None,2   ],
        [4   ,None,None,None,None,None,None,2   ,3   ],
        [None,None,None,None,9   ,1   ,None,None,None],
        [None,7   ,None,9   ,1   ,2   ,None,4   ,5   ],
        [None,None,9   ,1   ,2   ,None,4   ,None,None],
        [8   ,9   ,None,2   ,None,None,None,6   ,7   ],
        [None,1   ,None,None,4   ,None,6   ,7   ,None],
        ]
    solution = [
        [1   ,2,3,4   ,5,6,7,8,9],
        [2   ,3,4,5   ,6,7,8,9,1],
        [3   ,4,5,6   ,7,8,9,1,2],
        [4   ,5,6,7   ,8,9,1,2,3],
        [5   ,6,7,8   ,9,1,2,3,4],
        [6   ,7,8,9   ,1,2,3,4,5],
        [7   ,8,9,1   ,2,3,4,5,6],
        [8   ,9,1,2   ,3,4,5,6,7],
        [9   ,1,2,3   ,4,5,6,7,8],
        ]
    print("\nSudoku 9x9",sudoko(sudoko_table)[0]==solution and len(sudoko(sudoko_table))==1)
    print_table(sudoko_table)
    print_table(sudoko(sudoko_table)[0])
 
if __name__ == '__main__':
    main()



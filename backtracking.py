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
    return

def suduko():
    return

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
        

    
if __name__ == '__main__':
    main()
    

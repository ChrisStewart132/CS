# binary search that takes a sorted array and target, returns the index of target in the array otherwise -1

def search(arr, target, l=0, r=None):
    """
    Checks middle element, searches recursively left / right of the middle.
    """
    if r == None:
        r = len(arr)

    m = l + (r-l) // 2# same as (l+r) // 2
    
    if arr[m] == target:
        return m
    elif l >= r:
        return l if l == r and arr[l] == target else -1
    elif arr[m] < target:
        return search(arr, target, m+1, r)
    else:
        return search(arr, target, l, m-1)

def search_iterative(arr, target):
    """
    Iterative implementation of search
    """
    l, r = 0, len(arr)
    while l < r:
        i = (l + r) // 2
        if arr[i] == target:
            return i
        elif arr[i] < target:
            l = i + 1 
        else:
            r = i      
    return l if arr[l] == target else -1

def search2(arr, target, l=0, r=None):
    """
    Neglects checking until only 1 element exists (less "if" stmts but log(n) searches).
    """
    if r == None:
        r = len(arr)

    m = l + (r-l) // 2# same as (l+r) // 2
    
    if l == r:
        return l if arr[l] == target else -1
    elif arr[m] < target:
        return search(arr, target, m+1, r)
    else:
        return search(arr, target, l, m)

def search2_iterative(arr, target):
    """
    Iterative implementation of search2
    """
    l, r = 0, len(arr)
    while l < r:
        i = (l + r) // 2
        if arr[i] < target:
            l = i + 1 
        else:
            r = i      
    return l if arr[l] == target else -1

def main():
    sorted_array = [i for i in range(2**20)]
    random_elements = [random.randint(0,2**20-1) for i in range(2**8)]
    binary_search_functions = [search, search_iterative, search2, search2_iterative]
    binary_search_function_names = ['search', 'search_iterative', 'search2', 'search2_iterative']

    for f in range(len(binary_search_functions)):
        start = time.perf_counter()
        for i in range(len(random_elements)):
            if binary_search_functions[f](sorted_array, random_elements[i]) != random_elements[i]:
                print("error", binary_search_function_names[f])
                return
        finish = time.perf_counter()
        print(f"{finish-start:.6f}s ({binary_search_function_names[f]})")

        if binary_search_functions[f](sorted_array, -1) != -1:# make sure elements not found return -1
            print("error", binary_search_function_names[f])
            return 

if __name__ == '__main__':
    # tests
    import random, time
    main()




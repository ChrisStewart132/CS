# binary search that takes an array, target and returns the index of target in the array otherwise -1

def search(arr, target, l=0, r=None):
    '''checks middle element, searches recursively left / right of the middle'''
    if r == None:
        r = len(arr)-1

    m = l + (r-l) // 2# same as (l+r) // 2
    
    if arr[m] == target:
        return m
    elif l >= r:
        return l if l == r and arr[l] == target else -1
    elif arr[m] < target:
        return search(arr, target, m+1, r)
    else:
        return search(arr, target, l, m-1)

def search2(arr, target, l=0, r=None):
    '''neglects checking until only 1 element exists (less "if" stmts but log(n) searches)'''
    if r == None:
        r = len(arr)-1

    m = l + (r-l) // 2# same as (l+r) // 2
    
    if l == r:
        return l if arr[l] == target else -1
    elif arr[m] < target:
        return search(arr, target, m+1, r)
    else:
        return search(arr, target, l, m)


import random

x = sorted([random.randint(-2**8,2**8) for i in range(20)])
print(x)

# 0,1,2...,n
print([search(x, i) for i in x])
print([search2(x, i) for i in x])

# mostly -1,-1,...,-1
print([search(x, i-1) for i in x])
print([search2(x, i-1) for i in x])

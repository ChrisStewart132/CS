def selection_sort(arr):
    '''start at back of arr, swap current with the largest element, continue with element-1 and largest element in the un-sorted partition...'''
    for fill_slot in range(len(arr) -1, 0, -1):# last element to first
        index_of_max = 0
        for location in range(1, fill_slot +1):# 1 -> current element inclusive
            if arr[location] > arr[index_of_max]:# find largest element in un-sorted section
                index_of_max = location
        # swap items in fill_slot and index_of_max - puts max into fill_slot
        arr[fill_slot], arr[index_of_max] = arr[index_of_max], arr[fill_slot]
    return arr


def insertion_sort(arr):
    '''starts with arr[0] as sorted partition, incrementally inserts elements into the parition (swapping all elements in its path)'''
    for i in range(1, len(arr)):
        current = arr[i]
        while i > 0:
            if arr[i-1] > current:# keep moving previous elements up if they're > current
                arr[i] = arr[i-1]
                i -= 1
            else:# current index i is the correct position for the current element
                break
        arr[i] = current
    return arr


import random

x =[random.randint(-2**8,2**8) for i in range(22)]
a,b,c = [i for i in x], [i for i in x], [i for i in x]

print("unsorted", a==b==c)
print(a)
print(b)
print(c)

selection_sort(a)
insertion_sort(b)
c = sorted(c)

print("sorted", a==b==c)
print(a)
print(b)
print(c)


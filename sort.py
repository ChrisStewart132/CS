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


def gap_insertion_sort(arr, start, gap):
    '''In-place insertion sort on arr with given start and gap, used in shell_sort'''
    for i in range (start+gap, len(arr), gap):
        current = arr[i]
        while i >= gap:
            if arr[i-gap] > current:
                arr[i] = arr[i-gap]
                i -= gap
            else:
                break
        arr[i] = current
    return arr


def shell_sort(arr):
    '''Runs shell sort with gap starting at n//2 and then gap = gap // 2 etc'''
    gap = len(arr) // 2
    gap_list = []
    while gap > 0:
        gap_list.append(gap)  # build a list of gaps used as we go
        for start_position in range(gap):
            gap_insertion_sort(arr, start_position, gap)
        gap = gap // 2
    return arr

def merge(left, right, l=0, r=0):
    result = []
    while l < len(left) and r < len(right):
        if left[l] < right[r]:
            result.append(left[l])
            l+=1
        else:
            result.append(right[r])
            r+=1

    # append the remaining sorted list
    if l < len(left):
        result += left[l:]
    else:
        result += right[r:]

    return result

def merge_sort(nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    if len(nums) == 1:
        return nums        
    left = merge_sort(nums[:len(nums)//2])
    right = merge_sort(nums[len(nums)//2:])
    return merge(left, right)

import random

x =[random.randint(-2**8,2**8) for i in range(22)]
a,b,c,d,e = [i for i in x], [i for i in x], [i for i in x], [i for i in x], [i for i in x]

print("unsorted", a==b==c==d==e)
print(a)
print(b)
print(c)
print(d)
print(e)

selection_sort(a)
insertion_sort(b)
c = sorted(c)
shell_sort(d)
e = merge_sort(e)

print("sorted", a==b==c==d==e)
print(a)
print(b)
print(c)
print(d)
print(e)

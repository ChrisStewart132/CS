def selection_sort(arr):
    """
    start at back of arr, swap current with the largest element, continue with element-1 and largest
    element in the un-sorted partition
    """
    for fill_slot in range(len(arr) -1, 0, -1):# last element to first
        index_of_max = 0
        for location in range(1, fill_slot +1):# 1 -> current element inclusive
            if arr[location] > arr[index_of_max]:# find largest element in un-sorted section
                index_of_max = location
        # swap items in fill_slot and index_of_max - puts max into fill_slot
        arr[fill_slot], arr[index_of_max] = arr[index_of_max], arr[fill_slot]
    return arr


def insertion_sort(arr):
    """
    starts arr[0] as sorted partition, inserts remaining elements into the parition (swapping elements in its path)       
    """
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
    """
    In-place insertion sort on arr with given start and gap, used in shell_sort
    """
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
    """
    Runs shell sort with gap starting at n//2 and then gap = gap // 2 etc
    """
    gap = len(arr) // 2
    gap_list = []
    while gap > 0:
        gap_list.append(gap)  # build a list of gaps used as we go
        for start_position in range(gap):
            gap_insertion_sort(arr, start_position, gap)
        gap = gap // 2
    return arr

def merge(left, right, l=0, r=0):
    """
    merge operation used in merge sort
    """
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

def merge_sort(arr):
    """
    creates a sorted copy of the given array
    """
    if len(arr) == 1:
        return arr        
    left = merge_sort(arr[:len(arr)//2])
    right = merge_sort(arr[len(arr)//2:])
    return merge(left, right)

def quicksort(values, style='left-pivot'):
    """
    Starts the quicksort algorithm for sorting a list of values in-place
    """
    copy_of_list = list(values)

    if len(copy_of_list) == 1:
        # return the copy of the 1 item list
        return copy_of_list
    else:
        # Quicksort the copy of the list
        quicksort_helper(copy_of_list, 0, len(copy_of_list) - 1, style)
        return copy_of_list

def quicksort_helper(values, left, right, style):
    """
    Recursive quicksort helper.
    Sorts, in place, the portion of values between left and right.
    """
    # Stop when the left and right indices cross
    if left >= right:
        return

    # Partition the list
    split = partition(values, left, right, style)

    # Sort the left part
    quicksort_helper(values, left, split - 1, style)

    # Sort the right part
    quicksort_helper(values, split + 1, right, style)

def partition(values, left, right, style):
    """
    Partitions the values between left and right.
    Returns the index of the split.
    if style='left-pivot' then left item used as pivot
    if sytle='mo3-pivot' then index of median of three
     used as pivot
    if sytle is unknown then left-pivot is used.
    """

    # Figure out which index to use as the pivot
    if style == 'left-pivot':
        pivot_i = left
    elif style == 'mo3-pivot':
        pivot_i = pivot_index_mo3(values, left, right)
    else:
        print('Default left-pivot used...')
        pivot_i = left

    # Swap the pivot with the left item so we can keep the pivot
    # out of the way
    values[left], values[pivot_i] = values[pivot_i], values[left]

    # the pivot value is now the value in the left slot
    pivot = values[left]

    # move leftmark to first item after the pivot
    leftmark = left + 1
    rightmark = right

    # Move the left and right marks
    while True:
        # Find an item larger or equal to the pivot
        while leftmark <= rightmark and values[leftmark] < pivot:
            leftmark += 1
        # Find an item smaller than the pivot
        while leftmark <= rightmark and values[rightmark] >= pivot:
            rightmark -= 1

        # If the pointers cross, we're done
        if leftmark > rightmark:
            break
        else:
            # Otherwise... swap the items and keep going
            values[leftmark], values[rightmark] = values[
                rightmark], values[leftmark]

    # Put the pivot in its correct place
    values[left], values[rightmark] = values[rightmark], values[left]

    # Return the location of the split
    # values to right of rightmark are >= pivot value
    # values to left of rightmark are < pivot value
    return rightmark

def quicksort_range(values, start, end, style='left-pivot'):
    """
    Starts a quicksort that only guarantees that values between
    the start and end index (inclusive) are sorted.
    """
    copy_of_list = list(values)
    if len(copy_of_list) == 1:
        # return the copy of the only item in list
        return copy_of_list
    else:
        # Quicksort the copy of the list
        quicksort_range_helper(copy_of_list,
                               0,
                               len(copy_of_list) - 1,
                               start,
                               end,
                               style)
        return copy_of_list

def quicksort_range_helper(values, left, right, start, end, style):
    """
    Recursive quicksort range helper.
    Sorts, in place, the portion of values between left and right (inclusive)
    but only if the left-right range has any overlap with the start-end range.
    """
    # Stop when the left and right indices cross
    if left>=right or left > end or right < start:
        return

    # Partition the list
    split = partition(values, left, right, style)

    # Sort the left part
    quicksort_range_helper(values, left, split - 1, start, end, style)

    # Sort the right part
    quicksort_range_helper(values, split + 1, right, start, end, style)

def pivot_index_mo3(arr, left, right):
    """
    Returns the index of the item that is the median of the left, right and
    middle value in the list. The return value should normally be
    either left, right or middle.
    If there are only two items in the range, ie, if right==left+1
    then return the index of the first item as there are only two items
    to find the median of, so we can't get a middle index...
    If there is only one item in the range then also simply
    return the left index, ie, if left==right then return left.
    """
    if right-left < 2:
        return left
    middle = (left + right) // 2
    x = (arr[left], arr[middle], arr[right])
    return x[x.index(sorted(x)[1])]


def main():
    in_place_sort_functions = [selection_sort, insertion_sort, shell_sort]
    copy_sort_functions = [merge_sort, quicksort]

    # random un-sorted array
    unsorted_array =[random.randint(-2**8,2**8) for i in range(22)]

    # copies of the random un-sorted array for each sorting method
    arrays = [[i for i in unsorted_array] for j in range(len(in_place_sort_functions) + len(copy_sort_functions))]

    # sorted copy of unsorted_array to test against
    sorted_array = sorted(unsorted_array)

    print("unsorted", all([arrays[i] == unsorted_array for i in range(len(arrays))]))
    [print(a) for a in arrays]

    for i in range(len(in_place_sort_functions)):
        in_place_sort_functions[i](arrays[i])
        
    for i in range(len(copy_sort_functions)):
        arrays[i+len(in_place_sort_functions)] = copy_sort_functions[i](arrays[i+len(in_place_sort_functions)])

    print("sorted", all([arrays[i] == sorted_array for i in range(len(arrays))]))
    [print(a) for a in arrays]

if __name__ == '__main__':
    # tests
    import random
    main()


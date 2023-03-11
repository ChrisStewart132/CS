import heap, binary_search_tree, time

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

def chatGPT_selection_sort(arr):
    for i in range(len(arr)):
        # Find the minimum element in the unsorted portion of the array
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
                
        # Swap the minimum element with the first element of the unsorted portion
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
        
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

def chatGPT_insertion_sort(arr):
    for i in range(1, len(arr)):
        # Store the current element and its index
        current = arr[i]
        j = i - 1
        
        # Shift all larger elements to the right
        while j >= 0 and arr[j] > current:
            arr[j+1] = arr[j]
            j -= 1
        
        # Insert the current element in the correct position
        arr[j+1] = current
        
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

def chatGPT_shell_sort(arr):
    # Start with a large gap, then reduce the gap
    n = len(arr)
    gap = n // 2
    while gap > 0:
        # Do a gapped insertion sort for this gap size
        for i in range(gap, n):
            current = arr[i]
            j = i
            while j >= gap and arr[j - gap] > current:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = current
        gap //= 2
        
    return arr


def merge_simple(left, right, l=0, r=0):
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
    while l < len(left):
        result.append(left[l])
        l += 1
    while r < len(right):
        result.append(right[r])
        r += 1

    return result

def merge_sort_simple(arr):
    """
    creates a sorted copy of the given array
    """
    if len(arr) == 1:
        return arr        
    left = merge_sort_simple(arr[:len(arr)//2])
    right = merge_sort_simple(arr[len(arr)//2:])
    return merge_simple(left, right)

def merge(arr, left_slice, right_slice):
    """
    merge operation used in merge sort but the two arrays are combined with indices used to divide them
    """
    l, r = 0, right_slice[0] - left_slice[0]
    len_l, len_r = left_slice[1]+1 - left_slice[0], right_slice[1]+1 - left_slice[0]
    
    result = []
    while l < len_l and r < len_r:
        if arr[l] < arr[r]:
            result.append(arr[l])
            l+=1
        else:
            result.append(arr[r])
            r+=1

    # append the remaining sorted list
    while l < len_l:
        result.append(arr[l])
        l += 1
    while r < len_r:
        result.append(arr[r])
        r += 1

    return result

def _slice_array(left, right):
    """
    given left and right indices of an array, return two sets of left and right indices representing two halves
    """
    middle = (left+right) // 2
    left_slice = left, middle
    right_slice = middle + 1, right
    return left_slice, right_slice

def _merge_sort(arr, l=0, r=None):
    """
    implementation of merge_sort
    """
    if l == r:
        return [arr[l]]
    
    left_slice, right_slice = _slice_array(l, r)

    left = _merge_sort(arr, left_slice[0], left_slice[1])
    right = _merge_sort(arr, right_slice[0], right_slice[1])
    return merge(left+right, left_slice, right_slice)

def merge_sort(arr):
    """
    creates a sorted copy of the given array
    """
    return _merge_sort(arr, 0, len(arr)-1)

def chatGPT_merge_sort(arr):
    if len(arr) <= 1:
        return arr

    # Divide the array into two sub-arrays
    mid = len(arr) // 2
    left_arr = arr[:mid]
    right_arr = arr[mid:]

    # Recursively sort the sub-arrays
    left_arr = chatGPT_merge_sort(left_arr)
    right_arr = chatGPT_merge_sort(right_arr)

    # Merge the sorted sub-arrays
    i = j = k = 0
    while i < len(left_arr) and j < len(right_arr):
        if left_arr[i] < right_arr[j]:
            arr[k] = left_arr[i]
            i += 1
        else:
            arr[k] = right_arr[j]
            j += 1
        k += 1

    while i < len(left_arr):
        arr[k] = left_arr[i]
        i += 1
        k += 1

    while j < len(right_arr):
        arr[k] = right_arr[j]
        j += 1
        k += 1

    return arr


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

def chatGPT_quicksort(arr):
    """
    Sorts an array in ascending order using the quicksort algorithm.

    Args:
        arr: An array of integers to sort.

    Returns:
        The sorted array.
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return chatGPT_quicksort(left) + middle + chatGPT_quicksort(right)


def heap_sort(arr):
    h = heap.MinHeap(arr)
    return h.sorted()

def heap_sort_iterative(arr):
    h = heap.MinHeapIterative(arr)
    return h.sorted()

def tree_sort(arr):
    t = binary_search_tree.BinarySearchTree()
    for i in arr:
        t.insert(i)
    return t.in_order_items_with_duplicates()

def counting_sort(arr, key=lambda x:x**2):
    """
    Uses the positions obtained in the first stage to place the elements of
    the input at the right position in the output array
    """    
    smallest = min(arr)# to allow for sorting negative values
    arr_copy = [a-smallest for a in arr] if smallest < 0 else arr

    output = [0 for x in arr_copy]# output sorted array
    P = key_positions(arr_copy, key)
    for a in arr_copy:
        output[P[key(a)]] = a+smallest# places a in the correct position     
        P[key(a)] = P[key(a)] + 1
    return output

def key_positions(arr, key):
    """
    O(max(k,n)) where n = len(arr), k = largest key value output from a in arr
    Counts the number of times each key value occurs in the input array
    and uses this information to compute the position of objects with that
    key in the output (sorted array). Returns an array of positions len(k+1)
    """
    k = max([key(a) for a in arr])# largest key function result from all a in arr
    C = [0 for x in range(k+1)]# arr from 0->largest key function returned+1
    for a in arr:
        C[key(a)] = C[key(a)] + 1# counts a in arr who have same key output
    total = 0
    for i in range(k+1):
        C[i], total = total, total + C[i]# computes a running sum over C
    return C# C[i] is the number of elements whose key value is less than i

def chatGPT_counting_sort(arr):
    """
    Sorts an array of integers, including negative integers, in ascending order
    using the counting sort algorithm.

    Args:
        arr: An array of integers to sort.

    Returns:
        The sorted array.
    """
    if not arr:
        return arr

    min_val, max_val = min(arr), max(arr)
    count_arr = [0] * (max_val - min_val + 1)

    for x in arr:
        count_arr[x - min_val] += 1

    for i in range(1, len(count_arr)):
        count_arr[i] += count_arr[i - 1]

    sorted_arr = [0] * len(arr)
    for x in reversed(arr):
        index = count_arr[x - min_val] - 1
        sorted_arr[index] = x
        count_arr[x - min_val] -= 1

    return sorted_arr


def radix_sort(arr, d=3):
    """
    sorts arr based on its digits using a stable sort (counting sort)
    d=3 used as default
    """
    for i in range(1, d+1):
        arr = counting_sort(arr, lambda x:x%pow(10,i))
    return arr

def chatGPT_radix_sort(arr):
    """
    Sorts the input array using radix sort algorithm.
    Works with negative elements as well.
    """
    def _counting_sort(arr, exp):
        """
        Performs counting sort on the given array based on the given exponent.
        """
        n = len(arr)
        output = [0] * n

        # Count the occurrences of each digit
        count = [0] * 10
        for i in range(n):
            digit = (arr[i] // exp) % 10
            count[digit] += 1

        # Compute the cumulative counts
        for i in range(1, 10):
            count[i] += count[i - 1]

        # Place the elements in their correct positions
        for i in range(n - 1, -1, -1):
            digit = (arr[i] // exp) % 10
            output[count[digit] - 1] = arr[i]
            count[digit] -= 1

        # Copy the output array to the input array
        for i in range(n):
            arr[i] = output[i]

    # Separate positive and negative subarrays
    pos_arr, neg_arr = [], []
    for x in arr:
        if x >= 0:
            pos_arr.append(x)
        else:
            neg_arr.append(-x)  # Make negative numbers positive for now

    # Sort positive subarray using radix sort
    if pos_arr:
        max_val = max(pos_arr)
        exp = 1
        while max_val // exp > 0:
            _counting_sort(pos_arr, exp)
            exp *= 10

    # Sort negative subarray using radix sort
    if neg_arr:
        max_val = max(neg_arr)
        exp = 1
        while max_val // exp > 0:
            _counting_sort(neg_arr, exp)
            exp *= 10

        # Convert negative numbers back to their original form
        neg_arr = [-x for x in reversed(neg_arr)]

    # Combine the two sorted subarrays
    return neg_arr + pos_arr




    
def main():
    in_place_sort_functions = [selection_sort, chatGPT_selection_sort, insertion_sort, chatGPT_insertion_sort, shell_sort, chatGPT_shell_sort]
    copy_sort_functions = [merge_sort_simple, merge_sort, chatGPT_merge_sort, quicksort, chatGPT_quicksort, heap_sort, heap_sort_iterative,
                           tree_sort, counting_sort, chatGPT_counting_sort, radix_sort, chatGPT_radix_sort]
    function_names = ['selection_sort', 'chatGPT_selection_sort', 'insertion_sort', 'chatGPT_insertion_sort', 'shell_sort', 'chatGPT_shell_sort',
                      'merge_sort_simple', 'merge_sort', 'chatGPT_merge_sort', 'quicksort', 'chatGPT_quicksort',
                      'heap_sort', 'heap_sort_iterative', 'tree_sort', 'counting_sort', 'chatGPT_counting_sort', 'radix_sort', 'chatGPT_radix_sort']
    sorting_time = [0 for n in function_names]
    
    # random un-sorted array
    unsorted_array =[random.randint(-2**8,2**8) for i in range(10000)]
    
    # copies of the random un-sorted array for each sorting method
    arrays = [[i for i in unsorted_array] for j in range(len(in_place_sort_functions) + len(copy_sort_functions))]

    # sorted copy of unsorted_array to test against
    sorted_array = sorted(unsorted_array)

    print("Unsorted:", all([arrays[i] == unsorted_array for i in range(len(arrays))]))

    for i in range(len(in_place_sort_functions)):
        start = time.perf_counter()
        in_place_sort_functions[i](arrays[i])
        finish = time.perf_counter()
        sorting_time[i] = finish - start
        print(f" {sorting_time[i]:.6f}s,", function_names[i])
        
    for i in range(len(copy_sort_functions)):
        start = time.perf_counter()
        index = i+len(in_place_sort_functions)
        arrays[index] = copy_sort_functions[i](arrays[index])
        finish = time.perf_counter()
        sorting_time[index] = finish - start
        print(f" {sorting_time[index]:.6f}s,", function_names[index])

    success = all([arrays[i] == sorted_array for i in range(len(arrays))])
    if not success:
        for i, arr in enumerate(arrays):
            print(function_names[i], arr[:6],"...",arr[-6:])
    print("Sorted:", success)
    print(f"Total sorting time: {sum(sorting_time):.6f}s")
    print(f"Average sorting time: {sum(sorting_time) / len(sorting_time):.6f}s")

if __name__ == '__main__':
    # tests
    import random
    main()


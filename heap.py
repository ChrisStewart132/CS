class Heap(object):
    """An abstract interface for a Heap."""
    def __init__(self, items=None, fast=True):
        # Create a list to store heap items.
        # First item is simply a spacer
        # Heap contents will start from index 1
        if items:
            if fast:
                self.fast_heapify(items)
            else:                  
                self.slow_heapify(items)
        else:
            self._items = [None]    

    def slow_heapify(self, items):
        """insert initial items 1 at a time by appending each and sifting up"""
        self._items = [None]    
        for item in items:
            self.insert(item)
        return self._items

    def fast_heapify(self, items):
        """
        heapify the given list, because a heap is bottom heavy it is more efficient (in terms of complexity) to instead sift down

        However, sifting down during heapify is generally slower than sifting up because sifting up involves swapping the new element with its
        parent repeatedly until it reaches its correct position. On the other hand, sifting down involves swapping the parent with one
        of its children, then recursively sifting down that child until the heap property is restored. This process can involve more
        swaps than sifting up, leading to slower performance.

        Additionally, sifting down can lead to more cache misses since the elements being accessed are not in a contiguous memory location.
        In contrast, sifting up accesses elements in a contiguous memory location, which can lead to better cache performance and faster execution.
        """
        self._items = items
        self._items.append(None)# add None to back of items
        self._items[-1], self._items[0] = self._items[0], self._items[-1]# swap None to the front of the heap
        for i in range(len(self)//2, 0, -1):
            self._sift_down(i)
        return self._items

    def sorted(self):
        """pops the heap to a sorted array"""
        output = []
        while len(self) > 0:
            self._items[-1], self._items[1] = self._items[1], self._items[-1]# swap item at front of heap with rear         
            output.append(self._items.pop())
            self._sift_down(1)
        return output
            
    def insert(self, item):       
        pass# not implemented here

    def isEmpty(self):
        return len(self) == 0

    def __len__(self):
        """Returns the actual length of the heap,
        ie, how many items are in the heap
        Remember that item at index 0 is not part of the heap."""
        return len(self._items) - 1

    def __repr__(self):
        """Returns the _items list for the heap
        Note, there will be a None at index 0
        and this is just a place holder.
        The root value is at index 1"""
        return repr(self._items)


class MinHeap(Heap):
    """Implementation of a min-heap."""
    def insert(self, item):
        self._items.append(item)
        self._sift_up(len(self._items) - 1)

    def _sift_up(self, index):
        """
        Moves the item at the given index up through the heap until it finds
        the correct place for it. That is, the item is moved up through the heap
        while it is smaller than its parent.
        """
        parent = (index) // 2
        # While we haven't reached the top of the heap, and its parent is
        # smaller than the item
        if index > 1 and self._items[index] < self._items[parent]:
            # Swap the item and its parent
            self._items[index], self._items[parent] = self._items[parent], self._items[index]
            # Carry on sifting up from the parent index
            self._sift_up(parent)
        # else no more sifting needed as didn't swap.

    def peek_min(self):
        """
        Returns the smallest item in the heap.
        """
        return self._items[1]

    def pop_min(self):
        """
        Removes the smallest item in the heap and returns it. Returns None if
        there are no items in the heap. Can be thought of as Popping the min
        item off the heap.
        """
        if len(self._items) == 1:
            return None
        
        self._items[1], self._items[-1] = self._items[-1], self._items[1]#swap root and last item
        output = self._items.pop()
        self._sift_down(1)
        
        return output

    def _sift_down(self, index):
        """
        Moves an item at the given index down through the heap until it finds
        the correct place for it. That is, when the item is moved up through the
        heap while it is larger than either of its children.
        """
        # While the item at 'index' has at least one child...
        if (index * 2) <= len(self):
            smallest = 2 * index
            right = smallest + 1
            
            if right <= len(self) and self._items[smallest] > self._items[right]:
                smallest = right
                
            if self._items[index] > self._items[smallest]:
                self._items[smallest], self._items[index] = self._items[index], self._items[smallest]
                self._sift_down(smallest)

    def validate(self):
        """
        Validates the heap by ensuring each node is greater than its parent.
        Returns True if the heap is a valid min-heap, and False otherwise.
        """
        for i in range(2,len(self._items)):
            if self._items[i] < self._items[i//2]:
                return False
        return True

class MinHeapIterative(Heap):
    """Implementation of a min-heap."""
    def insert(self, item):
        self._items.append(item)
        self._sift_up(len(self._items) - 1)

    def _sift_up(self, index):
        """
        Moves the item at the given index up through the heap until it finds
        the correct place for it. That is, the item is moved up through the heap
        while it is smaller than its parent.
        """
        parent = (index) // 2
        while index > 1 and self._items[index] < self._items[parent]:
            self._items[index], self._items[parent] = self._items[parent], self._items[index]
            index = parent
            parent = (index) // 2

    def peek_min(self):
        """
        Returns the smallest item in the heap.
        """
        return self._items[1]

    def pop_min(self):
        """
        Removes the smallest item in the heap and returns it. Returns None if
        there are no items in the heap. Can be thought of as Popping the min
        item off the heap.
        """
        if len(self._items) == 1:
            return None
        
        self._items[1], self._items[-1] = self._items[-1], self._items[1]# swap root and last item
        output = self._items.pop()
        self._sift_down(1)
        
        return output

    def _sift_down(self, index):
        """
        Moves an item at the given index down through the heap until it finds
        the correct place for it. That is, when the item is moved up through the
        heap while it is larger than either of its children.
        """
        # While the item at 'index' has at least one child...
        while (index * 2) <= len(self):
            smallest = 2 * index
            right = smallest + 1
            
            if right <= len(self) and self._items[smallest] > self._items[right]:
                smallest = right
                
            if self._items[index] > self._items[smallest]:
                self._items[smallest], self._items[index] = self._items[index], self._items[smallest]
                index = smallest
            else:
                break

    def validate(self):
        """
        Validates the heap by ensuring each node is greater than its parent.
        Returns True if the heap is a valid min-heap, and False otherwise.
        """
        for i in range(2,len(self._items)):
            if self._items[i] < self._items[i//2]:
                return False
        return True


class MaxHeapIterative(Heap):
    """Implementation of a min-heap."""
    def insert(self, item):
        self._items.append(item)
        self._sift_up(len(self._items) - 1)

    def _sift_up(self, index):
        """
        Moves the item at the given index up through the heap until it finds
        the correct place for it.
        """
        parent = (index) // 2
        while index > 1 and self._items[index] > self._items[parent]:
            self._items[index], self._items[parent] = self._items[parent], self._items[index]
            index = parent
            parent = (index) // 2

    def peek_max(self):
        """
        Returns the largest item in the heap.
        """
        return self._items[1]

    def pop_max(self):
        """
        Removes the largest item in the heap and returns it. Returns None if
        there are no items in the heap. Can be thought of as Popping the max
        item off the heap.
        """
        if len(self._items) == 1:
            return None
        
        self._items[1], self._items[-1] = self._items[-1], self._items[1]# swap root and last item
        output = self._items.pop()
        self._sift_down(1)
        
        return output

    def _sift_down(self, index):
        """
        Moves an item at the given index down through the heap until it finds
        the correct place for it.
        """
        # While the item at 'index' has at least one child...
        while (index * 2) <= len(self):
            largest = 2 * index
            right = largest + 1
            
            if right <= len(self) and self._items[largest] < self._items[right]:
                largest = right
                
            if self._items[index] < self._items[largest]:
                self._items[largest], self._items[index] = self._items[index], self._items[largest]
                index = largest
            else:
                break

    def validate(self):
        """
        Validates the heap by ensuring each node is smaller than its parent.
        Returns True if the heap is a valid min-heap, and False otherwise.
        """
        for i in range(2,len(self._items)):
            if self._items[i] > self._items[i//2]:
                return False
        return True

class Max_3_Heap(Heap):
    """Implementation of a max-three-heap.
    Each child must be smaller than or equal to its parent.
    Each parent has up to 3 children.
    First element of the heap is stored in _items[1]
    left_child_index = parent_index * 3 - 1
    middle_child_index = parent_index * 3
    right_child_index = parent_index * 3 + 1
    parent = (child + 1) // 3
    """

    # Note: inherits the Heap __init__ method

    def insert(self, item):
        """
        Inserts a given item into the heap.
        """
        # Append the item to the end of the heap
        self._items.append(item)
        # Sift it up into place
        self._sift_up(len(self))

    def _sift_up(self, index):
        """
        Moves the item at the given index up through the heap until it finds
        the correct place for it. That is, the item is moved up through the heap
        while it is larger than its parent.
        """
        parent_index = (index + 1) // 3
        if index > 1 and self._items[parent_index] < self._items[index]:
            self._items[parent_index], self._items[index] = self._items[index], self._items[parent_index]
            self._sift_up(parent_index)

    def peek_max(self):
        """
        Returns the largest value in the heap, ie, the top of the heap. Doesn't change the heap.      
        """
        if len(self) > 0:
            return self._items[1]
        else:
            return None

    def pop_max(self):
        """
        Removes the largest item in the heap and returns it. Returns None if
        there are no items in the heap. Can be thought of as Popping the max
        item off the heap.
        """
        if len(self) == 0:
            return None
        
        self._items[1], self._items[-1] = self._items[-1], self._items[1]# swap root and last item
        output = self._items.pop()
        self._sift_down(1)
       
        return output
    
    def _sift_down(self, index):
        """
        Moves an item at the given index down through the heap until it finds
        the correct place for it. That is, the item is moved down through the
        heap while it is smaller than any of its children.
        """
        if (index * 3) <= len(self._items):# confirms index has a child (left child)
            left = index * 3 - 1
            middle = index * 3
            right = index * 3 + 1

            largest = left
            if len(self._items) == middle:
                pass
            elif len(self._items) == right:
                largest = middle if self._items[middle] > self._items[largest] else largest
            else:
                largest = middle if self._items[middle] > self._items[largest] else largest
                largest = right if self._items[right] > self._items[largest] else largest

            if self._items[index] < self._items[largest]:
                self._items[largest], self._items[index] = self._items[index], self._items[largest]               
                self._sift_down(largest)

    def validate(self):
        """
        Validates the heap. Returns True if the heap is a valid max-3-heap, and False otherwise.        
        """
        for i in range(2, len(self._items)):
            if self._items[i] > self._items[(i+1) // 3]:
                return False
        return True
     


def main():
    import random, time

    # initialize heaps and an array of random integers
    min_heap = MinHeap()
    min_heap_iterative = MinHeapIterative()
    max_heap_iterative = MaxHeapIterative()
    max_heap = Max_3_Heap()
    x = [random.randint(-2**8,2**8) for i in range(3**6)]

    # slow heapify a new min heap with x
    start = time.perf_counter() 
    slow_heapified_heap = MinHeapIterative(x, False)
    finish = time.perf_counter()
    delta_t = finish - start
    print("slow_heapify valid:", slow_heapified_heap.validate(), f", {delta_t:6f}s")

    # fast heapify a new min heap with a copy of x
    x_copy = [i for i in x]
    start = time.perf_counter() 
    fast_heapified_heap = MinHeapIterative(x_copy, True)
    finish = time.perf_counter()
    delta_t = finish - start
    print("fast_heapify valid:", fast_heapified_heap.validate(), f", {delta_t:6f}s")

    # confirm the slow and fast heaps when sorted == sorted(x)
    print("slow.sorted == fast.sorted == sorted:", slow_heapified_heap.sorted()==fast_heapified_heap.sorted()==sorted(x))

    # insert random integers into the heap(s)
    for n in x:
        min_heap.insert(n)
        min_heap_iterative.insert(n)
        if not min_heap.validate() or not min_heap_iterative.validate():
            print("min_heap(s) error1")
            return -1
        
        max_heap.insert(n)
        max_heap_iterative.insert(n)
        if not max_heap.validate() or not max_heap_iterative.validate():
            print("max_heap(s) error1")
            return -1

    # confirm that the heap pops in the correct order
    print("min heap:", min_heap.sorted() == sorted(x))
    print("min heap iterative:", min_heap_iterative.sorted() == sorted(x))   
    print("max 3 heap:", max_heap.sorted() == sorted(x, reverse=True))
    print("max heap iterative:", max_heap_iterative.sorted() == sorted(x, reverse=True))


if __name__ == '__main__':
    main()


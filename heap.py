class Heap(object):
    """An abstract interface for a Heap."""
    def __init__(self):
        # Create a list to store heap items.
        # First item is simply a spacer
        # Heap contents will start from index 1
        self._items = [None]

    def insert(self, item):
        # don't implement here
        # this is just a place holder
        pass

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
            left = 2 * index
            right = left + 1
            smallest = left
            try:
                if self._items[left] < self._items[right]:
                    smallest = left
                else:
                    smallest = right
            except IndexError:
                smallest = left
            if self._items[index] > self._items[smallest]:
                self._items[smallest], self._items[index] = self._items[index], self._items[smallest]
                self._sift_down(smallest)

    def validate(self):
        """
        Validates the heap by ensuring each node is greater than its parent.
        Returns True if the heap is a valid min-heap, and False otherwise.
        """
        min_value = self.peek_min()
        for i in range(2,len(self._items)):
            if self._items[i] < self._items[i//2]:
                return False
        return True


        


def main():
    import random
    h = MinHeap()

    x = [random.randint(-2**8,2**8) for i in range(20)]
    # insert random integers into the heap
    for n in x:
        h.insert(n)
        if not h.validate():
            print("error")
            return -1

    # confirm that the heap pops in the correct order
    sorted_heap = []
    while not h.isEmpty():
        if not h.validate():
            print("error")
            return -1
        sorted_heap.append(h.pop_min())

    print(sorted_heap == sorted(x))


if __name__ == '__main__':
    main()


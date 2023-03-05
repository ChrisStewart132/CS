#TODO when popping / removing tail need to do so in an efficient way to replace the tail with prev

class Node:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
    def __repr__(self):
        return str(self)
    def __str__(self):
        return str(self.val)

class LinkedList():
    """
    operation          complexity
    
    __len__            O(1)
    __eq__             O(n)
    length             O(n)
    get_node(i)        O(n)
    get_value(i)       O(n)   
    enqueue            O(1)
    append             O(1)
    insert             O(n)
    dequeue            O(1)
    pop                O(n)
    remove             O(n)
    
    """
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def __len__(self):
        return self.size

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        elif type(other) in [list, tuple]:
            return all([self.get_value(i) == other[i] for i in range(len(self))])
        elif isinstance(other, LinkedList):
            return all([self.get_value(i) == other.get_value(i) for i in range(len(self))])
        return False

    def length(self, head=1):
        """
        Counts the number of nodes contained in the list
        """
        if head == 1:# init
            head = self.head
        if head == None:
            return 0
        return 1 + self.length(head.next)
    
    def get_node(self, index, head):
        """
        Returns ith Node (or None if index >= len) from the initial head node given
        """
        if index == 0 or head == None:
            return head
        return self.get_node(index-1, head.next)
        
    def get_value(self, index):
        """
        Returns the value at the given index position
        """
        if -1 < index < self.size:
            return self.get_node(index, self.head).val
        raise IndexError
    
    def enqueue(self, val):
        """
        enqueues a new value to the front of the list
        """
        self.head = Node(val, self.head)
        if self.size == 0:
            self.tail = self.head
        self.size += 1
        
    def append(self, val):
        """
        appends a new value to the back of the list
        """
        if self.size == 0:
            self.enqueue(val)
        else:
            self.tail.next = Node(val)
            self.tail = self.tail.next
            self.size += 1

    def insert(self, index, val):
        """
        Inserts new value at the given index position
        """
        if index == 0:
            self.enqueue(val)
        elif index == self.size:
            self.append(val)
        elif 0 < index < self.size:
            prev = self.get_node(index-1, self.head)
            if prev:
                prev.next = Node(val, prev.next)
                self.size += 1
        else:
            raise IndexError

    def dequeue(self):
        """
        removes and returns the head of the list
        """
        output = self.head
        if self.head:
            self.head = self.head.next
            self.size -= 1
        if output == self.tail and self.size == 0:
            self.tail = None
        if self.size == 0:# last element == head == tail removed
            self.tail = None
        return output

    def pop(self):
        """
        removes and returns the value of the tail node
        """
        if self.size > 0:
            return self.remove(self.size-1)
 
    def remove(self, index):
        """
        removes and returns the node with the given value at the index'th position
        """
        if index == 0:
            return self.dequeue()
        elif 0 < index < self.size:
            prev = self.get_node(index-1, self.head)
            if index == self.size-1:# when popping tail, save the new tail
                self.tail = prev
            output = prev.next
            prev.next = prev.next.next
            self.size -= 1
            return output
        else:
            raise IndexError

    def __repr__(self):
        return str(self)
    def __str__(self):
        head = str(self.head) if self.head else "None"
        tail = str(self.tail) if self.tail else "None"
        return "head:" + head + ", tail:" + tail + "\n[" + self._str(self.head)[:-2] + "]"
    def _str(self, head):
        return "" if head == None else str(head) + ", " + self._str(head.next)
        
       
def main():   
    linked_list = LinkedList()
    print(linked_list==[])
    
    # Insert all elements
    random_numbers = [random.randint(-2**8,2**8) for i in range(2**8)]
    print(linked_list!=random_numbers)
    for i, number in enumerate(random_numbers):        
        if len(linked_list) != i:
            print("insert error")
        linked_list.append(number)
    if not (len(linked_list) == len(random_numbers) == linked_list.length()):
        print("insert error")
    print(linked_list==random_numbers)

    # get, insert, remove invalid index
    tests = []
    tests.append(linked_list.get_value(len(linked_list)//2) == random_numbers[len(linked_list)//2])
    tests.append(linked_list.get_value(len(linked_list)//4) == random_numbers[len(linked_list)//4])
    tests.append(linked_list.get_value(len(linked_list)//8) == random_numbers[len(linked_list)//8])
    try:
        linked_list.get_value(len(linked_list))
        tests.append(False)
    except IndexError:
        tests.append(True)
    try:
        linked_list.insert(len(linked_list)+1, -1)
        tests.append(False)
    except IndexError:
        tests.append(True)
    try:
        linked_list.remove(len(linked_list))
        tests.append(False)
    except IndexError:
        tests.append(True)
    print(all(tests), tests)

        
    # Remove some elements from random positions
    to_remove = [random.randint(0, len(linked_list)-1) for i in range(len(linked_list)//4)]
    for i in range(len(to_remove)):
        n = random_numbers[i]# element to be removed
        random_numbers = random_numbers[:i] + random_numbers[i+1:]
        removed_element = linked_list.remove(i)# node that was removed
        if not (linked_list==random_numbers) or not (n == removed_element.val):
            print("remove error")
    if not (len(linked_list) == len(random_numbers) == linked_list.length()):
            print("remove error")
    print(linked_list==random_numbers)
    
    # add some elements to random positions
    to_add = [random.randint(0, len(linked_list)-1) for i in range(len(linked_list)//4)]
    for i in range(len(to_add)):
        number = random.randint(-2**8,2**8)
        random_numbers = random_numbers[:i] + [number] + random_numbers[i:]
        linked_list.insert(i, number)
        if not (linked_list==random_numbers):
            print("add error")
    if not (len(linked_list) == len(random_numbers) == linked_list.length()):
            print("add error")
    print(linked_list==random_numbers)
    
    # Remove remaining elements
    while len(linked_list) > 0:
        random_numbers.pop()
        linked_list.remove(len(linked_list)-1)
        if not (linked_list==random_numbers):
            print("remove error")
    if not (len(linked_list) == len(random_numbers) == linked_list.length() == 0):
            print("remove error")
    print(linked_list==random_numbers)

    # Test __eq__, pop, empty linked_list 
    tests = []
    random_numbers.append(0)                    # [], [0]
    tests.append(linked_list!=random_numbers)   # []==[0]
    linked_list.insert(0,1)                     # [1], [0]
    tests.append(linked_list!=random_numbers)   # [1]==[0]
    linked_list.insert(0,0)                     # [0,1], [0]
    random_numbers.append(1)                    # [0,1], [0,1]
    tests.append(linked_list==random_numbers)   # [0,1]==[0,1]
    tests.append(linked_list.pop().val==1)      # [0  ], [0,1]
    tests.append(linked_list!=random_numbers)   # [0  ]==[0,1]
    tests.append(random_numbers.pop()==1)       # [0  ]==[0  ]
    tests.append(linked_list==random_numbers)   # [0  ]==[0  ]
    tests.append(linked_list.pop().val==0)      # [   ]==[0  ]
    tests.append(random_numbers.pop()==0)       # [   ]==[   ]
    tests.append(linked_list==random_numbers)   # [   ]==[   ]
    tests.append(linked_list.pop()==None)
    tests.append(linked_list.size==0)
    tests.append(linked_list.pop()==None)
    tests.append(linked_list.size==0)
    tests.append(linked_list.dequeue()==None)
    tests.append(linked_list.size==0)
    tests.append(linked_list.length()==0)
    tests.append(len(linked_list)==0)
    print(linked_list)
    print(all(tests), tests)

if __name__ == '__main__':
    import random, time
    main()

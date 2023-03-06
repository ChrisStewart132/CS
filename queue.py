import linked_list

class Queue(object):
    def __init__(self, arr=None):
        if arr:
            for a in arr:
                self.enqueue(a)
    def enqueue(self, item):
        pass   
    def dequeue(self):
        pass
    def __len__(self):
        pass
    def __repr__(self):
        return str(self)
    def __str__(self):
        return str(self.queue)

class ListQueue(Queue):
    queue = []    
    def enqueue(self, item):
        self.queue.append(item)       
    def dequeue(self):
        item = self.queue[0]
        self.queue = self.queue[1:]
        return item
    def __len__(self):
        return len(self.queue)

class DoubleStackQueue(Queue):
    s1 = []
    s2 = []
    
    def enqueue(self, item):
        self.s1.append(item)
        
    def dequeue(self):
        if len(self.s2) > 0:
            return self.s2.pop()       
        while len(self.s1) > 0:
            item = self.s1.pop()
            self.s2.append(item)            
        return self.dequeue()
        
    def __len__(self):
        return len(self.s1) + len(self.s2)
    
class LinkedListQueue(Queue):
    def __init__(self, arr=None):
        self.queue = linked_list.LinkedList()
        super().__init__(arr)
    def enqueue(self, item):
        self.queue.append(item)   
    def dequeue(self):
        return self.queue.dequeue()   
    def __len__(self):
        return len(self.queue)
    

#class CircularBuffer(Queue):

def main():
    lq = ListQueue()
    dsq = DoubleStackQueue()
    llq = LinkedListQueue()
    random_items = [random.randint(-2**8, 2**8) for i in range(2**8)]
    length = 0
    
    print("enqueue, length")
    tests = []
    for item in random_items:
        lq.enqueue(item)
        dsq.enqueue(item)
        llq.enqueue(item)
        length += 1
        tests.append(len(lq) == len(dsq) == len(llq) == length)
    print("",all(tests))

    print("dequeue, length")
    for item in random_items:
        lq_item = lq.dequeue()
        dsq_item = dsq.dequeue()
        llq_item = llq.dequeue().val
        length -= 1
        tests.append(lq_item == dsq_item == llq_item == item and len(lq) == len(dsq) == len(llq) == length)
    print("",all(tests))
        

if __name__ == '__main__':
    # tests
    import random, time
    main()


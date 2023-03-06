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
    def __len__(self):
        return len(self.queue)

class ListQueue(Queue):
    queue = []    
    def enqueue(self, item):
        self.queue.append(item)       
    def dequeue(self):
        item = self.queue[0]
        self.queue = self.queue[1:]
        return item

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
    
class CircularBufferQueue(Queue):
    def __init__(self, k):
        self.queue = [0 for x in range(k)]
        self.head, self.tail, self.length = -1, -1, 0      
    def enqueue(self, value):
        if self.isFull():
            return None                 
        if self.isEmpty():
            self.head, self.tail = 0, 0
        else:
            self.tail = (self.tail + 1) % len(self.queue)        
        self.queue[self.tail] = value
        self.length += 1
        return self.rear()             
    def dequeue(self):
        if self.isEmpty():
            return None
        item = self.front()
        self.head = (self.head + 1) % len(self.queue)     
        self.length -= 1
        return item       
    def front(self):
        return self.queue[self.head] if not self.isEmpty() else -1       
    def rear(self):
        return self.queue[self.tail] if not self.isEmpty() else -1
    def isEmpty(self):
        return self.length == 0        
    def isFull(self):
        return self.length >= len(self.queue)
    def __len__(self):
        return self.length
        

def main():
    lq = ListQueue()
    dsq = DoubleStackQueue()
    llq = LinkedListQueue()
    cbq = CircularBufferQueue(2**8)
    random_items = [random.randint(-2**8, 2**8) for i in range(2**8)]
    length = 0
    
    print("enqueue, length")
    tests = []
    for item in random_items:
        lq.enqueue(item)
        dsq.enqueue(item)
        llq.enqueue(item)
        cbq.enqueue(item)
        length += 1
        tests.append(len(lq) == len(dsq) == len(llq) == len(cbq) == length)
    tests.append(len(lq) == len(dsq) == len(llq) == len(cbq) == length == len(random_items))
    print("",all(tests))

    print("dequeue, length")
    for item in random_items:
        lq_item = lq.dequeue()
        dsq_item = dsq.dequeue()
        llq_item = llq.dequeue().val
        cbq_item = cbq.dequeue()
        length -= 1
        tests.append(lq_item == dsq_item == llq_item == cbq_item == item and len(lq) == len(dsq) == len(llq) == len(cbq) == length)
    tests.append(lq_item == dsq_item == llq_item == cbq_item == item and len(lq) == len(dsq) == len(llq) == len(cbq) == length == 0)
    print("",all(tests))

    print("enqueue, dequeue, length")
    for i, item in enumerate(random_items):
        if i % 5 == 4:
            lq_item = lq.dequeue()
            dsq_item = dsq.dequeue()
            llq_item = llq.dequeue().val
            cbq_item = cbq.dequeue()
            length -= 1
            print(lq_item == dsq_item == llq_item == cbq_item == item, len(lq) == len(dsq) == len(llq) == len(cbq) == length)
            tests.append(lq_item == dsq_item == llq_item == cbq_item == item and len(lq) == len(dsq) == len(llq) == len(cbq) == length)
            print("",all(tests))
        else:
            lq.enqueue(item)
            dsq.enqueue(item)
            llq.enqueue(item)
            cbq.enqueue(item)
            length += 1
            tests.append(len(lq) == len(dsq) == len(llq) == len(cbq) == length)
            print("",all(tests))
            
        if not all(tests):
            print(lq)
            break
        
    print("",all(tests))
        

if __name__ == '__main__':
    # tests
    import random, time
    main()


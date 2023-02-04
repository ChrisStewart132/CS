class Node(object):
    """Represents a node in a binary tree."""
    def __init__(self, value):
        self.left = None
        self.right = None
        self.value = value
        self.count = 1

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "[l:{}, {}, r:{}]".format(repr(self.value),
                                         repr(self.left),
                                         repr(self.right))

class BinarySearchTree(object):
    """
    Implementation of a simple unbalanced Binary Search Tree (BST).
    """
    def __init__(self):
        self.root = None

    def insert(self, value):
        """
        Inserts a new item into the tree.
        """
        if self.root is None:
            self.root = Node(value)
        else:
            self._insert(self.root, value)

    def _insert(self, subtree_root, value):
        """
        Recursively locates the correct position in the subtree starting
        at 'subtree_root' to insert the given 'value',
        and attaches a Node containing the 'value' to the tree.
        """
        if value == subtree_root.value:# inserting duplicate value, add to node.count            
            subtree_root.count += 1
        elif value < subtree_root.value:           
            if subtree_root.left is None:# Insert to the left
                subtree_root.left = Node(value)
            else:
                self._insert(subtree_root.left, value)
        else:           
            if subtree_root.right is None:# Insert to the right
                subtree_root.right = Node(value)
            else:
                self._insert(subtree_root.right, value)

    def in_order_items(self):
        """
        Returns a sorted list of all items in the tree using in-order traversal.
        """
        out_list = []
        self._in_order_items(self.root, out_list)
        return out_list

    def _in_order_items(self, subtree_root, out_list):
        """
        In-order traversal, adding values from visited nodes to out_list. 
        """
        if subtree_root == None:
            return
        left = self._in_order_items(subtree_root.left, out_list)
        out_list.append(subtree_root.value)
        right = self._in_order_items(subtree_root.right, out_list)

    def in_order_items_with_duplicates(self):
        """
        Returns a sorted list of all items (including duplicates) in the tree using in-order traversal.
        """
        out_list = []
        self._in_order_items_with_duplicates(self.root, out_list)
        return out_list

    def _in_order_items_with_duplicates(self, subtree_root, out_list):
        """
        In-order traversal, adding values (including duplicates) from visited nodes to out_list. 
        """
        if subtree_root == None:
            return
        left = self._in_order_items_with_duplicates(subtree_root.left, out_list)
        for i in range(subtree_root.count):
            out_list.append(subtree_root.value)
        right = self._in_order_items_with_duplicates(subtree_root.right, out_list)


    def pre_order_items(self):
        """
        Returns a list of all items in the tree using pre-order traversal.
        """
        out_list = []
        self._pre_order_items(self.root, out_list)
        return out_list

    def _pre_order_items(self, subtree_root, out_list):
        """
        Pre-order traversal, adding values from visited nodes to out_list. 
        """
        if subtree_root == None:
            return
        out_list.append(subtree_root.value)
        left = self._pre_order_items(subtree_root.left, out_list)            
        right = self._pre_order_items(subtree_root.right, out_list)

    def post_order_items(self):
        """
        Returns a list of all items in the tree using post-order traversal.
        """
        out_list = []
        self._post_order_items(self.root, out_list)
        return out_list

    def _post_order_items(self, subtree_root, out_list):
        """
        Post-order traversal, adding values from visited nodes to out_list.
        """
        if subtree_root == None:
            return
        left = self._post_order_items(subtree_root.left, out_list)            
        right = self._post_order_items(subtree_root.right, out_list)           
        out_list.append(subtree_root.value)           

    def __contains__(self, value):
        """
        Returns True if the tree contains an item, False otherwise. e.g. "x in tree" returns T/F
        """
        return self._contains(self.root, value)

    def _contains(self, subtree_root, value):       
        if subtree_root is None:# Base case -- reached the end of the subtree_root
            return False       
        elif value == subtree_root.value: # Found the item
            return True      
        elif value < subtree_root.value:# The item is to the left
            return self._contains(subtree_root.left, value)       
        else:# The item is to the right
            return self._contains(subtree_root.right, value)

    def __len__(self):
        """
        Returns the number of unique nodes in the tree.
        """
        return self._len(self.root)

    def _len(self, subtree_root):
        if subtree_root is None:
            return 0
        return 1 + self._len(subtree_root.left) + self._len(subtree_root.right)

    def count_duplicates(self):
        """
        Returns the number of elements in the tree (including duplicate values).
        """
        return self._count_duplicates(self.root)
    
    def _count_duplicates(self, node):
        if node == None:
            return 0
        left = self._count_duplicates(node.left)
        right = self._count_duplicates(node.right)
        return node.count + left + right
    
    def remove(self, value):
        """
        Removes the first occurrence of value from the tree.
        """
        self.root = self._remove(self.root, value)

    def _remove(self, subtree_root, value):       
        if subtree_root == None:# value is not in the tree
            return subtree_root        
        elif value < subtree_root.value:# The item should be on the left
            subtree_root.left = self._remove(subtree_root.left, value)        
        elif value > subtree_root.value:# The item should be on the right
            subtree_root.right = self._remove(subtree_root.right, value)       
        else:# The item to be deleted IS subtree_root
            if subtree_root.left == None and subtree_root.right == None:# No children.               
                subtree_root = None
            elif subtree_root.left and subtree_root.right == None:# One left child.               
                subtree_root = subtree_root.left
            elif subtree_root.left == None and subtree_root.right:# One right child.                
                subtree_root = subtree_root.right
            else:# Two children.               
                # subtree_root will be unchanged in this case
                # its value will be changed to the value of the in order successor
                subtree_root.value = self._pop_in_order_successor(subtree_root)
        return subtree_root

    def _pop_in_order_successor(self, subtree_root):
        """
        Returns the value of the in-order successor and removes it from the tree.
        The in order successor will be the smallest value in the right subtree.
        Note: this function will be called when the node to remove has two children
        If the right child has no left child this is easy otherwise it needs
        to use the _recursive_pop_min funciton ...
        """
        if subtree_root.right.left is None:
            successor_value = subtree_root.right.value
            subtree_root.right = subtree_root.right.right
        else:
            successor_value = self._pop_min_recursive(subtree_root.right)
        return successor_value

    def _pop_min_recursive(self, subtree_root):
        """ Recursive code.
         Returns the in min value and removes the node from the subtree
         If the left child of subtree has no left child,
         then the left child contains the min value,
         so de-link  the left child from the subtree and return its value.
         Remember to keep the left child's right child connected to the subtree.
        """
        if subtree_root.left.left == None:
            value = subtree_root.left.value                
            subtree_root.left = subtree_root.left.right              
            return value
      
        return self._pop_min_recursive(subtree_root.left)# traverse left

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.root)

def main():  
    unsorted_list = [random.randint(-2**8,2**8) for i in range(10000)]
    
    tree = BinarySearchTree()
    for n in unsorted_list:
        tree.insert(n)
        
    tree_node_count = len(tree)
    tree_array_len = tree.count_duplicates()
    
    print("tree_node_count:", tree_node_count==len(set(unsorted_list)))
    print("tree_array_len:", tree_array_len==len(unsorted_list))

    sorted_list = sorted(unsorted_list)
    tree_sorted_list = tree.in_order_items_with_duplicates()
    print("sorted_list:", sorted_list == tree_sorted_list)
    
    sorted_set = sorted(set(unsorted_list))
    tree_sorted_set = tree.in_order_items()
    print("sorted_set:", sorted_set == tree_sorted_set)

    for n in unsorted_list:
        if n not in tree:
            print("tree error")

    for n in unsorted_list:
        tree.remove(n)

    for n in unsorted_list:
        if n in tree:
            print("tree error")

    print("remove:", len(tree) == 0 and tree.count_duplicates() == 0)
    
if __name__ == '__main__':
    import random
    main()
    




















    


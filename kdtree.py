"""
Grid(TODO), QuadTree, and KdTree range searching data structure implementations for 2D points
"""
MAX_DEPTH = 10# node max depth, at max depth the leaf node will append all nodes to a list

import matplotlib.pyplot as plt
from collections import namedtuple

Node = namedtuple("Node", ["value", "left", "right"])

# with slicing
def binary_search_tree(nums, is_sorted=False):
    """Return a balanced binary search tree with the given nums
       at the leaves. is_sorted is True if nums is already sorted.
       Inefficient because of slicing but more readable.
    """
    if not is_sorted:
        nums = sorted(nums)
        
    n = len(nums)
    if n == 1:
        tree = Node(nums[0], None, None)  # A leaf
    else:
        mid = n // 2  # Halfway (approx)
        left = binary_search_tree(nums[:mid], True)
        right = binary_search_tree(nums[mid:], True)
        tree = Node(nums[mid - 1], left, right)
    return tree

# without slicing
def binary_search_tree(nums, is_sorted=False, l=None, r=None):
    """Return a balanced binary search tree with the given nums
       at the leaves. is_sorted is True if nums is already sorted.
    """
    if l == None:
        l = 0
        r = len(nums)-1
    if not is_sorted:
        nums = sorted(nums)
    if l >= r:
        tree = Node(nums[l], None, None)  # A leaf
    else:
        mid = (l+r) // 2
        left = binary_search_tree(nums, True, l, mid)
        right = binary_search_tree(nums, True, mid+1, r)
        tree = Node(nums[mid], left, right)
    return tree
    
def print_tree(tree, level=0):
    """Print the tree with indentation"""
    if tree.left is None and tree.right is None: # Leaf?
        print(2 * level * ' ' + f"Leaf({tree.value})")
    else:
        print(2 * level * ' ' + f"Node({tree.value})")
        print_tree(tree.left, level + 1)
        print_tree(tree.right, level + 1)

class Vec:
    """A simple vector in 2D. Can also be used as a position vector from
       origin to define points.
    """
    point_num = 0
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.label = 'P' + str(Vec.point_num)
        Vec.point_num += 1

    def __add__(self, other):
        """Vector addition"""
        return Vec(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        """Vector subtraction"""
        return Vec(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scale):
        """Multiplication by a scalar"""
        return Vec(self.x * scale, self.y * scale)

    def dot(self, other):
        """Dot product"""
        return self.x * other.x + self.y * other.y

    def lensq(self):
        """The square of the length"""
        return self.dot(self)

    def in_box(self, bottom_left, top_right):
        """True if this point (self) lies within or on the
           boundary of the given rectangular box area"""
        return bottom_left.x <= self.x <= top_right.x and bottom_left.y <= self.y <= top_right.y

    def rectangles_overlap(rect1, rect2):
        """given two rectangles (bottom_left_Vec, top_right_Vec) return True if they overlap"""
        p1, p2 = rect1
        p3, p4 = rect2         
        if (p1.x <= p4.x and p2.x >= p3.x and p1.y <= p4.y and p2.y >= p3.y):
            return True
        return False

    def __getitem__(self, axis):
        return self.x if axis == 0 else self.y

    def __repr__(self):
        return "Vec({}, {})".format(self.x, self.y)

    def __lt__(self, other):
        """Less than operator, for sorting"""
        return (self.x, self.y) < (other.x, other.y)

    def __eq__(self, other):
        if isinstance(other, Vec):
            return self.x == other.x and self.y == other.y 
        else:
            return False

# with slicing   
class KdTree:
    """A 2D k-d tree"""
    LABEL_POINTS = True
    LABEL_OFFSET_X = 0.25
    LABEL_OFFSET_Y = 0.25    
    def __init__(self, points, depth=0, max_depth=MAX_DEPTH):
        """Initialiser, given a list of points, each of type Vec, the current
           depth within the tree (0 for root), used during recursion, and the
           maximum depth allowable for a leaf node.
        """
        if len(points) < 2 or depth >= max_depth: # Ensure at least one point per leaf
            self.is_leaf = True
            self.points = points
        else:
            self.is_leaf = False
            self.axis = depth % 2  # 0 for vertical divider (x-value), 1 for horizontal (y-value)
            points.sort(key=lambda p: p[self.axis])
            halfway = len(points) // 2
            self.coord = points[halfway - 1][self.axis]
            self.leftorbottom = KdTree(points[:halfway], depth + 1, max_depth)
            self.rightortop = KdTree(points[halfway:], depth + 1, max_depth)
            
    def points_in_range(self, rectangle):
        """Return a list of all points in the tree 'self' that lie within or on the 
           boundary of the given query rectangle, which is defined by a pair of points
           (bottom_left, top_right).
        """

        bl_point, tr_point = rectangle[0], rectangle[1]
        matches = []
        if self.is_leaf:
            #matches = all points in node.points lying within rectangle
            for point in self.points:
                if point.in_box(bl_point, tr_point):
                    matches.append(point)
        else:
            matches = []
            if self.leftorbottom != None:
                matches += self.leftorbottom.points_in_range(rectangle)
            if self.rightortop != None:
                matches += self.rightortop.points_in_range(rectangle)
        return matches
    
    
    def plot(self, axes, top, right, bottom, left, depth=0):
        """Plot the the kd tree. axes is the matplotlib axes object on
           which to plot; top, right, bottom, left are the x or y coordinates
           the bounding box of the plot.
        """

        if self.is_leaf:
            axes.plot([p.x for p in self.points], [p.y for p in self.points], 'bo')
            if self.LABEL_POINTS:
                for p in self.points:
                    axes.annotate(p.label, (p.x, p.y),
                    xytext=(p.x + self.LABEL_OFFSET_X, p.y + self.LABEL_OFFSET_Y))
        else:
            if self.axis == 0:
                axes.plot([self.coord, self.coord], [bottom, top], '-', color='gray')
                self.leftorbottom.plot(axes, top, self.coord, bottom, left, depth + 1)
                self.rightortop.plot(axes, top, right, bottom, self.coord, depth + 1)
            else:
                axes.plot([left, right], [self.coord, self.coord], '-', color='gray')
                self.leftorbottom.plot(axes, self.coord, right, bottom, left, depth + 1)
                self.rightortop.plot(axes, top, right, self.coord, left, depth+1)
        if depth == 0:
            axes.set_xlim(left, right)
            axes.set_ylim(bottom, top)
       
    
    def __repr__(self, depth=0):
        """String representation of self"""
        if self.is_leaf:
            return depth * 2 * ' ' + "Leaf({})".format(self.points)
        else:
            s = depth * 2 * ' ' + "Node({}, {}, \n".format(self.axis, self.coord)
            s += self.leftorbottom.__repr__(depth + 1) + '\n'
            s += self.rightortop.__repr__(depth + 1) + '\n'
            s += depth * 2 * ' ' + ')'  # Close the node's opening parens
            return s


class QuadTree:
    """A 2D quadtree"""
    def __init__(self, points, centre, size, depth=0, max_leaf_points=2, max_depth=MAX_DEPTH):
        self.centre = centre                    # centre of the current quadrant
        self.size = size                        # diameter or length of base / column
        self.depth = depth                      # current tree depth (root depth==0)
        self.max_leaf_points = max_leaf_points  # max allowed points on a leaf node below max_depth
        self.max_depth = max_depth              # max QuadTree/node depth allowed
        self.children = []                      # child QuadTrees/nodes
        self.points = []                        # points within this quadtree/node
        self.is_leaf = False                    # all nodes contain a list of points within their quadrant
                                                    
        
        r = self.size / 2
        bottom_left = Vec(self.centre.x-r, self.centre.y-r)
        top_right = Vec(self.centre.x+r, self.centre.y+r)
        
        for point in points:# get points within the current quadrant
            if point.in_box(bottom_left, top_right):
                self.points.append(point)
                
        if len(self.points) > max_leaf_points and self.depth < self.max_depth:# divide points into sub-quadrants
            for i in range(4):# (i=0: bottom left, i=1: top left, i=2: bottom right, i=3: top right)
                if i < 2:       
                    x = centre.x - size / 4# left
                else:           
                    x = centre.x + size / 4# right
                if i % 2 == 0:  
                    y = centre.y - size / 4# bottom
                else:           
                    y = centre.y + size / 4# top
                child_centre = Vec(x, y)
                child_size = self.size / 2
                child = QuadTree(self.points, child_centre, child_size, depth + 1, max_leaf_points)
                self.children.append(child)
        else:# leaf node reached (no more division necessary or maximum depth reached)
            self.is_leaf = True

    def points_in_range(self, rectangle):
        """Return a list of all points in the tree 'self' that lie within or on the 
           boundary of the given query rectangle, which is defined by a pair of points
           (bottom_left, top_right).
        """
        #print("  "*self.depth, "searching:", self.centre)
        bl_point, tr_point = rectangle[0], rectangle[1]
        matches = []# all points in node.points lying within rectangle
        if self.is_leaf:
            for point in self.points:
                if point.in_box(bl_point, tr_point):
                    #print("  "*(self.depth+1),"point:",point,"in:",rectangle)
                    matches.append(point)
        else:
            matches = []
            for i in range(4):# for each child quadrant
                # check if the search rectangle is inside each quadrant
                if i < 2:       
                    x = self.centre.x - self.size / 4# left
                else:           
                    x = self.centre.x + self.size / 4# right
                if i % 2 == 0:  
                    y = self.centre.y - self.size / 4# bottom
                else:           
                    y = self.centre.y + self.size / 4# top
                child_centre = Vec(x, y)
                child_size = self.size / 2
                quadrant = (Vec(child_centre.x-child_size, child_centre.y-child_size), Vec(child_centre.x+child_size, child_centre.y+child_size))
                if Vec.rectangles_overlap(rectangle, quadrant):
                    matches += self.children[i].points_in_range(rectangle)
                

        return matches

    def plot(self, axes):
        """Plot the dividing axes of this node and
           (recursively) all children"""
        if self.is_leaf:
            axes.plot([p.x for p in self.points], [p.y for p in self.points], 'bo')
        else:
            axes.plot([self.centre.x - self.size / 2, self.centre.x + self.size / 2],
                      [self.centre.y, self.centre.y], '-', color='gray')
            axes.plot([self.centre.x, self.centre.x],
                      [self.centre.y - self.size / 2, self.centre.y + self.size / 2],
                      '-', color='gray')
            for child in self.children:
                child.plot(axes)
        axes.set_aspect(1)
                
    def __repr__(self, depth=0):
        """String representation with children indented"""
        indent = 2 * self.depth * ' '
        if self.is_leaf:
            return indent + "Leaf({}, {}, {})".format(self.centre, self.size, self.points)
        else:
            s = indent + "Node({}, {}, [\n".format(self.centre, self.size)
            for child in self.children:
                s += child.__repr__(depth + 1) + ',\n'
            s += indent + '])'
            return s

class Grid:
    """
    TODO
    """
    def __init__(self, points):
        return
        
def main():
    import random
    
    # kdTree tests
    print("KdTree tests")
    tests = []
    point_tuples = [(1, 3), (10, 20), (5, 19), (0, 11), (15, 22), (30, 5)]
    points = [Vec(*tup) for tup in point_tuples]
    tree = KdTree(points)
    
    bottom_left, top_right = Vec(0,0), Vec(40,40)
    rectangle = bottom_left, top_right
    test = sorted(tree.points_in_range(rectangle)) == sorted(points)# all points in range
    tests.append(test)
    #print("points_in_range", rectangle, test, tree.points_in_range(rectangle))

    rectangle = Vec(0,0), Vec(5,5)
    test = sorted(tree.points_in_range(rectangle)) == sorted([Vec(1,3)])
    tests.append(test)
    #print("points_in_range", rectangle, test, tree.points_in_range(rectangle))

    rectangle = Vec(5,15), Vec(15,21)
    test = sorted(tree.points_in_range(rectangle)) == sorted([Vec(5,19), Vec(10,20)])
    tests.append(test)
    #print("points_in_range", rectangle, test, tree.points_in_range(rectangle))

    rectangle = Vec(15,22), Vec(25,25)
    test = sorted(tree.points_in_range(rectangle)) == sorted([Vec(15, 22)])
    tests.append(test)
    #print("points_in_range", rectangle, test, tree.points_in_range(rectangle))

    rectangle = Vec(-1,-1), Vec(-999,-999)
    test = sorted(tree.points_in_range(rectangle)) == sorted([])# no points in range
    tests.append(test)
    #print("points_in_range", rectangle, test, tree.points_in_range(rectangle))

    # random tests
    point_tuples = [(random.randint(-2**8,2**8), random.randint(-2**8,2**8)) for x in range(2**8)]
    points = [Vec(*tup) for tup in point_tuples]
    tree = KdTree(points)
    for i in range(2**8):
        rectangle = Vec(random.randint(-2**8,0),random.randint(-2**8,0)), Vec(random.randint(0,2**8), random.randint(0,2**8))
        test = sorted(tree.points_in_range(rectangle)) == sorted([x for x in points if x.in_box((*rectangle))])
        tests.append(test)
        if test == False:
            print(i)
            print(rectangle)
            print(sorted(tree.points_in_range(rectangle)))
            print(sorted( [ x for x in points if points[i].in_box((*rectangle))]))
            return

    print(" KdTree:", all(tests))    
    #axes = plt.axes()
    #tree.plot(axes, 25, 35, 0, 0)
    #plt.show()
    

    # Vec.rectangles_overlap tests
    print("\n\nVec.rectangles_overlap tests")
    tests = []
    rect1 = Vec(1,1), Vec(2,2)
    rect2 = Vec(1,1), Vec(2,2)
    tests.append(Vec.rectangles_overlap(rect1,rect2) == True)

    rect2 = Vec(-1,-1), Vec(-2,-2)
    tests.append(Vec.rectangles_overlap(rect1,rect2) == False)

    rect2 = Vec(-1,-1), Vec(2,2)
    tests.append(Vec.rectangles_overlap(rect1,rect2) == True)

    rect2 = Vec(-2,-2), Vec(1,1)
    tests.append(Vec.rectangles_overlap(rect1,rect2) == True)

    rect2 = Vec(-2,-2), Vec(0.99,0.99)
    tests.append(Vec.rectangles_overlap(rect1,rect2) == False)
    
    rect2 = Vec(0,0), Vec(0,0)
    tests.append(Vec.rectangles_overlap(rect1,rect2) == False)
    print(" Vec.rectangles_overlap:", all(tests), tests)

    
    # QuadTree tests
    print("\n\nQuadTree tests")
    tests = []  
    points = [(60, 15), (15, 60), (30, 58), (42, 66), (40, 70)]
    vecs = [Vec(*p) for p in points]
    tree = QuadTree(vecs, centre=Vec(50, 50), size=100)# (points, centre, size, depth=0, max_leaf_points=2, max_depth=MAX_DEPTH):

    rectangle = Vec(-1,-1), Vec(-999,-999)
    points_in_range = tree.points_in_range(rectangle)
    test = sorted(points_in_range) == []# no points in range
    tests.append(test)
    #print("points_in_range", rectangle, test, points_in_range, "\n")

    rectangle = Vec(0,0), Vec(999,999)
    points_in_range = tree.points_in_range(rectangle)
    test = sorted(points_in_range) == sorted(vecs)# all points in range
    tests.append(test)
    #print("points_in_range", rectangle, test, points_in_range, "\n")

    rectangle = Vec(59,14), Vec(61,16)
    points_in_range = tree.points_in_range(rectangle)
    test = sorted(points_in_range) == sorted([Vec(*p) for p in [(60, 15)]])# single point in range (bottom right quadrant)
    tests.append(test)
    #print("points_in_range", rectangle, test, points_in_range, "\n")

    rectangle = Vec(14,59), Vec(16,61)
    points_in_range = tree.points_in_range(rectangle)
    test = sorted(points_in_range) == sorted([Vec(*p) for p in [(15, 60)]])# single point in range (top left quadrant)
    tests.append(test)
    #print("points_in_range", rectangle, test, points_in_range, "\n")

    rectangle = Vec(0,0), Vec(40,999)
    points_in_range = tree.points_in_range(rectangle)
    test = sorted(points_in_range) == sorted([Vec(*p) for p in [(15, 60), (30, 58), (40, 70)]])
    tests.append(test)
    #print("points_in_range", rectangle, test, points_in_range, "\n")

    
    rectangle = Vec(0,20), Vec(50,60)
    points_in_range = tree.points_in_range(rectangle)
    test = sorted(points_in_range) == sorted([Vec(*p) for p in [(15, 60), (30, 58)]])
    tests.append(test)
    #print("points_in_range", rectangle, test, points_in_range, "\n")

    print(all(tests))
    # random tests
    point_tuples = [(random.randint(-2**8,2**8), random.randint(-2**8,2**8)) for x in range(2**8)]
    points = [Vec(*tup) for tup in point_tuples]
    tree = QuadTree(points, centre=Vec(0, 0), size=2**8)
    for i in range(2**8):
        rectangle = Vec(random.randint(-2**8,0),random.randint(-2**8,0)), Vec(random.randint(0,2**8), random.randint(0,2**8))
        test = sorted(tree.points_in_range(rectangle)) == sorted([x for x in points if x.in_box((*rectangle))])
        tests.append(test)
        if test == False:
            print(i,'\n')
            print(rectangle,'\n')
            print(sorted(tree.points_in_range(rectangle)),'\n')
            print()
            print(sorted( [ x for x in points if points[i].in_box((*rectangle))]),'\n')
            print("\n", points)
            

    print(" QuadTree:", all(tests))    
    axes = plt.axes()
    tree.plot(axes)
    axes.set_xlim(-2**8, 2**8)
    axes.set_ylim(-2**8, 2**8)
    plt.show()
    


    # binary_search_tree tests
    print("\n\nbinary_search_tree tests")
    tests = []
    def bst_to_array(node):
        if node == None:
            return []
        if node.left == node.right == None:# leaf
            return [node.value]
        return bst_to_array(node.left) + bst_to_array(node.right)
    
    nums = [22, 41, 19, 27, 12, 35, 14, 20,  39, 10, 25, 44, 32, 21, 18]
    tree = binary_search_tree(nums)
    tests.append(bst_to_array(tree) == sorted(nums))
    
    nums = [15, 3, 11, 21, 7, 0, 19, 33, 29, 4]
    tree = binary_search_tree(nums)
    tests.append(bst_to_array(tree) == sorted(nums))
    
    nums = [228]
    tree = binary_search_tree(nums)
    tests.append(bst_to_array(tree) == sorted(nums))
    
    nums = [random.randint(-2**8,2**8) for x in range(2**10)]
    tree = binary_search_tree(nums)
    tests.append(bst_to_array(tree) == sorted(nums))

    # bst random tests
    for i in range(2**8):
        nums = [random.randint(-2**8,2**8) for x in range(2**8)]
        tree = binary_search_tree(nums)
        test = bst_to_array(tree) == sorted(nums)
        tests.append(test)
        if not test:          
            print('bst error',nums,tree)
            return
        

    print(" binary_search_tree:", all(tests))


if __name__ == '__main__':
    main()

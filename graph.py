from queue import deque
from heapq import *

example_graph_string = '''\
U 3 W
0 1 7
1 2 -2
0 2 0
'''

"""
Top row:
    Directed/Undirected number_of_vertices weight
following rows:
    source_vertex destination_vertex weight/cost
"""

example_parent_array = [None, 0, 0]
"""
after a bfs traversal of example_graph_string starting from vertex 0
    the parents of vertex 1 and 2 are vertex 0
"""

"""
Aspect                                      Adj matrix   Adj list                  Comment
Determine whether there is an edge
    (i, j) in the graph:                        Θ(1)       O(n)                     matrix[i][j]!= None vs (j,cost) in adj_list[i]
Adding an edge:                                 Θ(1)       O(n)                     matrix[i][j] = cost vs adj_list[i]+=(j,cost) if not in adj_list[i]
Deleting an edge:                               Θ(1)       O(n)                     matrix[i][j] = None vs adj_list[i][?] = None
Iterating over edges from a given vertex:       Θ(n)       O(n)                     matrix[i] vs adj_list[i] note for adj_list could be 0<->n edges               
Iterating over all edges:                       Θ(n2)      Θ(n + m)=Θ(max(n, m))    n*n vs n*(0<->n)
Space:                                          Θ(n2)      Θ(n + m)=Θ(max(n, m))    n*n vs n*(0<->n)

where n = number of vertices, m = total number of edges
"""
def adjacency_matrix(graph_str):
    """
    converts a graph_str to a list of costs to all other vertices in the graph (None if no edge exists)
        (e.g. adj_matrix[0][1] == cost to travel from vertex 0 to 1, cost==None if route impossible)   
    """
    rows = (graph_str.rstrip()).split("\n")
    top_row_data = rows[0].split()
    directed, n = top_row_data[0] == 'D', int(top_row_data[1])
    weighted = True if len(top_row_data) > 2 and top_row_data[2] == 'W' else False

    output = [[None for j in range(n)] for i in range(n)]
    for row in rows[1:]:
        row_data = row.split()
        src, dst = int(row_data[0]), int(row_data[1])
        cost = int(row_data[2]) if weighted else 1
        output[src][dst] = cost
        if not directed:
            output[dst][src] = cost
            
    return output

def adjacency_list(graph_str):
    """
    converts a graph_str to a list of edges for each vertex
        (e.g. adj_list[0] == list of all edges from vertex 0)
    """
    rows = (graph_str.rstrip()).split("\n")
    top_row_data = rows[0].split()
    directed, n = top_row_data[0] == 'D', int(top_row_data[1])
    weighted = True if len(top_row_data) > 2 and top_row_data[2] == 'W' else False

    output = [[] for i in range(n)]
    for row in rows[1:]:
        row_data = row.split()
        src, dst = int(row_data[0]), int(row_data[1])
        cost = int(row_data[2]) if weighted else None
        output[src].append((dst, cost))
        if not directed:
            output[dst].append((src, cost))
    
    return output

def dfs_tree(adj_list, start, state=None, parent=None, stack=None):
    """
    Depth First Search Traversal of a graph, returns state, parent, stack
        State:
            'P' for Processed i.e. Vertices reachable from the start
            'U' for Undiscovered i.e. Vertices un-reachable from the start
        Parent:
            An array where index i represents Vertex i and parent[i] is its direct parent Vertex
        Stack:
            Traversal from the first deepest Vertex to the next deepest, to the start
    """      
    n = len(adj_list)# Number of Vertices
    state = state if state else ['U' for x in range(n)]# All Vertices Undiscovered
    parent = parent if parent else [None for x in range(n)]# Parent array indicating parent[i]=p, i is the child vertex and p parent
    stack = stack if stack else []
    state[start] = 'D'# Starting Node set to Discovered
    return dfs_loop(adj_list, start, state, parent, stack)

def dfs_loop(adj_list, u, state, parent, stack, cycle_detection=False, cycle_type="D"):
    for v, w in adj_list[u]:# For each Vertex (v) connected to Vertex u (previous/starting Vertex)
        if state[v] == 'U':# If v has not yet been Discovered             
            state[v] = 'D'# Set v Discovered
            parent[v] = u# Set v's parent as u
            dfs_loop(adj_list, v, state, parent, stack, cycle_detection, cycle_type)# Recursively traverse deeper to the children of v
        elif cycle_detection and state[v] == 'D':# cycle detected
            if cycle_type == "D":
                stack.append("cycle")
            elif cycle_type == "U" and parent[u] != v:# for U graphs ignore when v is parent of u
                stack.append("cycle")
    stack.append(u)# Stack holds the traversal from the first deepest node to the next until reaching start
    state[u] = 'P'# Set u to Processed
    return state, parent, stack

def bfs_tree(adj_list, start):
    """
    Breadth First Search Traversal of a graph, returns a parent array
    """
    n = len(adj_list)
    state = ['U' for x in range(n)]
    parent = [None for x in range(n)]
    Q = deque()
    state[start] = 'D'
    Q.append(start)
    return bfs_loop(adj_list, Q, state, parent)

def bfs_loop(adj_list, Q, state, parent):
    while len(Q) > 0:
        u = Q.popleft()
        for v, w in adj_list[u]:
            if state[v] == 'U':
                state[v] = 'D'
                parent[v] = u
                Q.append(v)
        state[u] = 'P'
    return parent

def shortest_path(adj_list, s, t):
    """
    Generates a bfs parent array on the starting vertex
        parent array can be traversed to find the shortest path to any other reachable vertex
        if there is no path between s and t, [] is returned
    """
    parent = bfs_tree(adj_list, s)
    path = _tree_path(parent, s, t) 
    return path if path[0] != None else []

def _tree_path(parent, s, t):
    """
    Returns the vertex order from s to t of a given parent array/tree
    """
    if t == None:# No path between s and t
        return [None]
    elif s == t:
        return [s]
    return _tree_path(parent, s, parent[t]) + [t]

def connected_components(adj_list):
    """
    Components of an undirected graph are maximal sub-graphs in which all vertices are reachable from each other
    Given an undirected graph, returns a set of all maximal sub-graphs
    """
    n = len(adj_list)
    state = ['U' for x in range(n)]
    Q = deque()
    components = set()
    for i in range(n):
        if state[i] == 'U':
            prev_state = [x for x in state]
            state[i] = 'D'
            Q.append(i)
            parent = [None for x in range(n)]
            bfs_loop(adj_list, Q, state, parent)
            component = []
            for j in range(n):# see what vertices have changes
                if prev_state[j] != state[j]:
                    component.append(j)
            components.add(tuple(component))
    return components

def graph_transposed(adj_list):
    """
    Reverses all edges on the graph, note an undirected graph == its transpose
    """
    output = [[] for vertex in adj_list]
    for src, edges in enumerate(adj_list):
        for edge in edges:
            dst, cost = edge[0], edge[1]
            output[dst] += [(src, cost)]    
    return output

def strongly_connected(adj_list):
    """
    A directed graph is strongly connected if and only if there is a path between
    every ordered pair of vertices (i.e. the graph has no maximal sub-graphs / connected components)

    A simple way to test this is to run a bfs/dfs traversal on each vertex and confirm each traversal
    was able to Process / reach all other vertices

    This method compares a graph G to its tranpose G^T, where the transpose has all edges reversed
    """
    s = 0
    # 1: Run a graph traversal (BFS or DFS) on G from a starting point s
    state, parent, stack = dfs_tree(adj_list, s, [])
    # 2: If any vertex remains undiscovered, return False
    if 'U' in state:
        return False
    # 3: Construct the transpose
    transpose = graph_transposed(adj_list)
    # 4: Run a graph traversal (BFS or DFS) on G^T from a starting point s
    state, parent, stack = dfs_tree(adj_list, s, [])
    # 5: If any vertex remains undiscovered, return False, Else True
    if 'U' in state:
        return False
    return True

def topological_sorting(adj_list):
    """
    Returns a stack containing a topological sorting in reverse order
    
    A topological ordering of a directed graph is an ordering of its vertices, such
    that, if there is an edge (u, v) in the graph, then vertex u is placed in some
    position before vertex v. Any Directed Acyclic Graph (DAG) has at least one
    topological ordering.

    The following includes some examples of topological sorting on DAGs.
    
        • In software engineering, build automation systems (e.g. Make and Apache
        Maven) apply topological sort to the DAG of dependencies between soft-
        ware components to find a valid order of building software.
        
        • Package management systems (e.g. dpkg, RPM, and Homebrew) apply
        topological sort to the DAG of package dependencies to find a valid
        order of installation.
        
        • A prerequisite graph of courses at a university is a DAG. A topological
        ordering allows doing courses in some valid order.
        
        • Copying relational databases must be done according to a topological
        ordering of tables (where dependencies are based on foreign keys) so that
        the referential integrity of the data is maintained during the operation.
    """
    n = len(adj_list)
    state = ['U' for x in range(n)]
    parent = [None for x in range(n)]
    stack = []
    for i in range(n):
        if state[i] == 'U':# dfs search on un-discovered vertices
            dfs_loop(adj_list, i, state, parent, stack)

    return stack# returns the stack representing a topological sorting in reverse order

def cycle_detection(adj_list, graph_type="D"):
    """
    Sometimes we want to know if a graph has a cycle. This is, for example, useful
    to determine whether a directed graph is a DAG, and thus, has a topological
    ordering.

    During DFS traversal, when examining the outgoing edges of a vertex u, if
    the edge (u, v) goes to a vertex v that is already discovered (that is, it is on
    the call stack), then the graph has a cycle.

    For undirected graphs, the process is similar, but there is one exception: when
    examining the outgoing edges of a vertex u, we ignore the edge (u, v) where v
    is the parent of u (v is guaranteed to be discovered but that doesn’t count as
    a cycle because a cycle has to be a path, and a path is not allowed to use an
    edge more than once).

    For undirected graphs with multiple components, or for directed graphs, mul-
    tiple DFS calls may need to be made in order to find a cycle. Therefore,
    similar to finding the components of a graph or topological ordering, a for
    loop must check the state of each vertex in the graph, and run a DFS from
    that vertex if it is undiscovered.
    """
    n = len(adj_list)
    state = ['U' for x in range(n)]
    parent = [None for x in range(n)]
    stack = []
    for i in range(n):
        if state[i] == 'U':
            state, parent, stack = dfs_loop(adj_list, i, state, parent, state, True, graph_type)
            if "cycle" in stack:
                return True
    return False

def next_vertex(in_tree, distance):
    """
    Function used by the prim mst and dijkstra algorithms.
    Given a distance array, finds the next un-discovered vertex with the lowest cost.
    """
    n = None
    for i, vertex in enumerate(in_tree):
        # vertex not reached yet, and (init or lower cost vertex found)
        if not vertex and (n == None or distance[i] < distance[n]):
            n = i
    return n

def prim(adj_list, s=0):
    """
    A spanning tree of an undirected graph is a subgraph that is a tree and includes
    all the vertices of the graph. A minimum spanning tree (MST) of a weighted
    undirected graph is a spanning tree that has the lowest total weight among all
    other spanning trees.

    Given a weighted undirected graph with one component, Prim’s algorithm
    finds a minimum spanning tree. The algorithm is called with an arbitrary
    vertex as the starting point which forms a one-vertex tree. The tree gradually
    grows in each iteration by adding the smallest edge between a vertex that is
    part of the tree and a vertex that is not.
    """
    n = len(adj_list)
    in_tree = [False for x in range(n)]
    distance = [float('inf') for x in range(n)]
    parent = [None for x in range(n)]
    distance[s] = 0
    while not all(in_tree):
        u = next_vertex(in_tree, distance)
        in_tree[u] = True       
        for v, weight in adj_list[u]:
            if not in_tree[v] and weight < distance[v]:
                distance[v] = weight
                parent[v] = u
    return parent, distance

def dijkstra(adj_list, s=0):
    """
    Given a graph with non-negative edge weights and a starting vertex, the al-
    gorithm finds the shortest path from the starting vertex to any other vertex
    reachable from it.
    
    The algorithm gradually grows a shortest path tree rooted at the starting
    vertex. In each iteration, a new edge is added to the tree by selecting an edge
    that connects a vertex in the tree to a vertex outside that is closest to the
    starting vertex.
    """
    n = len(adj_list)
    in_tree = [False for x in range(n)]
    distance = [float('inf') for x in range(n)]
    parent = [None for x in range(n)]
    distance[s] = 0
    while not all(in_tree):
        u = next_vertex(in_tree, distance)
        in_tree[u] = True
        for v, weight in adj_list[u]:
            if not in_tree[v] and (distance[u] + weight) < distance[v]:
                distance[v] = distance[u] + weight
                parent[v] = u
    return parent, distance

def weighted_mst_path(adj_list, s, t):
    """
    Generates a minimum spanning tree/parent, distance array on the starting vertex
        parent, distance arrays are traversed to find the shortest path to any other reachable vertex
        if there is no path between s and t, [] is returned

        path[2] = (vertex, cost_from_prev_vertex)
    """
    parent, distance = prim(adj_list, s)
    path = _weighted_tree_path(parent, distance, s, t)
    return path if path[0][0] != None else []

def weighted_shortest_path(adj_list, s, t):
    """
    Generates a Dijkstra parent, distance array on the starting vertex
        parent, distance arrays are traversed to find the shortest path to any other reachable vertex
        if there is no path between s and t, [] is returned

        path[2] = (vertex, cost_from_starting_vertex)
    """
    parent, distance = dijkstra(adj_list, s)
    path = _weighted_tree_path(parent, distance, s, t)
    return path if path[0][0] != None else []

def _weighted_tree_path(parent, distance, s, t):
    """
    Returns the (vertex, total_cost) order from s to t of a given parent array/tree
    """
    if t == None:# No path between s and t
        return [(None, None)]
    elif s == t:
        return [(s, 0)]
    return _weighted_tree_path(parent, distance, s, parent[t]) + [(t, distance[t])]



"""
implicit graphs specify outgoing_arcs or neighbouring_arcs from a particular node (e.g. a position x,y)
allowing for endless graphs and graphs that arn't fully contained within memory (adj_list / adj_matrix)
"""
from collections import namedtuple

def generic_search(graph, frontier):
    for starting_node in graph.starting_nodes():
        # Paths are tuples and the first arc on each path is a dummy arc to a starting node
        frontier.add((Arc(None, starting_node, "no action", 0),)) 
    
    for path in frontier:
        node_to_expand = path[-1].head # head of the last arc in the path

        if graph.is_goal(node_to_expand):
            yield path# continously supplies paths when called (e.g. generic_searc(g,f).next())

        for arc in graph.outgoing_arcs(node_to_expand):
            frontier.add(path + (arc,)) # add back a new extended path


class Arc(namedtuple('Arc', 'tail, head, action, cost')):
    """Represents an arc in a graph.

    Keyword arguments:
    tail -- the source node (state)
    head -- the target node (state)
    action -- a string describing the action that must be taken in
             order to get from the source state to the destination state.
    cost -- a number that specifies the cost of the action"""


class Graph():
    """This is an abstract class for graphs. It cannot be directly
    instantiated. You should define a subclass of this class
    (representing a particular problem) and implement the expected
    methods."""

    def is_goal(self, node):
        """Returns true if the given node is a goal state, false otherwise."""

    def starting_nodes(self):
        """Returns a sequence of starting nodes. Often there is only one
        starting node but even then the function returns a sequence
        with one element. It can be implemented as an iterator if
        needed."""

    def outgoing_arcs(self, tail_node):
        """Given a node it returns a sequence of arcs (Arc objects)
        which correspond to the actions that can be taken in that
        state (node)."""

    def estimated_cost_to_goal(self, node):
        """Return the estimated cost to a goal node from the given
        state. This function is usually implemented when there is a
        single goal state. The function is used as a heuristic in
        search. The implementation should make sure that the heuristic
        meets the required criteria for heuristics."""
        raise NotImplementedError


class ExplicitGraph(Graph):
    """This is a concrete subclass of Graph where vertices and edges
     are explicitly enumerated. Objects of this type are useful for
     testing graph algorithms."""

    def __init__(self, nodes, edge_list, starting_nodes, goal_nodes, estimates=None):
        """Initialises an explicit graph.
        Keyword arguments:
        nodes -- a set of nodes
        edge_list -- a sequence of tuples in the form (tail, head) or 
                     (tail, head, cost)
        starting_nodes -- the list of starting nodes. We use a list
                          to remind you that the order can influence
                          the search behaviour.
        goal_node -- the set of goal nodes. It's better if you use a set
                     here to remind yourself that the order does not matter
                     here. This is used only by the is_goal method."""

        # A few assertions to detect possible errors in
        # instantiation. These assertions are not essential to the
        # class functionality.
        assert all(tail in nodes and head in nodes for tail, head, *_ in edge_list)\
           , "An edge must link two existing nodes!"
        assert all(node in nodes for node in starting_nodes),\
            "The starting_states must be in nodes."
        assert all(node in nodes for node in goal_nodes),\
            "The goal states must be in nodes."

        self.nodes = nodes      
        self.edge_list = edge_list
        self._starting_nodes = starting_nodes
        self.goal_nodes = goal_nodes
        self.estimates = estimates

    def starting_nodes(self):
        """Returns a sequence of starting nodes."""
        return self._starting_nodes

    def is_goal(self, node):
        """Returns true if the given node is a goal node."""
        return node in self.goal_nodes

    def outgoing_arcs(self, node):
        """Returns a sequence of Arc objects that go out from the given
        node. The action string is automatically generated."""
        arcs = []
        for edge in self.edge_list:
            if len(edge) == 2:  # if no cost is specified
                tail, head = edge
                cost = 1        # assume unit cost
            else:
                tail, head, cost = edge
            if tail == node:
                arcs.append(Arc(tail, head, str(tail) + '->' + str(head), cost))
        return arcs

class Frontier():
    """This is an abstract class for frontier classes. It outlines the
    methods that must be implemented by a concrete subclass. Concrete
    subclasses determine the search strategy."""

    def add(self, path):
        """Adds a new path to the frontier. A path is a sequence (tuple) of
        Arc objects. You should override this method.""" 
    def __iter__(self):
        """We don't need a separate iterator object. Just return self. You
        don't need to change this method."""
        return self
    def __next__(self):
        """
        Selects, removes, and returns a path on the frontier if there is
        any.Recall that a path is a sequence (tuple) of Arc
        objects. Override this method to achieve a desired search
        strategy. If there nothing to return this should raise a
        StopIteration exception.
        """

class DFSFrontier(Frontier):
    """Implements a frontier container appropriate for depth-first-search."""
    def __init__(self):
        """The constructor takes no argument. It initialises the
        container to an empty stack."""
        self.container = []
        self.expanded = set()# keep track of nodes that have been expanded (i.e. its neighbouring nodes have been added to the frontier to be expanded)
    def add(self, path):# path == Arc(tail='x', head='y', action='x->y', cost=1)
        if path[-1].head not in self.expanded:# do not add paths to nodes which we have already expanded from
            self.container.append(path)
    def __iter__(self):
        """The object returns itself because it is implementing a __next__
        method and does not need any additional state for iteration."""
        return self      
    def __next__(self):
        if len(self.container) > 0:
            next_path = self.container.pop()
            self.expanded.add(next_path[-1].head)# next_path has been returned to be expanded upon by generic_search calling graph.outgoing_arcs()
            return next_path
        else:
            raise StopIteration
class BFSFrontier(Frontier):
    def __init__(self):
        self.container = deque()
        self.expanded = set()
    def add(self, path):# path == Arc(tail='x', head='y', action='x->y', cost=1)
        if path[-1].head not in self.expanded:
            self.container.append(path)
    def __iter__(self):
        return self      
    def __next__(self):
        if len(self.container) > 0:
            next_path = self.container.popleft()
            self.expanded.add(next_path[-1].head)
            return next_path
        else:
            raise StopIteration
class LCFSFrontier(Frontier):
    def __init__(self):
        self.container = []# heap array/list
        self.expanded = set()
        self.size = 0# used to measure indices of items pushed into the heap (for stable sorting)
    def add(self, path):
        if path[-1].head not in self.expanded:
            cost = sum([float(arc.cost) for arc in path]) + 0
            #for arc in path:
                #cost += float(arc.cost)
            heappush(self.container, (cost, self.size, path))
            self.size += 1
    def __iter__(self):
        return self        
    def __next__(self):
        if len(self.container) > 0:
            next_path = heappop(self.container)[2]
            self.expanded.add(next_path[-1].head)
            return next_path
        else:
            raise StopIteration
        
def print_actions(path):
    """Given a path (a sequence of Arc objects), prints the actions that
    need to be taken and the total cost of those actions. The path is
    usually a solution (a path from the starting node to a goal
    node."""
    if path:
        print("Actions:")
        print(",\n".join("  {}".format(arc.action) for arc in path[1:]) + ".")
        print("Total cost:", sum(arc.cost for arc in path))
    else:
        print("There is no solution!")




def main():
    tests = []
    graph_string = """\
    D 3
    0 1
    1 0
    0 2
    """  
    tests.append(adjacency_matrix(graph_string) == [[None, 1, 1],[1,None, None],[None, None, None]])
    tests.append(adjacency_list(graph_string) == [[(1, None), (2, None)], [(0, None)], []])
    tests.append(dfs_tree(adjacency_list(graph_string), 0, []) == (['P','P','P'],[None, 0, 0],[1,2,0]) )
    tests.append(bfs_tree(adjacency_list(graph_string), 0) == [None, 0, 0])
    tests.append(shortest_path(adjacency_list(graph_string), 0, 1) == [0,1])
    tests.append(shortest_path(adjacency_list(graph_string), 1, 2) == [1,0,2])
    tests.append(shortest_path(adjacency_list(graph_string), 2, 1) == [])
    tests.append(connected_components(adjacency_list(graph_string)) == set([(0,1,2)]))
    tests.append(graph_transposed(adjacency_list(graph_string)) == [[(1, None)], [(0, None)], [(0, None)]])
    tests.append(strongly_connected(adjacency_list(graph_string)) == True)
    print(all(tests), tests)

    tests = []
    graph_string = """\
    D 3 W
    0 1 7
    1 0 -2
    0 2 0
    """
    tests.append(adjacency_matrix(graph_string) == [[None, 7, 0],[-2,None, None],[None, None, None]])
    tests.append(adjacency_list(graph_string) == [[(1, 7), (2, 0)], [(0, -2)], []])
    tests.append(dfs_tree(adjacency_list(graph_string), 0, []) == (['P','P','P'],[None, 0, 0],[1,2,0]) )
    tests.append(bfs_tree(adjacency_list(graph_string), 0) == [None, 0, 0])
    tests.append(shortest_path(adjacency_list(graph_string), 0, 1) == [0,1])
    tests.append(shortest_path(adjacency_list(graph_string), 1, 2) == [1,0,2])
    tests.append(shortest_path(adjacency_list(graph_string), 2, 1) == [])
    tests.append(connected_components(adjacency_list(graph_string)) == set([(0,1,2)]))
    tests.append(graph_transposed(adjacency_list(graph_string)) == [[(1, -2)], [(0, 7)], [(0, 0)]])
    tests.append(strongly_connected(adjacency_list(graph_string)) == True)
    print(all(tests), tests)

    tests = []
    graph_string = """\
    U 7
    1 2
    1 5
    1 6
    2 3
    2 5
    3 4
    4 5
    """
    tests.append(adjacency_matrix(graph_string) == [
        [None, None, None, None, None, None, None],
        [None, None, 1, None, None, 1, 1],
        [None, 1, None, 1, None, 1, None],
        [None, None, 1, None, 1, None, None],
        [None, None, None, 1, None, 1, None],
        [None, 1, 1, None, 1, None, None],
        [None, 1, None, None, None, None, None]])
    tests.append(adjacency_list(graph_string) == [
     [],
     [(2, None), (5, None), (6, None)],
     [(1, None), (3, None), (5, None)],
     [(2, None), (4, None)],
     [(3, None), (5, None)],
     [(1, None), (2, None), (4, None)],
     [(1, None)]])
    tests.append(dfs_tree(adjacency_list(graph_string), 1, []) == (['U','P','P','P','P','P','P'],
                                                            [None, None, 1, 2, 3, 4, 1],
                                                            [5,4,3,2,6,1]) )
    tests.append(bfs_tree(adjacency_list(graph_string), 1) == [None, None, 1, 2, 5, 1, 1])
    tests.append(shortest_path(adjacency_list(graph_string), 1, 4) == [1,5,4])
    tests.append(shortest_path(adjacency_list(graph_string), 4, 1) == [4,5,1])
    tests.append(shortest_path(adjacency_list(graph_string), 4, 0) == [])
    tests.append(connected_components(adjacency_list(graph_string)) == set([(0,), (1,2,3,4,5,6)]))
    tests.append(graph_transposed(adjacency_list(graph_string)) == [
     [],
     [(2, None), (5, None), (6, None)],
     [(1, None), (3, None), (5, None)],
     [(2, None), (4, None)],
     [(3, None), (5, None)],
     [(1, None), (2, None), (4, None)],
     [(1, None)]])
    tests.append(cycle_detection(adjacency_list(graph_string), "U"))
    print(all(tests), tests)  

    tests = []
    graph_string = """\
    U 7 W
    1 2 13
    1 5 22
    1 6 53
    2 3 42
    2 5 45
    3 4 66 
    4 5 72
    """
    tests.append(adjacency_matrix(graph_string) == [
        [None, None, None, None, None, None, None],
        [None, None, 13  , None, None, 22  , 53  ],
        [None, 13  , None, 42  , None, 45  , None],
        [None, None, 42  , None, 66  , None, None],
        [None, None, None, 66  , None, 72  , None],
        [None, 22  , 45  , None, 72  , None, None],
        [None, 53  , None, None, None, None, None]])
    tests.append(adjacency_list(graph_string) == [
     [],
     [(2, 13), (5, 22), (6, 53)],
     [(1, 13), (3, 42), (5, 45)],
     [(2, 42), (4, 66)],
     [(3, 66), (5, 72)],
     [(1, 22), (2, 45), (4, 72)],
     [(1, 53)]])
    tests.append(connected_components(adjacency_list(graph_string)) == set([(0,), (1,2,3,4,5,6)]))
    tests.append(strongly_connected(adjacency_list(graph_string)) == False)
    print(all(tests), tests)

    tests = []
    graph_string = """\
    D 7 W
    1 2 13
    1 5 22
    1 6 53
    2 3 42
    2 5 45
    3 4 66 
    4 5 72
    """
    tests.append(adjacency_matrix(graph_string) == [
        [None, None, None, None, None, None, None],
        [None, None, 13  , None, None, 22  , 53  ],
        [None, None, None, 42  , None, 45  , None],
        [None, None, None, None, 66  , None, None],
        [None, None, None, None, None, 72  , None],
        [None, None, None, None, None, None, None],
        [None, None, None, None, None, None, None]])
    tests.append(adjacency_list(graph_string) == [
     [],
     [(2, 13), (5, 22), (6, 53)],
     [(3, 42), (5, 45)],
     [(4, 66)],
     [(5, 72)],
     [],
     []])
    tests.append(connected_components(adjacency_list(graph_string)) == set([(0,), (1,2,3,4,5,6)]))
    tests.append(strongly_connected(adjacency_list(graph_string)) == False)
    print(all(tests), tests)

    tests = []
    graph_string = """\
    U 17
    1 2
    1 15
    1 6
    12 13
    2 15
    13 4
    4 5
    """
    tests.append(adjacency_list(graph_string) == [[],
     [(2, None), (15, None), (6, None)],
     [(1, None), (15, None)],
     [],
     [(13, None), (5, None)],
     [(4, None)],
     [(1, None)],
     [],
     [],
     [],
     [],
     [],
     [(13, None)],
     [(12, None), (4, None)],
     [],
     [(1, None), (2, None)],
     []])
    tests.append(connected_components(adjacency_list(graph_string)) == set([(0,), (1,2,6,15), (3,),
                                                                     (4,5,12,13),(7,),(8,),(9,),
                                                                     (10,),(11,),(14,),(16,) ]))
    print(all(tests), tests)

    tests = []
    graph_string = """\
    U 6
    3 0
    2 5
    1 0
    1 3
    """
    tests.append(connected_components(adjacency_list(graph_string)) == set([(0,1,3), (2,5), (4,)]))
    print(all(tests), tests)

    tests = []
    graph_string = """\
    D 7
    0 3
    4 0
    5 3
    3 2
    4 5
    """
    tests.append(topological_sorting(adjacency_list(graph_string)) == [2,3,0,1,5,4,6])
    tests.append(cycle_detection(adjacency_list(graph_string)) == False)
    print(all(tests), tests)

    tests = []
    graph_string = """\
    D 2
    0 1
    1 0
    """
    tests.append(cycle_detection(adjacency_list(graph_string)))
    graph_string = """\
    D 3
    0 1
    1 2
    2 0
    """
    tests.append(cycle_detection(adjacency_list(graph_string)))
    graph_string = """\
    D 5
    0 1
    1 2
    2 3
    3 4
    3 1
    """
    tests.append(cycle_detection(adjacency_list(graph_string)))
    graph_string = """\
    D 6
    0 1
    1 2
    2 3
    4 5
    5 4
    """
    tests.append(cycle_detection(adjacency_list(graph_string)))
    graph_string = """\
    U 6
    0 1
    1 2
    2 3
    3 4
    4 1
    4 5
    """
    tests.append(cycle_detection(adjacency_list(graph_string), "U"))
    graph_string = """\
    U 6
    0 1
    1 2
    2 3
    4 5
    5 4
    """
    tests.append(cycle_detection(adjacency_list(graph_string), "U") == False)
    graph_string = """\
    U 2
    0 1
    """
    tests.append(cycle_detection(adjacency_list(graph_string), "U") == False)
    print(all(tests), tests)

    tests = []
    graph_string = """\
    U 7 W
    0 1 5
    0 2 7
    0 3 12
    1 2 9
    2 3 4
    1 4 7
    2 4 4
    2 5 3
    3 5 7
    4 5 2
    4 6 5
    5 6 2
    """
    tests.append(prim(adjacency_list(graph_string)) == ([None, 0, 0, 2, 5, 2, 5], [0, 5, 7, 4, 2, 3, 2]))
    tests.append(dijkstra(adjacency_list(graph_string)) == ([None, 0, 0, 2, 2, 2, 5], [0, 5, 7,11,11,10,12]))
    tests.append(shortest_path(adjacency_list(graph_string), 0, 4) == [0, 1, 4])# path with least vertices traversed (optimised for less vertex/city visits)
    tests.append(weighted_mst_path(adjacency_list(graph_string), 0, 4) == [(0, 0), (2, 7), (5, 3), (4, 2)])# path on a MST (optimised for network/roads)
    tests.append(sum([x[1] for x in weighted_mst_path(adjacency_list(graph_string), 0, 4)]) == 0+7+3+2)# cost of the MST path from 0->4
    tests.append(weighted_shortest_path(adjacency_list(graph_string), 0, 4) == [(0, 0), (2, 7), (4, 11)])# lowest cost path (optimised for traveller)
    tests.append(weighted_shortest_path(adjacency_list(graph_string), 0, 4)[-1][1] == 11)# cost of the shortest path from 0->4
    tests.append(cycle_detection(adjacency_list(graph_string), "U"))
    print(all(tests), tests)




    # comparing "generic graph search and explicit_graph" with "graph xxx_tree() and _tree_path()"
    adj_list = adjacency_list(graph_string)
    
    # generate dfs_tree parent array
    _, parent, _ = dfs_tree(adj_list, 0) 
    print("dfs path:", _tree_path(parent, 0, 4))
    print("bfs shortest_path:", shortest_path(adj_list, 0, 4))
    print("dijkstra weighted_shortest_path:", weighted_shortest_path(adj_list, 0, 4))

    
    # create inputs for the explicit graph function
    edges = []
    for i in range(len(adj_list)):# for each source node
        for j in range(len(adj_list[i])):# add each edge from the source node to all its neighbours
            edges.append((str(i), str(adj_list[i][j][0]), adj_list[i][j][1]))           
    nodes = set([str(n) for n in range(len(adj_list))])  
    graph = ExplicitGraph(
        nodes,
        edges,
        starting_nodes=['0'],
        goal_nodes={'4'}
    )
    print("generic dfs search")
    solution = next(generic_search(graph, DFSFrontier()))
    print_actions(solution)
    print("generic bfs search")
    solution = next(generic_search(graph, BFSFrontier()))
    print_actions(solution)
    print("generic lcfs search")
    solution = next(generic_search(graph, LCFSFrontier()))
    print_actions(solution)

        
if __name__ == '__main__':
    main()
    



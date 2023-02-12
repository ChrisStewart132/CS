from queue import deque

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

def dfs_tree(adj_list, start, stack, state=None, parent=None):
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
    state = ['U' for x in range(n)]# All Vertices Undiscovered
    parent = [None for x in range(n)]# Parent array indicating parent[i]=p, i is the child vertex and p parent
    state[start] = 'D'# Starting Node set to Discovered
    return dfs_loop(adj_list, start, state, parent, stack)

def dfs_loop(adj_list, u, state, parent, stack):
    for v, w in adj_list[u]:# For each Vertex (v) connected to Vertex u (previous/starting Vertex)
        if state[v] == 'U':# If v has not yet been Discovered             
            state[v] = 'D'# Set v Discovered
            parent[v] = u# Set v's parent as u
            dfs_loop(adj_list, v, state, parent, stack)# Recursively traverse deeper to the children of v
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
    return

def cycle_detection(adj_list):
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
    return

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

def main():  
    graph_string = """\
    D 3
    0 1
    1 0
    0 2
    """   
    print(adjacency_matrix(graph_string) == [[None, 1, 1],[1,None, None],[None, None, None]])
    print(adjacency_list(graph_string) == [[(1, None), (2, None)], [(0, None)], []])
    print(dfs_tree(adjacency_list(graph_string), 0, []) == (['P','P','P'],[None, 0, 0],[1,2,0]) )
    print(bfs_tree(adjacency_list(graph_string), 0) == [None, 0, 0])
    print(shortest_path(adjacency_list(graph_string), 0, 1) == [0,1])
    print(shortest_path(adjacency_list(graph_string), 1, 2) == [1,0,2])
    print(shortest_path(adjacency_list(graph_string), 2, 1) == [])
    print(connected_components(adjacency_list(graph_string)) == set([(0,1,2)]))
    print(graph_transposed(adjacency_list(graph_string)) == [[(1, None)], [(0, None)], [(0, None)]])
    print(strongly_connected(adjacency_list(graph_string)) == True)
          
    graph_string = """\
    D 3 W
    0 1 7
    1 0 -2
    0 2 0
    """
    print(adjacency_matrix(graph_string) == [[None, 7, 0],[-2,None, None],[None, None, None]])
    print(adjacency_list(graph_string) == [[(1, 7), (2, 0)], [(0, -2)], []])
    print(dfs_tree(adjacency_list(graph_string), 0, []) == (['P','P','P'],[None, 0, 0],[1,2,0]) )
    print(bfs_tree(adjacency_list(graph_string), 0) == [None, 0, 0])
    print(shortest_path(adjacency_list(graph_string), 0, 1) == [0,1])
    print(shortest_path(adjacency_list(graph_string), 1, 2) == [1,0,2])
    print(shortest_path(adjacency_list(graph_string), 2, 1) == [])
    print(connected_components(adjacency_list(graph_string)) == set([(0,1,2)]))
    print(graph_transposed(adjacency_list(graph_string)) == [[(1, -2)], [(0, 7)], [(0, 0)]])
    print(strongly_connected(adjacency_list(graph_string)) == True)
          
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
    print(adjacency_matrix(graph_string) == [
        [None, None, None, None, None, None, None],
        [None, None, 1, None, None, 1, 1],
        [None, 1, None, 1, None, 1, None],
        [None, None, 1, None, 1, None, None],
        [None, None, None, 1, None, 1, None],
        [None, 1, 1, None, 1, None, None],
        [None, 1, None, None, None, None, None]])
    print(adjacency_list(graph_string) == [
     [],
     [(2, None), (5, None), (6, None)],
     [(1, None), (3, None), (5, None)],
     [(2, None), (4, None)],
     [(3, None), (5, None)],
     [(1, None), (2, None), (4, None)],
     [(1, None)]])
    print(dfs_tree(adjacency_list(graph_string), 1, []) == (['U','P','P','P','P','P','P'],
                                                            [None, None, 1, 2, 3, 4, 1],
                                                            [5,4,3,2,6,1]) )
    print(bfs_tree(adjacency_list(graph_string), 1) == [None, None, 1, 2, 5, 1, 1])
    print(shortest_path(adjacency_list(graph_string), 1, 4) == [1,5,4])
    print(shortest_path(adjacency_list(graph_string), 4, 1) == [4,5,1])
    print(shortest_path(adjacency_list(graph_string), 4, 0) == [])
    print(connected_components(adjacency_list(graph_string)) == set([(0,), (1,2,3,4,5,6)]))
    print(graph_transposed(adjacency_list(graph_string)) == [
     [],
     [(2, None), (5, None), (6, None)],
     [(1, None), (3, None), (5, None)],
     [(2, None), (4, None)],
     [(3, None), (5, None)],
     [(1, None), (2, None), (4, None)],
     [(1, None)]])
    
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
    print(adjacency_matrix(graph_string) == [
        [None, None, None, None, None, None, None],
        [None, None, 13  , None, None, 22  , 53  ],
        [None, 13  , None, 42  , None, 45  , None],
        [None, None, 42  , None, 66  , None, None],
        [None, None, None, 66  , None, 72  , None],
        [None, 22  , 45  , None, 72  , None, None],
        [None, 53  , None, None, None, None, None]])
    print(adjacency_list(graph_string) == [
     [],
     [(2, 13), (5, 22), (6, 53)],
     [(1, 13), (3, 42), (5, 45)],
     [(2, 42), (4, 66)],
     [(3, 66), (5, 72)],
     [(1, 22), (2, 45), (4, 72)],
     [(1, 53)]])
    print(connected_components(adjacency_list(graph_string)) == set([(0,), (1,2,3,4,5,6)]))
    print(strongly_connected(adjacency_list(graph_string)) == False)
    
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
    print(adjacency_matrix(graph_string) == [
        [None, None, None, None, None, None, None],
        [None, None, 13  , None, None, 22  , 53  ],
        [None, None, None, 42  , None, 45  , None],
        [None, None, None, None, 66  , None, None],
        [None, None, None, None, None, 72  , None],
        [None, None, None, None, None, None, None],
        [None, None, None, None, None, None, None]])
    print(adjacency_list(graph_string) == [
     [],
     [(2, 13), (5, 22), (6, 53)],
     [(3, 42), (5, 45)],
     [(4, 66)],
     [(5, 72)],
     [],
     []])
    print(connected_components(adjacency_list(graph_string)) == set([(0,), (1,2,3,4,5,6)]))
    print(strongly_connected(adjacency_list(graph_string)) == False)
    
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
    print(adjacency_list(graph_string) == [[],
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
    print(connected_components(adjacency_list(graph_string)) == set([(0,), (1,2,6,15), (3,),
                                                                     (4,5,12,13),(7,),(8,),(9,),
                                                                     (10,),(11,),(14,),(16,) ]))

    graph_string = """\
    U 6
    3 0
    2 5
    1 0
    1 3
    """
    print(connected_components(adjacency_list(graph_string)) == set([(0,1,3), (2,5), (4,)]))

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
    print(prim(adjacency_list(graph_string)) == ([None, 0, 0, 2, 5, 2, 5], [0, 5, 7, 4, 2, 3, 2]))
    print(dijkstra(adjacency_list(graph_string)) == ([None, 0, 0, 2, 2, 2, 5], [0, 5, 7,11,11,10,12]))
    print(shortest_path(adjacency_list(graph_string), 0, 4) == [0, 1, 4])# path with least vertices traversed (optimised for less vertex/city visits)
    print(weighted_mst_path(adjacency_list(graph_string), 0, 4) == [(0, 0), (2, 7), (5, 3), (4, 2)])# path on a MST (optimised for network/roads)
    print(weighted_shortest_path(adjacency_list(graph_string), 0, 4) == [(0, 0), (2, 7), (4, 11)])# lowest cost path (optimised for traveller)

    
if __name__ == '__main__':
    main()
    



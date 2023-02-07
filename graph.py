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
    
if __name__ == '__main__':
    main()
    


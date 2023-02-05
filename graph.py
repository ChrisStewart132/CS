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

def main():  
    graph_string = """\
    D 3
    0 1
    1 0
    0 2
    """
    print(adjacency_matrix(graph_string) == [[None, 1, 1],[1,None, None],[None, None, None]])
    print(adjacency_list(graph_string) == [[(1, None), (2, None)], [(0, None)], []])

    graph_string = """\
    D 3 W
    0 1 7
    1 0 -2
    0 2 0
    """
    print(adjacency_matrix(graph_string) == [[None, 7, 0],[-2,None, None],[None, None, None]])
    print(adjacency_list(graph_string) == [[(1, 7), (2, 0)], [(0, -2)], []])

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



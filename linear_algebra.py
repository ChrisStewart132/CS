ACCURACY = 0.0001

class Matrix:
    def __init__(self, m=3, n=3, val=0, data=None):
        """
        m rows (height), n columns (width), val=default_value, data=matrix_to_use (not copy)
        """
        self.data = data if data else [[val for j in range(n)] for i in range(m)]

    def height(self):
        return len(self.data)# number of rows, m
    def width(self):
        return len(self.data[0]) if len(self.data) > 0 else 0# number of cols, n
    def __len__(self):
        return len(self.data) * len(self.data[0])# m * n
    
    def __add__(self, other):
        if isinstance(other, Matrix):
            m,n,m2,n2 = self.height(), self.width(), other.height(), other.width()
            if m == m2 and n == n2:
                copy = [[self.data[i][j] + other.data[i][j] for j in range(n)] for i in range(m)]
                return Matrix(m, n, None, copy)
        
    def __sub__(self, other):
        if isinstance(other, Matrix):
            m, n, m2, n2 = self.height(), self.width(), other.height(), other.width()
            if m == m2 and n == n2:
                copy = [[self.data[i][j] - other.data[i][j] for j in range(n)] for i in range(m)]
                return Matrix(m, n, None, copy)
        
    def __mul__(self, other):
        if isinstance(other, (int,float)):# scale the matrix
            for i in range(self.height()):
                for j in range(self.width()):
                    self.data[i][j] *= other
            return self
        elif isinstance(other, Matrix):# transform the matrix
            return self._matrix_multiplication(other)
        elif isinstance(other, (list,tuple)):# A*x = b
            x = Matrix(0,0,0,[other])# convert list to matrix
            x = x.transpose()# transpose the list to a column vector/matrix
            return self*x# return A*x
        
    def _matrix_multiplication(self, other):
        if self.width() == other.height():
            output = [[0 for j in range(other.width())] for i in range(self.height())]
            for i in range(self.height()):
                for j in range(other.width()):
                    x,y,s = 0,0,0
                    while x < self.width():
                        s += self.data[i][x] * other.data[y][j]
                        x+=1# self col index
                        y+=1# other row index
                    output[i][j] = s
            return Matrix(None,None,None,output)

    def __pow__(self, exponent):
        m=self
        for i in range(exponent-1):
            m = self._matrix_multiplication(m)
        return m

    def __eq__(self, other):       
        if isinstance(other, Matrix):
            m,n,m2,n2 = self.height(), self.width(), other.height(), other.width()
            if m==m2 and n==n2:
                for i in range(m):
                    for j in range(n):
                        delta = abs(other.data[i][j] - self.data[i][j])
                        if delta > ACCURACY:
                            return False
                return True
        return False
            
    def __equals__(self, other):       
        if isinstance(other, Matrix):
            self.data = other.data

    def transpose(self):
        return Matrix(None,None,None,[[self.data[j][i] for j in range(self.height())] for i in range(self.width())])

    def augment(self, b):
        AT = self.transpose()
        AT.data.append(b)
        return AT.transpose()

    def __repr__(self):
        return str(self)
    
    def __str__(self):
        output = ""
        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                output += f"{self.data[i][j]:.2f} " if isinstance(self.data[i][j], float) else f"{self.data[i][j]} "
            output += "\n"
        return output

    def row_echelon_form(self, operations=None):
        """
        takes an augmented matrix ([a1 a2 a3 : b]) and returns a copy of it in row echelon form (triangle matrix)
            optional: operations list to store operations performed
        """
        if operations == None:
            operations = []

        m = [[self.data[i][j] for j in range(self.width())] for i in range(self.height())]
        free_vars = 0
        for x in range(self.width()):
            for y in range(x-free_vars, self.height()):
                pivot_value = m[x-free_vars][x]
                if pivot_value == 0 and m[y][x] != 0:# find the next pivot row
                    if y > x-free_vars:# swap pivot row to correct position
                        m[y], m[x-free_vars] = m[x-free_vars], m[y]
                        operations.append(f"swap R{y}<->R{x}")
                elif pivot_value != 0 and y != x-free_vars:# eliminate remaining rows using the pivot row (excluding pivot row)
                    ratio = -m[y][x] / m[x-free_vars][x]
                    for j in range(x, self.width()):
                        m[y][j] += m[x-free_vars][j]*ratio
                elif pivot_value == 0 and y == self.height()-1:
                    free_vars += 1

        return Matrix(0,0,0,m)

    def range(self):
        m = self.row_echelon_form().data
        rank = self.height()
        for i in range(self.height()-1,-1,-1):
            if m[i] == [0 for x in range(self.width())]:# full zero row
                rank -= 1
        return rank

    def back_substitution(self, free_index=None):
        """
        solves an augmented triangular matrix
        x1 x2 x3 : b
        1  2  3  : 8
        0  2  2  : 4
        0  0  2  : 1

        1(x1) + 2(x2) + 3(x3) = 8
                2(x2) + 2(x3) = 4
                        2(x3) = 1

        returns a tuple:
            basic solution vector (x1,x2,x3)
            list of all free variable vectors [t(1,1,0), s(4,0,1)] etc (without the t/s)     
        """
        m = [[self.data[i][j] for j in range(self.width())] for i in range(self.height())]
        solution = [None for x in range(self.width()-1)]# [x1,x2...xn]
        if free_index:# set a known free variable index to find a different solution
            solution[free_index] = 1# set the free variable, e.g. t = 1 where x = (..,..,..) + t(..,..,..)
        pivots, free_vars = [0]*(self.width()-1), [0]*(self.width()-1)
        
        for y in range(self.height()-1,-1,-1):
            if m[y] == [0] * self.width():
                continue
            else:
                for x in range(self.width()-1):# A[y][0] -> A[y][-1]
                    if abs(m[y][x]) != 0:# found pivot point
                        break
                pivot = m[y][x]
                pivots[x] = 1# set the variable xn as constraint (not free)

                # zero all free variables (except the specified free_var) to get a simple solution
                for j in range(x+1, self.width()-1):
                    if pivots[j] == 0 and j != free_index:
                        m[y][j], solution[j] = 0, 0
                        free_vars[j] = 1
                                        
                rhs = m[y][-1] - sum([solution[j]*m[y][j] for j in range(x+1, self.width()-1)])# b - a1.x1 - a2.x2...
                solution[x] = rhs / m[y][x]



        free_vectors = []# free variable equations
        if sum(free_vars) > 0 and free_index == None:# first function call calls for all children free variables
            for i in range(len(free_vars)):
                if free_vars[i] == 1:
                    free_solution, _ = self.back_substitution(i)
                    free_vector = [free_solution[j] - solution[j] for j in range(len(solution))]
                    free_vectors.append(tuple(free_vector))
            
        return tuple(solution), free_vectors
                
    def determinant(self, method=1):
        """
        The Determinant is simply how much an input vector is scaled when transformed by a full rank matrix.

        det(A) = signed scaling factor = +- area of image region / area of original region
        det(AB) = det(A)*det(B)
        if A is invertible (det(A) != 0), det(A) = 1/det(A^-1)
        det(a|b|c+d) = det(a|b|c) + det(a|b|d), where a,b,c,d are columns
        det(a|kb) = k*det(a|b)
        det(kA) = k^n * det(A) for an nxn matrix
        det(A) = det(A^T)

        Summary: Effect of Row Operations on the Determinant
            • Adding a multiple of one row to another row does not change
                the determinant.
            • Swapping two rows changes the sign of the determinant.
            • Multiplying a row by a scalar multiplies the determinant by the
                scalar.

        Method1:
            if a matrix is diagonal or triangular, the determinant == product of all diagonal entries.
            Therefore it is possible to use row operations to get the row echelon matrix
            saving each operation performed to transform the calculated determinant by the above rules.

        Method2:
            co-factor expansion
                   [a b]
            det of [c d] = ad-bc

                   [a b c]
                   [d e f]        [e f]        [d f]        [d e]
            det of [g h k] = a*det[h k] - b*det[g k] + c*det[g h]       
        """
        if self.height() != self.width():
            return 0
        
        if method == 1:
            operations, det = [], 1
            row_echelon_form = self.row_echelon_form(operations)
            for i in range(self.width()):
                det *= row_echelon_form.data[i][i]
            # transform det with operations performed during gaussian elimination to echelon form
            return det if len(operations) % 2 == 0 else -det
        elif method == 2:
            det = self._cofactor_expansion(self.data)
            return det if self.height() < 4 else -det# hack fixing sign reversed above order 9 (3x3)

    def _cofactor_expansion(self, table, depth=0):
        if len(table) == 2:
            det = table[0][0]*table[1][1] - table[0][1]*table[1][0]
            return det
                
        output = 0
        for j in range(len(table)):           
            sub_matrix = [table[y][:j] + table[y][j+1:] for y in range(1,len(table))]
            child_det = self._cofactor_expansion(sub_matrix, depth+1)
            if (j+depth) % 2 == 0:# + - + -\n- + - +\n+ - + -...
                output += table[0][j]*child_det
            else:
                output -= table[0][j]*child_det
        
        return output

def main():
    
    print("eq, equals")
    tests = []
    tests.append(Matrix(3,4,16) == Matrix(3,4,16))
    m = Matrix(3,3,None,[[(i+1)*(j+1) for j in range(3)]for i in range(3)])
    m2 = Matrix(3,3,None,[[((i+1)*(j+1))-1 for j in range(3)]for i in range(3)])
    tests.append(m == m2+Matrix(3,3,1))
    tests.append(m != m2)
    m = Matrix(2,2,2)
    tests.append(m == Matrix(2,2,2))
    print(" ",all(tests), tests)
    
    print("add, sub, mul")
    tests = []
    m = Matrix(3,4,4)
    m2 = Matrix(3,4,1)
    tests.append(m+m2 == Matrix(3,4,5))
    tests.append(m-m2 == Matrix(3,4,3))
    tests.append(m2-m == Matrix(3,4,-3))
    tests.append(m2*2 == Matrix(3,4,2))
    tests.append(m*4 == Matrix(3,4,16))
    print(" ",all(tests), tests)

    print("matrix_multiplication, pow")
    tests = []
    m = Matrix(3,3,None,[[(i+1)*(j+1) for j in range(3)]for i in range(3)])
    m2 = Matrix(3,3,None,[[((i+1)*(j+1))**2 for j in range(3)]for i in range(3)])
    m_times_m2 = [
        [36,144,324],
        [72,288,648],
        [108,432,972]
        ]
    tests.append(m*m2 == Matrix(3,3,None,m_times_m2))
    m2_times_m = [
        [36,72,108],
        [144,288,432],
        [324,648,972]
        ]
    tests.append(m2*m == Matrix(3,3,None,m2_times_m))
    tests.append(Matrix(3,3,1)**2 == Matrix(3,3,3))
    tests.append(Matrix(3,3,1)**3 == Matrix(3,3,9))
    tests.append(m**2 == m*m)
    tests.append(m2**2 == m2*m2)
    tests.append(m**3 == m*m*m)
    tests.append(m2**3 == m2*m2*m2)
    tests.append(m2**8 == m2*m2*m2*m2*m2*m2*m2*m2)
    print(" ",all(tests), tests)

    print("transpose")
    tests = []
    m = Matrix(3,3,None,[[(i+1)*(j+1) for j in range(3)]for i in range(3)])# symmetric
    m2 = Matrix(3,3,None,[[((i+1)*(j+1))**2 for j in range(3)]for i in range(3)])
    m2_transposed = [
        [1,4,9],
        [4,16,36],
        [9,36,81]
        ]
    tests.append(m.transpose() == m)
    tests.append(m2.transpose() == Matrix(None,None,None,m2_transposed))
    tests.append(Matrix(3,1,1).transpose() == Matrix(1,3,1))
    tests.append(Matrix(7,2,1).transpose() == Matrix(2,7,1))
    tests.append(Matrix(7,2,-1).transpose() != Matrix(2,7,1))   
    print(" ",all(tests), tests)


    print("full rank Gaussian elimination and back substitution")
    tests = []

    # identity matrix with x1,x2,x3 = 1,2,3
    Ab = [
          [0,0,1,3],
          [0,1,0,2],
          [1,0,0,1]
          ]
    m=Matrix(0,0,0,Ab)
    m_row_echelon = Matrix(0,0,0,
    [
          [1,0,0,1],
          [0,1,0,2],
          [0,0,1,3]
          ])    
    tests.append(m.row_echelon_form() == m_row_echelon)
    tests.append(m.range() == 3)
    solution = (1,2,3)
    calculated_solution, free_solutions = m.row_echelon_form().back_substitution()
    # testing solution is close in accuracy (as floating point errors occur)
    tests.append(sum([abs(solution[i] - x) for i, x in enumerate(calculated_solution)]) < ACCURACY)
    tests.append(len(free_solutions) == 0)
    
    # example matrix with x1,x2,x3 = 1,2,3
    Ab = [
          [3,2,8,31],
          [2,1,3,13],
          [3,4,1,14]
          ]
    m=Matrix(0,0,0,Ab)
    m_row_echelon = Matrix(0,0,0,
    [
        [3,2,8,31],
        [0,1-2*2/3,3-8*2/3,13-31*2/3],
        [0,0,-21,-63]
    ])
    
    tests.append(m.row_echelon_form() == m_row_echelon)
    tests.append(m.range() == 3)
    solution = (1,2,3)
    calculated_solution, free_solutions = m.row_echelon_form().back_substitution()
    # testing solution is close in accuracy (as floating point errors occur)
    tests.append(sum([abs(solution[i] - x) for i, x in enumerate(calculated_solution)]) < ACCURACY*len(solution))
    tests.append(len(free_solutions) == 0)
    print(" ",all(tests), tests)


    print("determinant")
    tests = []
    # 2x2
    table = [
            [5,1],
            [3,1]
            ]
    m = Matrix(0,0,0,table)
    tests.append(abs(m.determinant() - (5*1-3*1)) < ACCURACY)
    tests.append(m.range() == 2)
    tests.append(abs(m.determinant()*m.determinant() - (m*m).determinant()) < ACCURACY)
    # 2x2 non full rank / singular matrix
    table = [
            [5,10],
            [3,6]
            ]
    m = Matrix(0,0,0,table)
    tests.append(m.determinant() == 0)
    tests.append(m.range() == 1)
    # 2x3
    table = [
            [5,1,0],
            [3,1,1]
            ]
    m = Matrix(0,0,0,table)
    tests.append(m.determinant() == 0)
    tests.append(m.range() == 2)
    # 3x3
    table = [
            [2,4,3],
            [0,5,1],
            [3,1,0]
            ]
    m = Matrix(0,0,0,table)
    tests.append(abs(m.determinant() - -35) < ACCURACY)
    tests.append(m.range() == 3)
    tests.append(abs(m.determinant()*m.determinant() - (m*m).determinant()) < ACCURACY)
    m = m * -1
    tests.append(abs(m.determinant() - 35) < ACCURACY)
    tests.append(m.range() == 3)
    tests.append(abs(m.determinant()*m.determinant() - (m*m).determinant()) < ACCURACY)
    # 4x4
    table = [
            [2,4,3,9],
            [0,5,1,4],
            [3,1,0,6],
            [0,3,4,1]
            ]
    m = Matrix(0,0,0,table)
    tests.append(abs(m.determinant() - 168) < ACCURACY)
    tests.append(m.range() == 4)
    tests.append(abs(m.determinant()*m.determinant() - (m*m).determinant()) < ACCURACY)   
    m = m * -1 
    tests.append(abs(m.determinant() - 168) < ACCURACY)
    tests.append(abs(m.determinant()*m.determinant() - (m*m).determinant()) < ACCURACY)
    # 5x5
    table = [
            [7,5,9,85,9],
            [7,-3,-3,7,2],
            [8,-2,8,-7,9],
            [9,3,7,54,-2],
            [-9,-7,3,-9,4]
            ]
    m = Matrix(0,0,0,table)
    tests.append(abs(m.determinant() - -800052) < ACCURACY)
    tests.append(m.range() == 5)
    tests.append(abs(m.determinant()*m.determinant() - (m*m).determinant()) < ACCURACY)
    m = m * -1
    tests.append(abs(m.determinant() - 800052) < ACCURACY)
    tests.append(abs(m.determinant()*m.determinant() - (m*m).determinant()) < ACCURACY)  
    # 9x9
    table = [
            [-1,2,3,4,5,6,7,8,9],
            [1,-2,3,4,5,6,7,8,9],
            [1,2,-3,4,5,6,7,8,9],
            [1,2,3,-4,5,6,7,8,9],
            [1,2,3,4,-5,6,7,8,9],
            [1,2,3,4,5,-6,7,8,9],
            [1,2,3,4,5,6,-7,8,9],
            [1,2,3,4,5,6,7,-8,9],
            [1,2,3,4,5,6,7,8,-9]
            ]
    m = Matrix(0,0,0,table)
    tests.append(abs(m.determinant() - 650280960) < ACCURACY)
    tests.append(m.range() == 9)
    tests.append(abs(m.determinant()*m.determinant() - (m*m).determinant()) < ACCURACY*10000000)# large determinant and therefore error
    print(" ",all(tests), tests)



    print("Singular / non-full rank Gaussian elimination and back substitution")
    tests = []
  
    A = [
          [1,0,1,0,4],
          [0,1,1,0,5],
          [0,0,0,0,0],
          [0,0,0,1,4]
          ]
    m=Matrix(0,0,0,A)
    m_row_echelon = Matrix(0,0,0,
    [
          [1,0,1,0,4],
          [0,1,1,0,5],
          [0,0,0,1,4],
          [0,0,0,0,0]
          ])    
    tests.append(m.row_echelon_form() == m_row_echelon)
    tests.append(m.range() == 3)
    solution = (4,5,0,4)
    calculated_solution, free_solutions = m.row_echelon_form().back_substitution()
    tests.append(sum([abs(solution[i] - x) for i, x in enumerate(calculated_solution)]) < ACCURACY*len(solution))
    A = [
          [1,0,1,0],
          [0,1,1,0],
          [0,0,0,0],
          [0,0,0,1]
          ]
    m=Matrix(0,0,0,A)
    tests.append(m*calculated_solution == Matrix(0,0,0,[[4,5,0,4]]).transpose())# convert list to column Matrix
    tests.append((m*calculated_solution).transpose().data[0] == [4,5,0,4])# same as above but converting b from Matrix to list
    # testing solution with the free variable 'x3' set to 2
    tests.append(m*([calculated_solution[i]+2*free_solutions[0][i] for i in range(len(solution))]) == Matrix(0,0,0,[[4,5,0,4]]).transpose())

    A = [
          [1,0,1,0],
          [0,1,1,0],
          [0,0,0,1]
          ]
    m=Matrix(0,0,0,A)
    m_row_echelon = Matrix(0,0,0,
    [
          [1,0,1,0],
          [0,1,1,0],
          [0,0,0,1],
          ])
    tests.append(m.row_echelon_form() == m_row_echelon)
    tests.append(m.range() == 3)
    A = [
          [1,0,1,0],
          [0,1,1,0]
          ]
    m=Matrix(0,0,0,A)
    m_row_echelon = Matrix(0,0,0,
    [
          [1,0,1,0],
          [0,1,1,0]
          ])
    tests.append(m.row_echelon_form() == m_row_echelon)
    tests.append(m.range() == 2)
    A = [
          [1,0,1,0]
          ]
    m=Matrix(0,0,0,A)
    m_row_echelon = Matrix(0,0,0,
    [
          [1,0,1,0]
          ])
    tests.append(m.row_echelon_form() == m_row_echelon)
    tests.append(m.range() == 1)

    A = [
            [0,0,0,0,0,0,0,0,1],
            [0,0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0,0],
            [1,0,0,0,0,0,0,0,0]
            ]
    m = Matrix(0,0,0,A)
    m_row_echelon = Matrix(0,0,0,
    [
            [1,0,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0,0],
            [0,0,0,1,0,0,0,0,0],
            [0,0,0,0,1,0,0,0,0],
            [0,0,0,0,0,1,0,0,0],
            [0,0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,0,1]
          ])
    tests.append(m.row_echelon_form() == m_row_echelon)
    tests.append(m.range() == 9)
    A = [
            [0,0,0,0,0,0,0,0,1],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,1,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,1,0,0,1,0],
            [0,0,0,1,0,0,0,1,0],
            [0,1,1,0,0,1,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [1,1,0,0,0,0,0,0,0]
            ]
    m = Matrix(0,0,0,A)
    m_row_echelon = Matrix(0,0,0,
    [
            [1,1,0,0,0,0,0,0,0],
            [0,1,1,0,0,1,0,0,0],
            [0,0,0,1,0,0,0,1,0],
            [0,0,0,0,1,0,0,1,0],
            [0,0,0,0,0,1,1,0,0],
            [0,0,0,0,0,0,0,0,1],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0]
          ])

    tests.append(m.row_echelon_form() == m_row_echelon)
    tests.append(m.range() == 6)

    m = Matrix(0,0,0,
    [
            [1,1,0,0,0,0,0,0,0,3],
            [0,1,1,0,0,1,0,0,0,8],
            [0,0,0,1,0,0,0,1,0,4],
            [0,0,0,0,1,0,0,1,0,5],
            [0,0,0,0,0,1,1,0,0,6],
            [0,0,0,0,0,0,0,0,1,9],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0]
          ])

    A = Matrix(0,0,0,
    [
            [1,1,0,0,0,0,0,0,0],
            [0,1,1,0,0,1,0,0,0],
            [0,0,0,1,0,0,0,1,0],
            [0,0,0,0,1,0,0,1,0],
            [0,0,0,0,0,1,1,0,0],
            [0,0,0,0,0,0,0,0,1],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0]
          ])
    b = [3,8,4,5,6,9,0,0,0]
    solution = (1,2,0,4,5,6,0,0,9)
    calculated_solution, free_solutions = m.row_echelon_form().back_substitution()
    tests.append(solution == calculated_solution)
    tests.append(Matrix(0,0,0,[b]).transpose() == A*[calculated_solution[i] + 3*free_solutions[0][i] for i in range(len(solution))])
    tests.append(Matrix(0,0,0,[b]).transpose() == A*[calculated_solution[i] + 3*free_solutions[1][i] for i in range(len(solution))])
    tests.append(Matrix(0,0,0,[b]).transpose() == A*[calculated_solution[i] + 3*free_solutions[2][i] for i in range(len(solution))])

    """
    variables x[iron, copper, tin, coal]
    2 1 1 2 = $4
    2 0 0 1 = $3
    4 2 2 4 = $8
    0 2 2 2 = $2

    x2 = 1-(x3)-(x4)
    2x1 = 4-2(x4)-(x3)-(x2) = 4-2(x4)-(x3)-(1-(x3)-(x4)) = 4-1-2(x4)+(x4)-(x3)+(x3) = 3-(x4)
     x1 = 3/2 - (x4)/2

    (4,3,8,2) = (3/2, 1, 0, 0) + x3(0, -1, 1, 0) + x4(-1/2, -1, 0, 1)

    e.g.
        6 tin, 8 coal, copper = 1-6-8 = -13, iron = 3/2 - 4 = -5/2
    """
    A = [
            [2,1,1,2,4],
            [2,0,0,1,3],
            [4,2,2,4,8],
            [0,2,2,2,2]
            ]
    m = Matrix(0,0,0,A)
    m_row_echelon = Matrix(0,0,0,
    [
            [2,1,1,2,4],
            [0,-1,-1,-1,-1],
            [0,0,0,0,0],
            [0,0,0,0,0]
          ])
    tests.append(m.row_echelon_form() == m_row_echelon)
    solution = (3/2,1,0,0)
    calculated_solution, free_solutions = m.row_echelon_form().back_substitution()
    tests.append(solution == calculated_solution)   
    print(" ",all(tests), tests)


    print("\n\nExample Practical Applications.\n")
    """
    Curve fitting 3 2D points: (1,16),(2,6),(3,58)
    p(x) = ax**2 + bx + c 
    a  b  c : y
    1  1  1 : 16
    4  2  1 : 6
    9  3  1 : 58
    """
    print("\nCurve fitting 3 points [(1,16),(2,6),(3,58)] to a quadratic / bi-nomial\n")
    Ab = [
        [1,1,1,16],
        [4,2,1,6],
        [9,3,1,58]
        ]
    m = Matrix(0,0,0,Ab)
    print(m)
    m_row_echelon = m.row_echelon_form()
    print(m_row_echelon)
    solution, free_solutions = m_row_echelon.back_substitution()
    print(f"y = {solution[0]:.2f}x^2 + {solution[1]:.2f}x + {solution[2]:.2f}")# unique solution



    
    """
    An investment company sells three types of unit investment fund —
    Standard (S), Deluxe(D) and Gold Star (G). The composition of these units is as follows:
        Each unit of S contains 12 shares of stock A, 16 of stock B and 8 of stock C.
        Each unit of D contains 20 shares of stock A, 12 of stock B and 28 of stock C.
        Each unit of G contains 32 shares of stock A, 28 of stock B and 36 of stock C.       
    An investor wishes to purchase exactly 220 shares of stock A, 176 shares of stock B and
    264 shares of stock C by buying units of the three investment funds.
        (a) Set up a system of linear equations for this problem and solve the system.
        (b) Find the combinations of units of S, D and G which will meet the investor’s require-
            ments. (You will need to impose certain restrictions which arise naturally from the problem.)
        (c) Suppose each unit of S, D and G costs the investor $300, $400 and $600 respectively.
            Which investment will minimise the cost to the investor?
    (a)
        Ax=b
        a  b  c  : RHS
        12 20 32 : 220
        16 12 28 : 176
        8  28 36 : 264
    (b)
        row echelon form
        3  5  8 : 55
        0  1  1 : 8
        0  0  0 : 0

        c = free var
        b = 8-c
        3a = 55-8c-5b
        3a = 55-8c-5(8-c)
        3a = 55-8c-40+5c
        3a = 15-3c
        a  = 5 - c

        r = (5, 8, 0) + t(-1, -1, 1), where t is a free variable (can take any form and the solution holds) for (S,D,G)
    (c)  
         because 5,8 is a solution, but 5-1, 8,-1, 1 is also a solution (with t = 1)
            S=300,D=400,G=600, therefore t has a cost of -300-400+600 = - 100
            so maximise t (while not going to negative stock package purchases)

        (5,8,0) + 5(-1,-1,1) = 0,3,5 to minimise cost at 3(400) + 5(600) = 1200 + 3000 = $ 4200
            vs 5(300) + 8(400) = 1500 + 3200 = $ 4700     
    """
    print("\n\nInvestment package optimisation\n")
    Ab = [
          [12,20,32,220],
          [16,12,28,176],
          [8 ,28,36,264]
          ]
    m=Matrix(0,0,0,Ab)
    print(m)
    print(m.row_echelon_form())
    print('x = (5x1,8x2,0x3) + t(-x1,-x2,x3)')# inf solutions
    print('x =', m.row_echelon_form().back_substitution())# basic solution


    def fibonacci(n):
        """
        0,1,1,2,3,5,8,13...
        fibo [f0, f1][0 1]=[f1,f0+f1=f2]
                     [1 1]
        """
        if n < 2:
            return n    
        f = Matrix(0,0,0,[[0,1],[1,1]])
        current = Matrix(0,0,0,[[0,1]])
        for i in range(n-1):
            current = current*f
        return current.data[0][1]
    
    x=[]
    for i in range(8):
        x.append(fibonacci(i))
    print('\n\nfibonacci',0,'->',i,x)


    print("\n\nLeast Squares Approximation: fitting many data points to a polynomial curve\n  In this case a projectile falling from a height of 100m (t, h(m))")
    """
    Assuming a cannon ball is launched horizontally at a height of 100m with a horizontal velocity of 10 m/s and gravity == -9.81 m/s^2
    pos.x = pos.x + v.x*t
    pos.y = pos.y - v.y*t
    v.x   = 10
    v.y   = v.y - g*t
    [px 0 vx 0  0][1  0  0  0  0] = [px+t.vx 0       vx  0      0]
    [0  py 0 vy g][0  1  0  0  0]   [0       py+t.vy 0   vy+t.g g]
    [0  0  0 0  0][t  0  1  0  0]   [0       0       0   0      0]
    [0  0  0 0  0][0  t  0  1  0]   [0       0       0   0      0]
    [0  0  0 0  0][0  0  0  t  1]   [0       0       0   0      0]
    f''(t) = -9.81, f'(t) = -9.81x + v, f(t) = (-9.81/2)x**2 + (v)x + h
    """
    t, g = 1/200, -9.81
    px, py = 0, 100
    vx, vy = 10, 0
    x = Matrix(0,0,0,
        [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [t, 0, 1, 0, 0],
        [0, t, 0, 1, 0],
        [0, 0, 0, t, 1]
        ])
    A = Matrix(0,0,0,
        [
        [px, 0, vx, 0,  0],
        [0,  py, 0, vy, g],
        [0,  0,  0, 0,  0],
        [0,  0,  0, 0,  0],
        [0,  0,  0, 0,  0]
        ]) 
    data_set, time = [], 0
    
    def calculated_height(t, h, v=0):
        return (g/2)*t**2 + v*t + h
    
    while A.data[1][1] > 0:# while height > 0, simulate a physics frame
        A = A*x
        time += t
        data_set.append( ( round(time, 3), round(A.data[1][1], 3) ) )

    print("data =",data_set[:3],"..",data_set[-3:])
    print(f"time:{time:.3f}s, calculated_height:{calculated_height(time,100,0):.3f}, px:{A.data[0][0]:.3f}m, py:{A.data[1][1]:.3f}m, vx:{A.data[0][0]:.3f}m/s, vy:{A.data[1][3]:.3f}m/s\n")

    # using the normal equation A^T.A.x = A^T.b to solve [a,b,c]
    # y = ax^2 + bx + c
    a = []# A matrix
    b = []# b vector   
    for point in data_set:
        row = []#
        row.append(point[0]**2)#ax^2
        row.append(point[0])#bx
        row.append(1)#c
        a.append(row)#matrix
        b.append(point[1])#y height(m)

    #print(data_set)
    
    A = Matrix(0,0,0,a)
    AT = A.transpose()
    ATA = AT*A
    B = Matrix(0,0,0,[b]).transpose()
    ATB = AT*B
    augmented_A = ATA.augment(ATB.transpose().data[0])
    solution, _ = augmented_A.row_echelon_form().back_substitution()
    print(f"y = {solution[0]:.3f}x^2 + {solution[1]:.3f}x + {solution[2]:.3f}")

if __name__ == '__main__':
    main()

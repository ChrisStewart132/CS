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

    def __repr__(self):
        return str(self)
    
    def __str__(self):
        output = ""
        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                output += f"{self.data[i][j]:.2f} "
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

    def back_substitution(self):
        """
        solves an augmented triangular matrix
        x1 x2 x3 : b
        1  2  3  : 8
        0  2  2  : 4
        0  0  2  : 1

        1(x1) + 2(x2) + 3(x3) = 8
                2(x2) + 2(x3) = 4
                        2(x3) = 1
        """
        m = self.data
        solution = [None for x in range(self.width()-1)]# [x1,x2...xn]
        y = self.height()-1
        for x in range(self.width()-2,-1,-1):
            pivot = m[y][x]
            rhs = m[y][-1] - sum([solution[j]*m[y][j] for j in range(x+1, self.width()-1)])# b - a1.x1 - a2.x2...
            solution[y] = rhs / m[y][x]
        
            y -= 1
            if y == -1:
                break

        return solution
                
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
    solution = [1,2,3]
    # testing solution is close in accuracy (as floating point errors occur)
    tests.append(sum([abs(solution[i] - x) for i, x in enumerate(m.row_echelon_form().back_substitution())]) < 0.1)
    
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
    solution = [1,2,3]
    # testing solution is close in accuracy (as floating point errors occur)
    tests.append(sum([abs(solution[i] - x) for i, x in enumerate(m.row_echelon_form().back_substitution())]) < ACCURACY*len(solution))
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



    print("Gaussian elimination")
    tests = []
  
    A = [
          [1,0,1,0],
          [0,1,1,0],
          [0,0,0,0],
          [0,0,0,1]
          ]
    m=Matrix(0,0,0,A)
    m_row_echelon = Matrix(0,0,0,
    [
          [1,0,1,0],
          [0,1,1,0],
          [0,0,0,1],
          [0,0,0,0]
          ])    
    tests.append(m.row_echelon_form() == m_row_echelon)
    tests.append(m.range() == 3)
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

    """
    variables x[iron, copper, tin, coal]
    2 1 1 2 = 4
    2 0 0 1 = 3
    4 2 2 4 = 8
    0 2 2 2 = 2

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
    print(" ",all(tests), tests)


if __name__ == '__main__':
    main()

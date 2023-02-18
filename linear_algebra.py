Accuracy = 0.0001

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
                        if delta > Accuracy:
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

    def row_echelon_form(self):
        """
        takes an augmented matrix ([a1 a2 a3 : b]) and returns a copy of it in row echelon form (triangle matrix)
        """
        m = [[self.data[i][j] for j in range(self.width())] for i in range(self.height())]
        for x in range(self.width()):
            # x column
            pivot = None
            for y in range(x, self.height()):
                # find the pivot
                if pivot == None and m[y][x] != 0:
                    pivot = y
                elif pivot!=None:# eliminate remaining rows using the pivot row
                    ratio = -m[y][x] / m[pivot][x]
                    for j in range(x, self.width()):
                        m[y][j] += m[pivot][j]*ratio
            
        # re-order rows to a triangular matrix
        free_vars = 0
        for x in range(self.width()):
            y = x - free_vars
            # find pivot row and swap it with current
            while y < self.height():
                if m[y][x] != 0:
                    m[y], m[x] = m[x], m[y]
                    break
                y+=1
            if y == self.height():
                free_vars += 1
            
        return Matrix(0,0,0,m)        

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
    
    solution = [1,2,3]
    # testing solution is close in accuracy (as floating point errors occur)
    tests.append(sum([abs(solution[i] - x) for i, x in enumerate(m.row_echelon_form().back_substitution())]) < Accuracy*len(solution))
    print(" ",all(tests), tests)

    

if __name__ == '__main__':
    main()

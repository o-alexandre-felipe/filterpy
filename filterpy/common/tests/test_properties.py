import sys, os, unittest;
import numpy as np;

# Since tests are meant to be used in development
# ensure it will not load a system wide version of the package
if __name__ == '__main__':
  sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../../../'))

from filterpy.common import properties;

class PropertyAccess(unittest.TestCase):
  class Class:
    prop1 = properties.ClassProperty('prop1')
    prop2 = properties.ClassProperty('prop2')
  def test_access(self):
    c = PropertyAccess.Class()
    with self.assertRaises(AttributeError):
      x = c.prop1
    c.prop1 = 'x'
    c.prop2 = 'y'
    # make sure each property is stored separately
    self.assertEqual(c.prop1, 'x')
    self.assertEqual(c.prop2, 'y')

  def test_multiple_instances(self):
    c = PropertyAccess.Class()
    d = PropertyAccess.Class()
    c.prop1 = 'x'
    d.prop1 = 'y'
    # make sure a given property is stored separately for
    # each instance
    self.assertEqual(c.prop1, 'x')
    self.assertEqual(d.prop1, 'y')

class PropertyValidation(unittest.TestCase):
  n = properties.ClassProperty('n')
  def test_validation(self):
    class ExpectEvenNumber(properties.Template):
      def __call__(self, v):
        assert (v % 2 == 0)
        return v
    self.n = ExpectEvenNumber()
    self.n = 4
    self.n = 8
    with self.assertRaises(Exception):
      self.n = 3

class AutomaticConversionOnAssignment(unittest.TestCase):
  n = properties.ClassProperty('n');
  def test_transformation(self):
    class EnsureEvenNumber(properties.Template):
      def __call__(self, v):
        return v - (v % 2)
    self.n = EnsureEvenNumber()
    self.n = 20;
    self.assertEqual(self.n, 20)
    self.n = 11
    self.assertEqual(self.n, 10)

class Test_1x1(unittest.TestCase):
  A = properties.ClassProperty('A')

    
  def test_assign_0D(self):
    self.A = properties.MatrixTemplate(1,1)
    self.A = 1234
    self.assertEqual(self.A.shape, (1,1))
    self.assertEqual(self.A[0][0], 1234)
  def test_assign_1D(self):
    self.A = properties.MatrixTemplate(1,1)
    # vector (1D) matrx
    self.A = [3256]
    self.assertEqual(self.A.shape, (1,1))
    self.assertEqual(self.A[0][0], 3256)
  def test_assign_2D(self):
    self.A = properties.MatrixTemplate(1,1)
    # matrix 1x1, 2D array-like
    self.A = [[2394876]]
    self.assertEqual(self.A.shape, (1,1))
    self.assertEqual(self.A[0][0], 2394876)
  def test_assign_3D(self):
    self.A = properties.MatrixTemplate(1,1)
    # 3D array-like
    self.A = [[[32786]]]
    self.assertEqual(self.A.shape, (1,1))
    self.assertEqual(self.A[0][0], 32786)

class TestVector(unittest.TestCase):

  A = properties.ClassProperty('A')
  def test_valid_multi_dimensional(self):
    for n in range(2, 5):
      x = list(range(n))
      for shape in [(1,n), (n, 1)]:
        self.A = properties.MatrixTemplate(*shape)
        for d1 in range(3):
          x = [xi for xi in x];  
          y = x
          for d2 in range(3):
            y = [y]
            self.A = y
            self.assertEqual(self.A.shape, shape);
  
  def test_invalid_1D(self):
    for n in range(2, 5):
      for x in [list(range(1, n+2)), list(range(1, n))]:
        for shape in [(1,n), (n, 1)]:
          ### CHANGE THE TEMPLATE HERE ###
          self.A = properties.MatrixTemplate(*shape)
          for d1 in range(3):
            x = [xi for xi in x];  
            y = x
            for d2 in range(3):
              y = [y]
              with self.assertRaises(ValueError):
                self.A = y
  
  def test_invalid_interpretations_from_2D(self):
    m,n=3,5
    for shape in [(m*n,1), (1,m*n)]:
      self.A = properties.MatrixTemplate(*shape)
      with self.assertRaises(ValueError):
        # Number of elements of the whole matrix match the size of the vector
        self.A = np.zeros((m,n))
      with self.assertRaises(ValueError):
        # Columns of the matrix match the vector shape, but not the matrix
        self.A = np.zeros((m*n, 2))
      with self.assertRaises(ValueError):
        # Rows of the matrix match the vector shape, but not the matrix
        self.A = np.zeros((2, m*n))

class TestMatrix(unittest.TestCase):

  A = properties.ClassProperty('A')
  def test_invalid(self):
    m,n = 3,5
    self.A = properties.MatrixTemplate(m,n)
    
    with self.assertRaises(ValueError):
      # A 1D vector whose number of elements is the same as that of the matrix
      self.A = np.zeros((m*n))
    with self.assertRaises(ValueError):
      # Transposed will not be matched automatically of both dimensions are greater than 1
      self.A = np.zeros((n,m))
    
    for shape in [(n,), (m,), (m,1), (1,m), (n,1), (1,n)]:
      with self.assertRaises(ValueError):
        self.A = np.zeros(shape)
    
    self.A = np.zeros((m,n))

  def test_multiplicative(self):
    self.A = properties.MatrixTemplate(4,4,multiplicative=True)
    self.A = 4
    self.assertTrue(np.allclose(self.A, 4 * np.eye(4)))
    with self.assertRaises(ValueError):
      self.A = np.ones(4)
    
    self.A = properties.MatrixTemplate(4,4,multiplicative=False)
    with self.assertRaises(ValueError):
      self.A = 4
    with self.assertRaises(ValueError):
      self.A = np.ones(4)

class TestMatrixFunctionProperty(unittest.TestCase):
  F = properties.ClassProperty('F')

  def test_decorated_function(self):
    for shape in [(3,5), (7,4)]:
      @properties.MatrixFunction(*shape)
      def f(x):
        return np.zeros(x)
      
      # check that a matrix was returned
      self.assertEqual(f(shape).shape, shape)

      with self.assertRaises(ValueError):
        f(shape[::-1]) # Tell F to return a matrix with wrog shape

  def test_decorated_function_assignment(self):
    shape = (3,7)
    @properties.MatrixFunction(*shape)
    def f():
      raise TypeError("This should not be called")

    self.F = properties.MatrixFunctionTemplate(*shape)
    self.F = f # matches the template

    self.F = properties.MatrixFunctionTemplate(*shape[::-1])
    with self.assertRaises(ValueError):
      self.F = f # does not match the template

  def test_undecorated_function_assignment(self):
    shape = (3,7)
    def f():
      return 1
    self.F = properties.MatrixFunctionTemplate(*shape)

    # since f was not decorated it will accept expecting it
    # to return matrices of the given shape
    self.F = f

    with self.assertRaises(ValueError):
      self.F() # when f is called it will see an invalid value

  def test_multiplicativify_of_function(self):
    # Non multiplicative template
    self.F = properties.MatrixFunctionTemplate(4,4,multiplicative=False)
    def identity(x): return x
    # Undecorated function
    self.F = identity
    with self.assertRaises(ValueError):
      self.F(4)
    self.assertTrue(np.allclose(self.F(4*np.eye(4)), 4*np.eye(4)))

    # Non multiplicative template multiplicative function
    self.F = properties.DecoratedMatrixFunction((4,4), identity, multiplicative=True)
    self.assertTrue(np.allclose(self.F(4*np.eye(4)), 4*np.eye(4)))

    # Multiplicative template
    self.F = properties.MatrixFunctionTemplate(4,4,multiplicative=False)
    # Multiplicative template undecorated function
    self.F = identity
    self.assertTrue(np.allclose(self.F(4*np.eye(4)), 4*np.eye(4)))

    # Multiplicative template multiplicative function
    self.F = properties.DecoratedMatrixFunction((4,4), identity, multiplicative=True)
    self.assertTrue(np.allclose(self.F(4*np.eye(4)), 4*np.eye(4)))

    # Multiplicative template non-multiplicative function
    self.F = properties.DecoratedMatrixFunction((4,4), identity, multiplicative=False)
    with self.assertRaises(ValueError):
      self.F(6)

class TestZero(unittest.TestCase):
  A = properties.ClassProperty('A')
  F = properties.ClassProperty('F')

  def test_zero_matrix(self):
    for shape in [(1,4), (5, 1), (3,6), (7,3)]:
      self.A = properties.MatrixTemplate(*shape)
      self.A = 0
      self.assertTrue(np.all(self.A == 0))
      self.assertEqual(self.A.shape, shape)

  def test_zero_matrix_function(self):
    def zero():
      return 0
    for shape in [(1,4), (5, 1), (3,6), (7,3)]:
      self.F = properties.MatrixFunctionTemplate(*shape)
      self.F = zero
      self.assertTrue(np.all(self.F() == 0))
      self.assertEqual(self.F().shape, shape)

if __name__ == '__main__':
  unittest.main(verbosity=1);
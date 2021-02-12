import numpy as np

def randomization(n):
    """
    Arg:
      n - an integer
    Returns:
      A - a randomly-generated nx1 Numpy array.
    """
    return np.random.rand(n,1)
    #Your code here
    raise NotImplementedError

def operations(h, w):
    """
    Takes two inputs, h and w, and makes two Numpy arrays A and B of size
    h x w, and returns A, B, and s, the sum of A and B.

    Arg:
      h - an integer describing the height of A and B
      w - an integer describing the width of A and B
    Returns (in this order):
      A - a randomly-generated h x w Numpy array.
      B - a randomly-generated h x w Numpy array.
      s - the sum of A and B.
    """
    #Your code here
    A = np.random.rand(h,w)
    B = np.random.rand(h,w)
    S=A+B
    return A,B,S
    raise NotImplementedError


def norm(A, B):
    """
    Takes two Numpy column arrays, A and B, and returns the L2 norm of their
    sum.

    Arg:
      A - a Numpy array
      B - a Numpy array
    Returns:
      s - the L2 norm of A+B.
    """
    #Your code here
    return np.linalg.norm(A+B)
    raise NotImplementedError


def neural_network(inputs, weights):
    """
     Takes an input vector and runs it through a 1-layer neural network
     with a given weight matrix and returns the output.

     Arg:
       inputs - 2 x 1 NumPy array
       weights - 2 x 1 NumPy array
     Returns (in this order):
       out - a 1 x 1 NumPy array, representing the output of the neural network
    """
    #Your code here
    raise NotImplementedError

def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.
    """
    #Your code here
    raise NotImplementedError

def vector_function(x, y):
    """
    Make sure vector_function can deal with vector input x,y 
    """
    #Your code here
    raise NotImplementedError

##########################################################
#
#                         TESTS
#
##########################################################

def green(s):
    return '\033[1;32m%s\033[m' % s


def yellow(s):
    return '\033[1;33m%s\033[m' % s


def red(s):
    return '\033[1;31m%s\033[m' % s

def log(*m):
    print(" ".join(map(str, m)))

def test_randomization():
  n = 5 # input integer
  A = randomization(n) # output matrix
  if A.shape[0] == 5:
    if A.shape[1] == 1:
      log(green("PASS"),"randomization()")
    else:
      log(red("FAIL"),f"got shape: {A.shape}, expecting (n,1)")
  else:
    log(red("FAIL"),f"got length {A.shape}, expecting {n}")

def test_operations():
  h = 5 # first input integer
  w = 5 # second input integer
 
  A,B,S = operations(h, w)

  if np.array_equal(A,B):
    log(red("FAIL"),f"A and B are the same - not random: A:{A} \n B:{B}")
    return 

  if not np.array_equal(S,A+B):
    log(red("FAIL"),f"S is not equal to sum of A and B")
    return 

  if A.shape[0] != h:
    log(red("FAIL"),f"got A of length {A.shape}, expecting {h}")
    return 
  elif A.shape[1] != w:
    log(red("FAIL"),f"got A of shape: {A.shape}, expecting ({h},{w})")
    return 

  if B.shape[0] != h:
    log(red("FAIL"),f"got B of length {B.shape}, expecting {h}")
    return 
  elif B.shape[1] != w:
    log(red("FAIL"),f"got B of shape: {B.shape}, expecting ({h},{w})")
    return 
 
  log(green("PASS"),f"operations()")
  
def test_norm():
  n = 5 # input integer
  A = [1,2,3,4,5,6]
  B = [1,2,3,4,5,6]
  S = A+B #A+B 
  norm_check = 0
  for i in S:
    norm_check += i**2
  norm_check=norm_check**(1/2)

  n = norm(A,B)
  if norm(A,B) == norm_check: # output matrix
    log(green("PASS"),"norm()")
  else:
    log(red("FAIL"),f"got norm: {n}, expecting: {norm_check}")
  

def runTest():
  test_randomization()
  test_operations()
  test_norm()
  pass

##########################################################
#
#                         MAIN
#
##########################################################

if __name__=="__main__":
  runTest()
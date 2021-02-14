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
    return np.tanh(np.dot(weights.T,inputs))
    raise NotImplementedError

def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.
    """
    #Your code here
    if(x<=y):
      return x*y
    else:
      if y==0:
        raise NotImplementedError
      return x/y
    raise NotImplementedError

def vector_function(x, y):
    """
    Make sure vector_function can deal with vector input x,y 
    """
    #Your code here
    return np.where(x <= y,x*y,x/y)
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
  
def test_neural_network():
  n = 5 # input integer
  I = np.array([[1],[2]]) # Inputs
  W = np.array([[1],[2]]) # Weights

  nn = neural_network(I,W)
  if nn.shape[0] != 1: # output matrix
    log(red("FAIL"),f"got first shape for nn: {nn.shape[0]}, expecting: {1}") 
    return
  elif nn.shape[1] != 1:
    log(red("FAIL"),f"got second shape for nn: {nn.shape[1]}, expecting: {1}")  
    return
  
  log(green("PASS"),"neural_network()")

def test_scalar_function():
  # test 1 - feature division
  x = 10
  y = 9
  output = scalar_function(x,y)
  if(x < output): # x greater than y, so x should be smaller than output to pass
    log(red("FAIL"),"Div: x is greater than output")
    return 

  # test 1 - feature multiplication
  x = 4
  y = 9
  output = scalar_function(x,y)
  if(x > output): # x smaller than y, so x should be greater than output to pass
    log(red("FAIL"),"Mult: x is smaller than output")
    return 

  log(green("PASS"),"scalar_function()")  


def test_vector_function():
  # test 1 - feature division
  x = np.array([1,2,3,4,5,6,7,8])
  y = np.array([8,7,6,5,4,3,2,1])
  test_result = np.array([True]*x.shape[0])
  # output = np.array([1,1,5,5,5,5,1,1])
  output = vector_function(x,y)
  # if(x<y): # x greater than y, so x should be smaller than output to pass
  # print(x,y,x<y)
  
  for ix,jy,ot in zip(x,y,output):
    if(ix<=jy): # check if it is mul or div case 
      if(ot < ix): # check if result is correct
        log(red("FAIL"),f"MUL: output result is smaller than input")  
     
    elif(ix>jy):
      if(ot > ix): # check if result is correct
        log(red("FAIL"),f"DIV: output result is greater than input")  

  log(green("PASS"),"vector_function()"))

def runTest():
  test_randomization()
  test_operations()
  test_norm()
  test_neural_network()
  test_scalar_function()
  test_vector_function()
  pass

##########################################################
#
#                         MAIN
#
##########################################################

if __name__=="__main__":
  runTest()
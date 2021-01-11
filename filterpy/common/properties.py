"""

This module implements validations for matrices to facilitate the 
validation of matrix properties.

This is licensed under an MIT license. See the readme.MD file
for more information.

Copyright 2020 Alexandre Felipe.

"""


import numpy as np;

class Template:
  '''
    This class provides a mechanism to set validators to class parameters as 
    an assignment to the property. The decision to implement validators this
    way comes from the fact that, as far as I know, python cannot set 
    descriptors per instance.
  '''
  def __init__(self):
    self.name = 'template' # the name of the associated property
  def __call__(self, v):
    return v

class ClassProperty:
  '''
    Implements the property descriptor with custom validations.

    https://docs.python.org/2.7/howto/descriptor.html
    https://docs.python.org/3.9/howto/descriptor.html

  '''

  # Make name argument optional and 
  # stop using it when support for python<3.6
  # is dropped.
  def __init__(self, name):
    self.name = name
    self.template = Template()

  def __set_name__(self, owner, name):
    '''
    For python > 3.6, the name will be taken from the name of the property
    in the containing class 
    https://docs.python.org/3.9/reference/datamodel.html#object.__set_name__
    '''
    self.name = name
  def __get__(self, obj, objtype=None):
    try:
      return obj.__dict__[self.name + '#value']
    except KeyError:
      raise AttributeError('Property ' + self.name + ' is not defined yet');
  def __set__(self, obj, v):
    if isinstance(v, Template):
      obj.__dict__[self.name + '#template'] = v
      v.name = self.name
    else:
      template = None
      try:
        template = obj.__dict__[self.name + '#template']
      except KeyError:
        pass
      if callable(template):
        v = template(v)
      obj.__dict__[self.name + '#value'] = v
def as_matrix(shape, v, name='matrix'):
  '''
    Ensures that v is a matrix of the given shape. Axes of length 1 may be 
    added or removed to match the given shape
  '''
  M,N = shape
  m = np.squeeze(v)

  if (M,N) == m.shape:
    return m
  elif m.ndim <= 1 and (M == 1 or N == 1):
    return m.reshape((M, N))
  elif m.ndim == 0 and M == N:
    return m * np.eye(M)
  else:
    raise ValueError("Expected {} to be of shape {}, value has shape {}"
      .format(name, shape, m.shape))

class MatrixTemplate(Template):
  '''
    A template that makes sure that the value of the target property
    is a matrix of the given shape.
  '''
  def __init__(self, m, n):
    self.shape = (m, n)
  def __call__(self, v):
    return as_matrix(self.shape, v, self.name)

class DecoratedMatrixFunction:
  '''
    A template that makes sure that a function will return a matrix
    of the given shape.
  '''
  def __init__(self, shape, f):
    if not callable(f):
      raise ValueError('f must be callable')
    self.shape = shape
    self.f = f
  def __call__(self, *args, **kwargs):
    return as_matrix(self.shape, self.f(*args, **kwargs), 'return value')

class MatrixFunctionTemplate(Template):
  def __init__(self, m, n):
    self.shape = (m,n)
  def __call__(self, f):
    if isinstance(f, DecoratedMatrixFunction):
      if f.shape != self.shape:
        raise ValueError(
          'The return value of {} don\'t properly in a {} x {} matrix',
          self.name, *f.shape)
      return f
    elif not callable(f):
      raise ValueError('{} must be a function', self.name)
    else:
      # The function was not decorated we may decorate it now and hope
      # that it will return a value of the expected shape, if not
      # an error will be reported when the value is seen.
      return DecoratedMatrixFunction(self.shape, f)

class MatrixFunction:
  def __init__(self, m, n):
    self.shape = (m,n)
  def __call__(self, f):
    return DecoratedMatrixFunction(self.shape, f)


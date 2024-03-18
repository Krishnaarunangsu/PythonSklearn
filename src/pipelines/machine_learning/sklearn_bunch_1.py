# class sklearn.utils.Bunch(**kwargs)[source]
# Container object exposing keys as attributes.

# Bunch objects are sometimes used as an output for functions and methods.
# They extend dictionaries by enabling values to be accessed by key,
# bunch["value_key"], or by an attribute, bunch.value_key.

from sklearn.utils import Bunch

b=Bunch(a=1, b=2)
print(b['b'])
print(b.b)
b.a=3
print(b['a'])
b.c=6
print(b['c'])
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
# Function: main()
#------------------------------------------------------------------------------
def L02_main():

    # parameters
    N = 10
    
    # loop “for”
    for n in range(0, N):
      print("%d" % (n))
    
    # loop “while”
    n = 0
    while n < N + 1: # do while condition is true
      print("%d" % (n))
      n = n + 1
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L02_main()
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Function, add_1()
#------------------------------------------------------------------------------
def add_1(a, b):

    # division
    a = b + 1
    b = a + 1
    
    return a, b

#------------------------------------------------------------------------------
# Function, add_2()
#------------------------------------------------------------------------------
def add_2(a, b):

    # division
    a = a + 2
    b = b + 2
    
    return a, b

#------------------------------------------------------------------------------
# Function, P1_matlab_debugger
# Description,
#   - set a "Breakpoint" (to stop the execution)
#   - use button "Continue" (to continue the execution until next breakpoint)
#   - use button "Step" (to go to the next line)
#   - use button "Quit Debugging" (to quit debugging)
#   - use button "Run to Cursor" (to continue the execution until cursor position)
#   - use button "Step in" (go to the function)
#------------------------------------------------------------------------------
def L02_main():

    # init
    a = 1
    b = 0
    
    # check
    # print('a = %d, b = %d' % (a, b))
    
    # call function
    a, b = add_1(a, b)
    a, b = add_2(a, b)
    
    # result
    print('a = %d, b = %d' % (a, b))
            
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L02_main()

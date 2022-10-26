# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Function: add()
#------------------------------------------------------------------------------
def add(nStudents, n):
    
    nStudents = nStudents + n
    return nStudents

#------------------------------------------------------------------------------
# Description
#  Print message about number of students
#  Parameters
#    nStudents / number of students
#------------------------------------------------------------------------------
def L02_main():

    nStudents = 15
    
    # print message
    print('There are %d students in the class before add()' % (nStudents))
    
    # call sub-function
    n = 5
    nStudents = add(nStudents, n)
    
    # print message
    print('There are %d students in the class after add()' % (nStudents))

#------------------------------------------------------------------------------
if __name__ == '__main__':
    L02_main()

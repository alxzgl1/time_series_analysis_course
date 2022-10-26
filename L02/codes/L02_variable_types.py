# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Function: main()
#------------------------------------------------------------------------------
def L02_main():

    # types
    number = 10.25                    # number
    string = 'lecture_2'              # string
    data = [1, 0.25, [1, 2], 'types'] # list 
    
    # print
    print("number: %1.2f" % (number))
    print("string: %s" % (string))
    print("list item [0]: %d" % (data[0]))
    print("list item [1]: %f" % (data[1]))
    print("list item [2]: %s" % (data[2]))
    print("list item [3]: %s" % (data[3]))

#------------------------------------------------------------------------------
if __name__ == '__main__':
    L02_main()

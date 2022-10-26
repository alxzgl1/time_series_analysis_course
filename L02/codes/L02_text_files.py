# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Function: main()
#------------------------------------------------------------------------------
def L02_main():

    # text to write
    aLine1 = 'Line 1\n' 
    aLine2 = 'Line 2\n'
    
    # WRITE TO FILE
    
    # open file to write
    hFile = open('output.txt', 'w')
    # write text to file
    hFile.write(aLine1)
    # add second line
    hFile.write(aLine2)
    # close file
    hFile.close()
    
    # READ FROM FILE
    # open file to read
    hFile = open('output.txt', 'r')
    # read text from file
    aLines = hFile.read()
    # print file content
    print(aLines)
    # close file
    hFile.close()
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L02_main()

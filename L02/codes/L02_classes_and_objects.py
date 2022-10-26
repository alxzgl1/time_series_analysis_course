# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Class: Student
#------------------------------------------------------------------------------
class Student():
    
    # init
    def __init__(self, firstname, surname, ID):
        self.firstname = firstname
        self.surname = surname
        self.ID = ID
        self.grade = 0

    # print_info
    def print_info(self):
        print("%s %s, ID: %d, grade: %d" % (self.firstname, self.surname, \
                                            self.ID, self.grade))
        
    # get_ID
    def get_ID(self):
        return self.ID
    
    # set_grade
    def set_grade(self, grade):
        self.grade = grade

#------------------------------------------------------------------------------
# Function: main()
#------------------------------------------------------------------------------
def L02_main():

    # create object
    tStudentA = Student('Muhammad', 'Lee', 123456)
    tStudentB = Student(firstname='Lena', surname='Lena', ID=234567)
    tStudentC = Student(ID=345678, surname='Chang', firstname='Max')
    
    # list of objects
    tStudentList = [tStudentA, tStudentB, tStudentC]
    
    # print info
    tStudentList[1].print_info()
    
    # set grade
    tStudentList[1].set_grade(5)
    
    # print info
    tStudentList[1].print_info()
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L02_main()

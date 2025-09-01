import sys

VERSION_NUMBER = "2.5.1"

def addSwigInterfacePath(version=3):
    if version == 2:
        sys.path.insert(0, '../interfaces/Python2')
    else:
        sys.path.insert(0, '../interfaces/Python')

def getDataDirPath():
    return "../tests/data/"

import numpy as np
import os
import gzip
import csv
from shutil import move

__fileType = 534587
__bytesCount = 4
dt = np.dtype(np.int32).newbyteorder('B')

def readCsv(path):
    data = []
    if not os.path.isfile(path):
        return data
    with open(path, 'rt') as csvfile:        
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            data.append(row)
    return data

def saveToFile(path, array, useTempFile = False):
    if not useTempFile:
        __save(path, array)
    else:
        p = path + '.tmp'
        __save(p, array)
        move(p, path)

def __save(path, array):
    array = np.array(array, dtype=dt)
    shape = np.array(array.shape, dtype=dt)
    with gzip.open(path, 'wb') as f:
        f.write(__fileType.to_bytes(__bytesCount, 'big'))
        f.write(len(shape).to_bytes(__bytesCount, 'big'))
        f.write(shape.tobytes())
        f.write(array.tobytes())

def readFile(path):
    if not os.path.isfile(path):
        return None
    
    with gzip.open(path, 'rb') as bytestream:
        fileType = __readInt(bytestream)
        if fileType != __fileType:
            raise ValueError("Invalid file type")
        shape = __readShape(bytestream)
        buferLen = __getBuferLen(shape)
        bufer = bytestream.read(__bytesCount * buferLen)
        data = np.frombuffer(bufer, dtype=dt)
        data = data.reshape(shape)
    return data

def __readInt(bytestream):
  return np.frombuffer(bytestream.read(__bytesCount), dtype=dt)[0]

def __getBuferLen(shape):
    len = 1
    for s in shape:
        len = len * s
    return len

def __readShape(bytestream):
    shape = []
    shapeLen = __readInt(bytestream)
    for i in range(shapeLen):
        shape.append(__readInt(bytestream))
    return shape

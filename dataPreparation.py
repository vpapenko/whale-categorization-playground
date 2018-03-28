import csv
import os.path
import numpy as np
import tensorflow as tf
from commonData import commonData
import fileOperations as fo

def createTypesMapping():
    if not os.path.isdir(cd.convertedDataPath):
        os.makedirs(cd.convertedDataPath)

    if os.path.isfile(cd.typesMappingPath):
        return
    
    typesMapping = []
    with open(cd.trainPath, 'rt') as csvTrain, open(cd.typesMappingPath, 'w', newline='') as csvTypesMapping:
        reader = csv.reader(csvTrain)
        writer = csv.writer(csvTypesMapping)
        writer.writerow(['number','Id'])
        next(reader)
        line = 0
        for row in reader:
            if not any(row[1] in s for s in typesMapping) and row[1] != cd.newWhaleName:
                newId = [line, row[1]]
                typesMapping.append(newId)
                writer.writerow(newId)
                line += 1
        newId = [line, cd.newWhaleName]
        typesMapping.append(newId)
        writer.writerow(newId)

def createLabels():
    if os.path.isfile(cd.labelsPath):
        return
    
    typesMapping = fo.readCsv(cd.typesMappingPath)
    labels=[]
    with open(cd.trainPath, 'rt') as csvTrain:
        reader = csv.reader(csvTrain)
        next(reader)
        for row in reader:
            id = [x[0] for x in typesMapping if x[1] == row[1]]
            labels.append(id[0])
    fo.saveToFile(cd.labelsPath, labels)

def createData(saveResizedImage):

    rowCount = 0
    data = fo.readFile(cd.convertedImageDataPath)
    if data is None:
        line = 0
    else:
        line = np.array(data).shape[0]

    with open(cd.trainPath, 'rt') as csvTrain:
        reader = csv.reader(csvTrain)
        rowCount = sum(1 for row in reader)
    
    with open(cd.trainPath, 'rt') as csvTrain:
        reader = csv.reader(csvTrain)
        next(reader)        
        for i in range(line):
            next(reader)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            saveCounter = 0
            for row in reader:
                line += 1
                image = getImage(row[0], cd.currentImageSize, saveResizedImage)
                
                if data is None:
                    data = np.array([image.tolist()])
                else:
                    data = np.concatenate((data, [image.tolist()]))
                
                if saveResizedImage:
                    i = tf.image.encode_jpeg(image, format = 'rgb', quality = 100)
                    sess.run(tf.write_file(cd.convertedImagePath + row[0], i))
                
                print(str(line) + ' of ' + str(rowCount))

                saveCounter += 1
                if saveCounter == 50:
                    print('save')
                    fo.saveToFile(cd.convertedImageDataPath, data, True)
                    saveCounter = 0

def getImage(fileName, targetSize, saveResizedImage):
    image_contents = tf.read_file(cd.trainImagesPath + fileName)
    image = tf.image.decode_image(image_contents, channels=3)
    x = image.eval().shape[0]
    y = image.eval().shape[1]
    size = x if x > y else y
    image = tf.image.resize_image_with_crop_or_pad(image, size, size)
    image = tf.image.resize_images(image, [targetSize, targetSize])
    image = tf.cast(image, tf.uint8)
    return image.eval()

cd = commonData(commonData.modes[1])
createTypesMapping()
createLabels()
createData(True)

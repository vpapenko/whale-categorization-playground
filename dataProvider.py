import fileOperations as fo
import numpy as np

class dataProvider:
    imagesPath = ''
    labelsPath = ''
    labelsMappingPath = ''
    labelsCount = 0
    labels = []

    def __init__(self, imagesPath, labelsPath, labelsMappingPath):
        self.imagesPath = imagesPath
        self.labelsPath = labelsPath
        self.__readTypes(labelsMappingPath)

    def __readTypes(self, labelsMappingPath):
        self.types = fo.readCsv(labelsMappingPath)
        self.typesCount = len(self.types)

    def getData(self):
        result = {}
        result["Images"] = fo.readFile(self.imagesPath).astype(np.float32)
        result["Labels"] = (fo.readFile(self.labelsPath)[:result["Images"].shape[0]]).astype(np.int32)
        return result
        
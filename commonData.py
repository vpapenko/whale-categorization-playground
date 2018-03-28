class commonData:
    modes = ['Micro', 'Small', 'Large']
    currentMode = ''
    imageSizes = {modes[0]:100, modes[1]:500, modes[2]:1000}
    currentImageSize = -1
    newWhaleName = 'new_whale'
    dataPath = './data/'
    trainPath = dataPath + '/train.csv'
    trainImagesPath = dataPath + 'train/'
    convertedDataPath = dataPath
    convertedImagePath = ''
    convertedImageDataPath = ''
    typesMappingPath = convertedDataPath + 'typesMapping.csv'
    labelsPath = convertedDataPath + 'labels.gz'

    def __init__(self, mode):
        self.currentMode = mode
        self.convertedImagePath = self.convertedDataPath + mode + '/'
        self.convertedImageDataPath = self.convertedDataPath + 'image.' + mode + '.gz'
        self.currentImageSize = self.imageSizes[mode]


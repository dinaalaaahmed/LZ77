import math
import cv2
import numpy as np
import os

# read inputs
print('Enter image name')
image_name = str(input())

print('Enter block size')
blockSize = int(input())


print('Enter Type (float32,16,..)')
type_of_float = str(input())

# inputs validation
if os.path.exists(image_name) == 0:
    print('you entered an image that does not exist')
    image_name = 'gray.png'

if blockSize < 0:
    blockSize = 2

if type_of_float == ('float16'or'float32'or'float64'):
    type_of_float = type_of_float
else:
    type_of_float = 'float64'

# read image as gray
img = cv2.imread(image_name, 0)

# flatten the image
flattenImage = img.flatten()

# Encoding Part
probability = {}

# calculate probability of the image
for x in flattenImage:
    if x in probability:
        probability[x] += 1
    else:
        probability[x] = 1

for x in probability:
    probability[x] = probability[x]/len(flattenImage)

# calculate the upper and lower ranges of the image
start = {}
end = {}
startD = 0

for x in probability:
    start[x] = startD
    end[x] = start[x] + probability[x]
    startD = end[x]

# encode the image
dictionary = []


def encode_arithmetic(levels):
    lower = 0
    upper = 1
    for code in levels:
        range = upper - lower
        low = lower
        lower = low + start[code] * range
        upper = low + end[code] * range
    dictionary.append(lower + (upper-lower)/2.0)


def encode_array(array):
    i = 0
    for x in range(math.floor(len(array)/blockSize)):
        arr = []
        for y in range(blockSize):
            arr.append(array[i])
            i = i+1
        encode_arithmetic(arr)


encode_array(flattenImage)
# save the encoded image as numpy array
Dictionary = np.array(dictionary, dtype=type_of_float)
np.save('dictionary.npy', Dictionary)
np.save('probability.npy', probability)

# Decoding Part
width = img.shape[0]
height = img.shape[1]

# read the encoded image
if os.path.exists('dictionary.npy'):
    dictionary = np.load('dictionary.npy', allow_pickle=True)

decodingDictionary = []


# decode the image
def decode_arithmetic(arr):

    for code in arr:
        upper = 1
        lower = 0
        for y in range(blockSize):

            for c in probability:
                d_lower = lower
                rang = upper - lower
                lower = d_lower + start[c] * rang
                upper = d_lower + end[c] * rang
                if lower <= code <= upper:
                    decodingDictionary.append(c)
                    break
                else:
                    lower = d_lower
                    upper = rang + lower


decode_arithmetic(dictionary)

while len(decodingDictionary) < height*width:
    decodingDictionary.append(decodingDictionary[0])

# reshape decoded image and save it
image = np.reshape(decodingDictionary, (height, width))
cv2.imwrite("decoded"+image_name, image)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

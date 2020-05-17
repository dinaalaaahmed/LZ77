import numpy as np
import os
import cv2

print('Enter height')
y = int(input())
print('Enter wedth')
x = int(input())

arr=[]
# read the encoded image
if os.path.exists('encode.npy'):
    arr = np.load('encode.npy', allow_pickle=True)

decoded_text = []

def decode(array):

    for y in range(len(array[0])):
        length = len(decoded_text)-int(array[0][y])
        for i in range(int(array[1][y])):
            decoded_text.append(decoded_text[length])
            length += 1
        decoded_text.append(array[2][y])


decode(arr)


image = np.reshape(decoded_text, (x, y))
cv2.imwrite('image.png', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
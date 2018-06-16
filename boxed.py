import cv2
import skimage.io
import numpy

image = skimage.io.imread("./test.png")
L = [[1, 3], [1, 100], [100, 3], [100, 100]]
ctr = numpy.array(L).reshape((-1,1,2)).astype(numpy.int32)
print(ctr)
cv2.drawContours(image, [ctr], -1, (0, 0, 255), thickness=1)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
import sys
from sklearn.decomposition import RandomizedPCA
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt

filename = ['image2.png','image3.png','image5.png']

for i in range(0,3):
    img = mpimg.imread('input/' + filename[i])
    print(img.shape)
    plt.axis('off')
    plt.imshow(img)

    img_r = np.reshape(img, (img.shape[0], img.shape[1] * img.shape[2]))
    print(img_r.shape)

    ipca = RandomizedPCA(int(sys.argv[1])).fit(img_r)
    img_c = ipca.transform(img_r)
    print(img_c.shape)
    print(np.sum(ipca.explained_variance_ratio_))

    temp = ipca.inverse_transform(img_c)
    print(temp.shape)

    temp = np.reshape(temp, img.shape)

    print(temp.shape)

    plt.axis('off')
    plt.imshow(temp)
    mpimg.imsave('compressedImages/'+ filename[i],temp)
    plt.show()
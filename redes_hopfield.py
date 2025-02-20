import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#img handle functions
def load_img(path, side):
    img = Image.open(path)
    img = img.resize((side, side))
    img = img.convert('1')
    img = 2 * np.array(img, int) -1
    return img.flatten()

def show_array(img_array):
    side = int(np.sqrt(img_array.shape[0]))
    img_array = img_array.reshape((side, side))
    plt.figure(figsize=(3, 3))
    plt.imshow(img_array)
    plt.axis('off')
    plt.show()

def show_multiple_arraysi(img_arrays):
    fig = plt.figure(figsize=(3, 3))
    for i in range(len(img_arrays)):
        side = int(np.sqrt(img_arrays[i].shape[0]))
        plt.subplot(1, len(img_arrays), i+1)
        plt.imshow(img_arrays[i].reshape((side, side)))
        plt.axis('off')
    plt.show()

def show_matrix(img_matrix):
    plt.figure(figsize=(3, 3))
    plt.imshow(img_matrix)
    plt.axis('off')
    plt.show()

def negative_array(img_array):
    return -1 * img_array

def modify_img(n, img):
    #make 1/2% of image negative
    for i in range(n):
        if i > n/2-1:
            img[i] = -1
    return img

#hopfield net equations
#weigths matrix
def calculate_w(img):
    '''
    w = np.zeros((n,n))
    for i in range(n):
      for j in range(n):
        if i!=j:
          w[i,j]=img[i]*img[j]
    '''
    return np.outer(img, img) # matrix product

#reconstruct image
def reconstructed_image(n, w, state):
    for i in range(n):
        print(i)
        sum = 0
        for j in range(n):
            sum += w[i,j]*state[j]
        state[i] = 1 if sum>0 else -1
    return state
    #return np.dot(w, state)

#one pattern memory
side = 110
n = side * side # number of neurons

img = Image.open('111.jpg')

#image to memorize
img = load_img('111.jpg', side)
show_array(img)

#calc weights
w = calculate_w(img)

#random state
state = np.random.choice([-1,1], size=n)
show_array(state)

#reconstruct image
state = reconstructed_image(n, w, state)

#show_array(negative_array(state))
show_array(state)

#multile patterns memory
side = 50
n = side * side #number of neurons(pixels)


#memory images
imgs=[]
for i in range(1,6): #memorize images from 1 to x-1
    imgs.append(load_img(f'p{i}.jpeg', side))

print('memorized images:')
show_multiple_arraysi(imgs)

#weights matrix
w = np.zeros((n,n))
for p in range(len(imgs)):
    w+=calculate_w(imgs[p])

#set inital state
state = modify_img(n, load_img('p2.jpeg', side)) #modified image
#state = np.random.choice([-1,1], size=n) #random pixels
print('init state:')
show_array(state)

#reconstruct image
state = reconstructed_image(n, w, state)
print('reconstructed image:')
show_array(state)

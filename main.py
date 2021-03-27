import numpy as np
import matplotlib.pyplot as plt
import random

# A function to plot images
def show_image(img):
    image = img.reshape((28, 28))
    plt.imshow(image, 'gray')

#our activation function
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

#calculate output of nn
def calculate_output(input_neurons,weights,biases):
    layer1_neurons = sigmoid(weights[0]*input_neurons+biases[0])
    layer2_neurons = sigmoid(weights[1]*layer1_neurons+biases[1])
    output_neurons = sigmoid(weights[2]*layer2_neurons+biases[2])
    return np.where(output_neurons==np.amax(output_neurons))[0][0]

def checkaccuracy(train_set,weights,biases,count):
    correct = 0
    wrong = 0
    for i in range(100):
        answer=calculate_output(train_set[i][0],weights,biases)
        if(answer==np.where(train_set[i][1]==1)[0][0]):
            correct+=1
        else:
            wrong+=1
    print(correct/count)


def train_with_SGD(learning_rate,number_of_epoches,batch_size,train_set,weights,biases):
    for ep in range(number_of_epoches):
        random.shuffle(train_set)


# Reading The Train Set
train_images_file = open('train-images.idx3-ubyte', 'rb')
train_images_file.seek(4)
num_of_train_images = int.from_bytes(train_images_file.read(4), 'big')
train_images_file.seek(16)

train_labels_file = open('train-labels.idx1-ubyte', 'rb')
train_labels_file.seek(8)

train_set = []
num_of_train_images=100
for n in range(num_of_train_images):
    image = np.zeros((784, 1))
    for i in range(784):
        image[i, 0] = int.from_bytes(train_images_file.read(1), 'big') / 256
    
    label_value = int.from_bytes(train_labels_file.read(1), 'big')
    label = np.zeros(10)
    label[label_value] = 1
    
    train_set.append((image, label))


# # Reading The Test Set
# test_images_file = open('t10k-images.idx3-ubyte', 'rb')
# test_images_file.seek(4)

# test_labels_file = open('t10k-labels.idx1-ubyte', 'rb')
# test_labels_file.seek(8)

# num_of_test_images = int.from_bytes(test_images_file.read(4), 'big')
# test_images_file.seek(16)

# test_set = []
# for n in range(num_of_test_images):
#     image = np.zeros((784, 1))
#     for i in range(784):
#         image[i] = int.from_bytes(test_images_file.read(1), 'big') / 256
    
#     label_value = int.from_bytes(test_labels_file.read(1), 'big')
#     label = np.zeros((10, 1))
#     label[label_value, 0] = 1
    
#     test_set.append((image, label))


# Plotting an image
# show_image(train_set[0][0])
# plt.show()

weights = []
biases = []
weights.append(np.matrix(np.random.randn(16,784))) #random normal weights for layer 1
weights.append(np.matrix(np.random.randn(16,16))) #random normal weights for layer 2
weights.append(np.matrix(np.random.randn(10,16))) #random normal weights for layer 3
biases.append(np.zeros(16).reshape(16,1)) #zero bias for layer 1
biases.append(np.zeros(16).reshape(16,1)) #zero bias for layer 2
biases.append(np.zeros(10).reshape(10,1)) #zero bias for layer 3

learning_rate=1
number_of_epoches=20
batch_size=10


train_with_SGD(learning_rate,number_of_epoches,batch_size,train_set,weights,biases)


import numpy as np
import matplotlib.pyplot as plt

# A function to plot images
def show_image(img):
    image = img.reshape((28, 28))
    plt.imshow(image, 'gray')


def calculate_output(input_neurons,weights,biases):
    layer1_neurons = weights[0]*input_neurons+biases[0]
    layer2_neurons = weights[1]*layer1_neurons+biases[1]
    output_neurons = weights[2]*layer2_neurons+biases[2]
    print(output_neurons)


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
# show_image(train_set[0][1])
# plt.show()

weights = []
biases = []
weights.append(np.matrix(np.random.randn(16,784))) #random normal weights for layer 1
weights.append(np.matrix(np.random.randn(16,16))) #random normal weights for layer 2
weights.append(np.matrix(np.random.randn(10,16))) #random normal weights for layer 3
biases.append(np.zeros(16).reshape(16,1)) #zero bias for layer 1
biases.append(np.zeros(16).reshape(16,1)) #zero bias for layer 2
biases.append(np.zeros(10).reshape(10,1)) #zero bias for layer 3


calculate_output(train_set[0][0],weights,biases)


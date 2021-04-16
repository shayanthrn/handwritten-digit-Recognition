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

def sigmoid_derv(x):
    return sigmoid(x)*(1-sigmoid(x))

#calculate output of nn
def calculate_output(input_neurons,weights,biases):
    layer1_neurons = sigmoid(weights[0]@input_neurons+biases[0])
    layer2_neurons = sigmoid(weights[1]@layer1_neurons+biases[1])
    output_neurons = sigmoid(weights[2]@layer2_neurons+biases[2])
    nodes={'L1':layer1_neurons,'L2':layer2_neurons,'LO':output_neurons}
    return [np.where(output_neurons==np.amax(output_neurons))[0][0],nodes]

def checkaccuracy(train_set,weights,biases,count):
    correct = 0
    wrong = 0
    for i in range(count):
        answer=calculate_output(train_set[i][0],weights,biases)[0]
        if(answer==np.where(train_set[i][1]==1)[0][0]):
            correct+=1
        else:
            wrong+=1
    print(correct/count)


def train_with_SGD(learning_rate,number_of_epoches,batch_size,train_set,weights,biases):
    cost_array=[]
    for ep in range(number_of_epoches):
        # random.shuffle(train_set)
        for index in range(0,100,batch_size):
            batch=train_set[index:index+10]
            grad_w = []
            grad_b = []
            grad_w.append(np.zeros(16*784).reshape(16,784)) #grad weights for layer 1
            grad_w.append(np.zeros(16*16).reshape(16,16)) #grad weights for layer 2
            grad_w.append(np.zeros(10*16).reshape(10,16)) #grad weights for layer 3
            grad_b.append(np.zeros(16).reshape(16,1)) #grad bias for layer 1
            grad_b.append(np.zeros(16).reshape(16,1)) #grad bias for layer 2
            grad_b.append(np.zeros(10).reshape(10,1)) #grad bias for layer 3
            #chain rule derivation
            for img in batch:
                nodes=calculate_output(img[0],weights,biases)[1]
                inputs=img[0]
                labels=img[1].reshape(10,1)
                derv_a_layer2=np.zeros(16)
                derv_a_layer1=np.zeros(16)
                #layer 3 (grad_w[2],grad_b[2])
                W3=grad_w[2]
                B3=grad_b[2]
                for j in range(10):
                    for k in range(16):
                        W3[j,k]+=2*(nodes['LO'][j,0]-labels[j,0])*nodes['LO'][j,0]*(1-nodes['LO'][j,0])*nodes['L2'][k,0]
                    B3[j,0]+=2*(nodes['LO'][j,0]-labels[j,0])*nodes['LO'][j,0]*(1-nodes['LO'][j,0])
                for k in range(16):
                    for jj in range(10):
                        derv_a_layer2[k]+=2*(nodes['LO'][jj,0]-labels[jj,0])*nodes['LO'][jj,0]*(1-nodes['LO'][jj,0])*weights[2][jj,k]
                # layer 2 (grad_w[1],grad_b[1])
                W2=grad_w[1]
                B2=grad_b[1]
                for k in range(16):
                    for m in range(16):
                        derv_ak=derv_a_layer2[k]
                        W2[k,m]+=derv_ak*nodes['L2'][k,0]*(1-nodes['L2'][k,0])*nodes['L1'][m,0]
                    B2[k,0]+=derv_ak*nodes['L2'][k,0]*(1-nodes['L2'][k,0])
                for m in range(16):
                    for kk in range(16):
                        derv_a_layer1[m]+=derv_a_layer2[kk]*nodes['L2'][kk,0]*(1-nodes['L2'][kk,0])*weights[1][kk,m]
                #layer1 (grad_w[0],grad_b[0])
                W1=grad_w[0]
                B1=grad_b[0]
                for m in range(16):
                    for v in range(784):
                        derv_am=derv_a_layer1[m]
                        W1[m,v]+=derv_am*nodes['L1'][m,0]*(1-nodes['L1'][m,0])*inputs[v,0]
                    B1[m,0]+=derv_am*nodes['L1'][m,0]*(1-nodes['L1'][m,0])


            weights[2]=weights[2]-learning_rate*(grad_w[2]/batch_size)
            weights[1]=weights[1]-learning_rate*(grad_w[1]/batch_size)
            weights[0]=weights[0]-learning_rate*(grad_w[0]/batch_size)
            biases[2]=biases[2]-learning_rate*(grad_b[2]/batch_size)
            biases[1]=biases[1]-learning_rate*(grad_b[1]/batch_size)
            biases[0]=biases[0]-learning_rate*(grad_b[0]/batch_size)
        cost=0
        for img in train_set:
            nodes=calculate_output(img[0],weights,biases)[1]
            labels=img[1].reshape(10,1)
            for j in range(10):
                cost+=(nodes['LO'][j,0]-labels[j,0])**2
        cost/=100
        cost_array.append(cost)
    plt.plot([i for i in range(number_of_epoches)],cost_array)
    plt.show()


def train_with_SGD_vectorized(learning_rate,number_of_epoches,batch_size,train_set,weights,biases):
    cost_array=[]
    for ep in range(number_of_epoches):
        random.shuffle(train_set)
        for index in range(0,len(train_set),batch_size):
            batch=train_set[index:index+10]
            grad_w = []
            grad_b = []
            grad_w.append(np.zeros(16*784).reshape(16,784)) #grad weights for layer 1
            grad_w.append(np.zeros(16*16).reshape(16,16)) #grad weights for layer 2
            grad_w.append(np.zeros(10*16).reshape(10,16)) #grad weights for layer 3
            grad_b.append(np.zeros(16).reshape(16,1)) #grad bias for layer 1
            grad_b.append(np.zeros(16).reshape(16,1)) #grad bias for layer 2
            grad_b.append(np.zeros(10).reshape(10,1)) #grad bias for layer 3
            #calculate dervations using vectors
            for img in batch:
                nodes=calculate_output(img[0],weights,biases)[1]
                inputs=img[0]
                labels=img[1].reshape(10,1)
                #layer 3
                grad_w[2]+=((2*(nodes['LO']-labels))*nodes['LO']*(1-nodes['LO']))@np.transpose(nodes['L2'])
                grad_b[2]+=((2*(nodes['LO']-labels))*nodes['LO']*(1-nodes['LO']))
                grad_a2=np.transpose(weights[2])@((2*(nodes['LO']-labels))*nodes['LO']*(1-nodes['LO']))
                #layer 2
                grad_w[1]+=grad_a2*nodes['L2']*(1-nodes['L2'])@np.transpose(nodes['L1'])
                grad_b[1]+=grad_a2*nodes['L2']*(1-nodes['L2'])
                grad_a1=np.transpose(weights[1])@(grad_a2*nodes['L2']*(1-nodes['L2']))
                #layer 1
                grad_w[0]+=grad_a1*nodes['L1']*(1-nodes['L1'])@np.transpose(inputs)
                grad_b[0]+=grad_a1*nodes['L1']*(1-nodes['L1'])

            weights[2]=weights[2]-learning_rate*(grad_w[2]/batch_size)
            weights[1]=weights[1]-learning_rate*(grad_w[1]/batch_size)
            weights[0]=weights[0]-learning_rate*(grad_w[0]/batch_size)
            biases[2]=biases[2]-learning_rate*(grad_b[2]/batch_size)
            biases[1]=biases[1]-learning_rate*(grad_b[1]/batch_size)
            biases[0]=biases[0]-learning_rate*(grad_b[0]/batch_size)

        cost=0
        for img in train_set:
            nodes=calculate_output(img[0],weights,biases)[1]
            labels=img[1].reshape(10,1)
            for j in range(10):
                cost+=(nodes['LO'][j,0]-labels[j,0])**2
        cost/=len(train_set)
        cost_array.append(cost)
    plt.plot([i for i in range(number_of_epoches)],cost_array)
    plt.show()




def adversarialattack(test_set):
    for img in test_set:
        temp=img[0].reshape((28,28))
        temp=np.roll(temp,4)
        img[0]=temp.reshape((784,1))




# Reading The Train Set
train_images_file = open('train-images.idx3-ubyte', 'rb')
train_images_file.seek(4)
num_of_train_images = int.from_bytes(train_images_file.read(4), 'big')
train_images_file.seek(16)

train_labels_file = open('train-labels.idx1-ubyte', 'rb')
train_labels_file.seek(8)
train_set = []
for n in range(num_of_train_images):
    image = np.zeros((784, 1))
    for i in range(784):
        image[i, 0] = int.from_bytes(train_images_file.read(1), 'big') / 256
    
    label_value = int.from_bytes(train_labels_file.read(1), 'big')
    label = np.zeros(10)
    label[label_value] = 1
    
    train_set.append((image, label))


# Reading The Test Set
test_images_file = open('t10k-images.idx3-ubyte', 'rb')
test_images_file.seek(4)

test_labels_file = open('t10k-labels.idx1-ubyte', 'rb')
test_labels_file.seek(8)

num_of_test_images = int.from_bytes(test_images_file.read(4), 'big')
test_images_file.seek(16)

test_set = []
for n in range(num_of_test_images):
    image = np.zeros((784, 1))
    for i in range(784):
        image[i] = int.from_bytes(test_images_file.read(1), 'big') / 256
    
    label_value = int.from_bytes(test_labels_file.read(1), 'big')
    label = np.zeros(10)
    label[label_value] = 1
    
    test_set.append([image, label])



weights = []
biases = []
weights.append(np.random.normal(size=(16,784))) #random normal weights for layer 1
weights.append(np.random.normal(size=(16,16))) #random normal weights for layer 2
weights.append(np.random.normal(size=(10,16))) #random normal weights for layer 3
biases.append(np.zeros(16).reshape(16,1)) #zero bias for layer 1
biases.append(np.zeros(16).reshape(16,1)) #zero bias for layer 2
biases.append(np.zeros(10).reshape(10,1)) #zero bias for layer 3

learning_rate=1
number_of_epoches=5
batch_size=50

train_with_SGD_vectorized(learning_rate,number_of_epoches,batch_size,train_set,weights,biases)

print("Accuracy for train set:")
checkaccuracy(train_set,weights,biases,num_of_train_images)
print("Accuracy for test set:")
checkaccuracy(test_set,weights,biases,num_of_test_images)
print("Accuracy for test set after adversarial attack:")
adversarialattack(test_set)
checkaccuracy(test_set,weights,biases,num_of_test_images)
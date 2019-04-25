# load pickle
import pickle

# other utilities
from sklearn import svm
import tensorflow as tf
from sklearn.metrics import confusion_matrix


# %% Load the training data
def MNIST_DATASET():
    # Load dataset

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Print training data size
    print('Training data size: ', x_train.shape)
    print('Training data label size:', x_test.shape)

    x_train, x_test = x_train / 255.0, x_test / 255.0

    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = MNIST_DATASET()

print(x_train.shape)
x_train = x_train.reshape(x_train.shape[0],784)
x_test = x_test.reshape(x_test.shape[0],784)

# Training SVM
print('------Training and testing SVM------')
clf = svm.SVC(C=5, gamma=0.05, verbose=True)
clf.fit(x_train, y_train)

# Test on test data
test_result = clf.predict(x_test)
precision = sum(test_result == y_test) / y_test.shape[0]
print('Test precision: ', precision)

# Test on Training data
#train_result = clf.predict(training_features)
#precision = sum(train_result == train_label) / train_label.shape[0]
#print('Training precision: ', precision)

# Show the confusion matrix
#matrix = confusion_matrix(test_label, test_result)
#print(matrix)
#with open('output_file.pickle','wb') as outFile:
#    pickle.dump(outFile, clf)
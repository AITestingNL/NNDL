"""
mnist_svm
~~~~~~~~~

A classifier program for recognizing handwritten digits from the MNIST
data set, using an SVM classifier."""

#### Libraries
# My libraries
import mnist_loader 

# Third-party libraries
from sklearn import svm

def svm_baseline():
    training_data, validation_data, test_data = mnist_loader.load_data()
    # train
    print 'start training ....'
    clf = svm.SVC()
    clf.fit(training_data[0][:2000], training_data[1][:2000])

    # test
    print 'start testing ...'
    predictions = [int(a) for a in clf.predict(test_data[0][:100])]
    print test_data[0][2]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1][:100]))
    
    print "Baseline classifier using an SVM."
    print "%s of %s values correct. " % (num_correct, len(test_data[1][:100]))

if __name__ == "__main__":
    svm_baseline()
    

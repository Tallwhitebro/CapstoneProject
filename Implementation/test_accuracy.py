import neural_network
import support_vector_machine
import data
import util

def accuracy_NN():
    X_train, X_test, y_train, y_test = data.retrieve_data()
    model = neural_network.retrieve_model()

    # evaluate the model
    _, accuracy_model = model.evaluate(X_train, y_train, verbose=False)
    print('Accuracy of model on dataset: %.2f' % (accuracy_model * 100))

    # evaluate the algorithm
    _, accuracy_algorithm = model.evaluate(X_test, y_test, verbose=False)
    print('Accuracy of algorithm on test data: %.2f' % (accuracy_algorithm * 100))

def accuracy_SVM():
    X_train, X_test, y_train, y_test = data.retrieve_data()
    model = support_vector_machine.retrieve_model()

    # evaluate the model
    accuracy_model = model.score(X_train, y_train)
    print('Accuracy of model on dataset: %.2f' % (accuracy_model * 100))

    # evaluate the algorithm
    accuracy_algorithm = model.score(X_test, y_test)
    print('Accuracy of algorithm on test data: %.2f' % (accuracy_algorithm * 100))
    

if __name__ == "__main__":
    accuracy_NN()
    accuracy_SVM()
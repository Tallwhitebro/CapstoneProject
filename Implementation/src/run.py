import data
import neural_network
import support_vector_machine
import predict
import test_accuracy
import util

if __name__ == "__main__":
    config = util.read_yaml()

    model_NN = neural_network.retrieve_model()
    model_SVM = support_vector_machine.retrieve_model()

    all_students_data_path = config["all_students_data_path"]
    target_cap = config["target_cap"]

    # Neural Network
    print("---------------------------------------------------------")
    print("Generating Neural Network results:\n")
    cutoff_NN = predict.calculate_cutoff(all_students_data_path, target_cap, model_NN)
    test_accuracy.accuracy_NN()
    print("\n")
    
    # Support Vector Machine
    print("---------------------------------------------------------")
    print("Generating Support Vector Machine results:\n")
    cutoff_SVM = predict.calculate_cutoff(all_students_data_path, target_cap, model_SVM)
    test_accuracy.accuracy_SVM()
    print("\n")


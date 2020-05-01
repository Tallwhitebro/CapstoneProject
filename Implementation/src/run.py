import data
import neural_network
import support_vector_machine
import predict
import test_accuracy
import util

if __name__ == "__main__":
    config = util.read_yaml()
    all_students_data_path = config["all_students_data_path"]
    target_cap = config["target_cap"]

    # Build Neural Network
    print("---------------------------------------------------------")
    print("Generating Neural Network:\n")
    try:
        model_NN = neural_network.create_model()
    except:
        model_NN = neural_network.retrieve_model()
    print("\n")

    # Build Support Vector Machine
    print("---------------------------------------------------------")
    print("Generating Support Vector Machine:\n")
    try:
        model_SVM = support_vector_machine.create_model()
    except:
        model_SVM = support_vector_machine.retrieve_model()
    print("\n")

    # Evaluate Neural Network
    print("---------------------------------------------------------")
    print("Neural Network results:\n")
    acc_NN = test_accuracy.accuracy_NN()
    print("\n")
    
    # Evaluate Support Vector Machine
    print("---------------------------------------------------------")
    print("Support Vector Machine results:\n")
    acc_SVM = test_accuracy.accuracy_SVM()
    print("\n")

    # Determing cutoff GPA
    print("---------------------------------------------------------")
    print("Determining Cutoff:\n")
    if (acc_NN > acc_NN):
        predict.calculate_cutoff(all_students_data_path, target_cap, model_NN)
    else:
        predict.calculate_cutoff(all_students_data_path, target_cap, model_SVM)




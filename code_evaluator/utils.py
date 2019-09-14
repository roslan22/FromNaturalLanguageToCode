import pickle
import json
import matplotlib.pyplot as plt
import numpy as np 

class Utils:
    @staticmethod
    def save_obj(path, obj, name):
        with open(path + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_obj(path, name):
        with open(path + name + '.pkl', 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def load_json(path, name):
        with open(path + name, 'r') as f:
            return json.load(f)

    @staticmethod
    def show_train_plots(history):
        # Plot training & validation accuracy values
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    @staticmethod
    def run_counter_stats(counter):
        labels = []
        sizes = []

        print('Statistics:')
        for counter_name in counter.counterByName:
            print('Exception of type {} : {}'.format(counter_name, counter.counterByName[counter_name]))
            # Pie chart, where the slices will be ordered and plotted counter-clockwise:
            if counter.counterByName[counter_name] != 0:
                labels.append(counter_name)
                sizes.append(counter.counterByName[counter_name])

        fig, ax = plt.subplots()
        plt.title('Number of Exceptions by type')
        plt.bar(np.arange(len(labels)), sizes)
        plt.xticks(np.arange(len(labels)), labels)
        plt.show()

    @staticmethod
    def get_sorted_errors_by_occurance(errors_by_type, err_type):
        error_calculation = {}
        for snipet_error in errors_by_type['Name Error']:
            if snipet_error == None:
                continue 

            if snipet_error['error'] in error_calculation:
                error_calculation[snipet_error['error']] += 1
            else:
                error_calculation[snipet_error['error']] = 0

        sorted_error_calc = sorted(error_calculation.items(), key=lambda x: x[1], reverse=True)
        return sorted_error_calc


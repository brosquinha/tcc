import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from sklearn.model_selection import KFold, StratifiedKFold

from utils.loggable import Loggable


class Base(Loggable):

    def __init__(self, log_level, k_splits=4, balance_data=True, balance_tolerance=1.0, **kwargs):
        super().__init__(log_level=log_level)
        self.k_splits = k_splits
        self.balance_data = balance_data
        self.balance_tolerance = balance_tolerance
        self.test_data = None
        self.train_data = None
    
    def _repeated_k_fold_training(self, all_data, x_data_function, y_data_function):
        if self.balance_data:
            rkf = StratifiedKFold(n_splits=self.k_splits, shuffle=True)
            x_data = x_data_function(all_data)
            y_data = y_data_function(all_data)
            data_split = rkf.split(x_data, y_data)
        else:
            rkf = KFold(n_splits=self.k_splits, shuffle=False)
            data_split = rkf.split(all_data)
        
        for train, test in data_split:
            self.logger.info("Cross-validating")
            self.train_data = np.array(all_data)[train]
            self.test_data = np.array(all_data)[test]
            
            if self.balance_data:
                self.logger.info("Balacing training dataset...")
                self._balance_train_data(x_data_function, y_data_function)

            x_train_data = x_data_function(self.train_data)
            y_train_data = y_data_function(self.train_data)

            self._train_model(x_train_data, y_train_data)
            yield self.text_clf

    def _balance_train_data(self, x_data_function, y_data_function):
        x_train_data = x_data_function(self.train_data)
        y_train_data = y_data_function(self.train_data)

        classes = {}
        for y_class in np.unique(y_train_data):
            x_class = np.array(x_train_data)[[index for index, value in enumerate(y_train_data) if value == y_class]]
            classes[y_class] = x_class
        
        max_quantity = int(min([len(classes[x]) for x in classes]) * self.balance_tolerance)
        self.logger.debug(', '.join([str(len(classes[x])) for x in classes]))

        elements_to_be_transferred_index = []
        for y_class, x_class in classes.items():
            enough_x_class = x_class[:max_quantity]
            excess_x_class = x_class[max_quantity:]
            for x in excess_x_class:
                x_index = x_train_data.index(x)
                elements_to_be_transferred_index.append(x_index)

        self.logger.debug('Original length of train_data: %d' % len(self.train_data))
        self.logger.debug('Original length of test_data: %d' % len(self.test_data))

        elements = self.train_data[elements_to_be_transferred_index]
        self.train_data = np.delete(self.train_data, elements_to_be_transferred_index, 0)
        self.test_data = np.concatenate((self.test_data, elements))

        self.logger.debug('New length of train_data: %d' % len(self.train_data))
        self.logger.debug('New length of test_data: %d' % len(self.test_data))

    def _plot_confusion_matrix(
            self, predicted_y, expected_y, labels, largesize=False, 
            title='Confusion matrix', filename='confusion_matrix.png'):
        cm = confusion_matrix(expected_y, predicted_y)
        self.logger.info(cm)

        disp = ConfusionMatrixDisplay(cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues, values_format='')
        plt.xlabel('Predicted result')
        plt.ylabel('Expected result')
        plt.title(title)
        if largesize:
            plt.rcParams["figure.figsize"] = (18.5, 10.5)
        plt.savefig(os.path.join("output", filename), dpi=200)
        plt.close()
        
        self.logger.info(f'\n{classification_report(expected_y, predicted_y, digits=4)}')

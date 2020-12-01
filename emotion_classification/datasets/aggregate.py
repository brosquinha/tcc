from random import shuffle


class Aggregate():
    
    def __init__(self, *args):
        self.all_data = []
        x_data = []
        y_data = []

        for dataset in args:
            x_data += dataset.get_x_data(dataset.all_data)
            y_data += dataset.get_y_data(dataset.all_data)

        self.all_data = list(zip(x_data, [str(x) for x in y_data]))
        shuffle(self.all_data)

    def get_x_data(self, data):
        return [x[0] for x in data]

    def get_y_data(self, data):
        return [x[1] for x in data]

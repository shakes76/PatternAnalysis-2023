from util import reset_dir
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

class Logger:

    def __init__(self, reset=False, file_name='VAE_log.txt') -> None:
        self.data = {}
        self.file_name = f"log/{file_name}"
        if file_name:
            try:
                os.mkdir('log')
            except Exception as e:
                # file is exist.
                pass

        # Should we re-write the log
        if reset and os.path.exists(self.file_name):
            os.remove(self.file_name)

    def update_value(self, variable_name, variable, save=True) -> None:
        # Update variable for logger
        if variable_name not in self.data:
            self.data[variable_name] = []
        self.data[variable_name].append(variable)
        if save:
            with open(self.file_name, "a") as fout:
                print(f'{variable_name}|{float(variable)}', file=fout)
            

    def update_dict(self, data_dict, save=True) -> None:
        # Update all vaiables then auto-save.
        with open(self.file_name, "a") as fout:
            for variable_name, variable in data_dict.items():
                if variable_name not in self.data:
                    self.data[variable_name] = []
                if save:
                    print(f'{variable_name}|{float(variable)}', file=fout)
                self.data[variable_name].append(variable)

    def load(self, file_name) -> None:
        # Clear the dict
        self.data = {}
        with open(self.file_name) as fin:
            for row in fin:
                variable_name, variable = row.split('|')
                variable = float(variable)
                self.update_value(variable_name, variable, save=False)

    @staticmethod
    def smooth_num(X, window=5):
        Y = []
        cur = 0
        for i in range(len(X)):
            if i >= window:
                cur -= X[i - window]
            cur += X[i]
            Y.append(cur / min(i+1, window))
        return Y

    def show_plot(self, smooth_window=5, xlabel='iteration', ylabel='loss'):
        print("!")
        for k, v in self.data.items():
            plt.plot(np.arange(len(v)), self.smooth_num(v, window=smooth_window), label=k)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()

    def get_range_status(self, start=0, end=None):
        ret = {}
        for k, v in self.data.items():
            target = v[start:end]
            ret[k] = sum(target) / len(target)
        return ret
        

if __name__ == '__main__':
    logger = Logger(reset=False)
    logger.load('VAE_log.txt')
    logger.show_plot()
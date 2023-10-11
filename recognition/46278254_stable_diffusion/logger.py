import os
import numpy as np
import matplotlib.pyplot as plt


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
        self.file_name = f"log/{file_name}"
        self.data = {}
        with open(self.file_name) as fin:
            for row in fin:
                # The key,value pair is splited with '|'
                variable_name, variable = row.split('|')
                variable = float(variable)
                self.update_value(variable_name, variable, save=False)

    @staticmethod
    def smooth_num(X, window=5):
        # This function will smooth the line with a specific width window.
        # And this will help to generate more comfortable lines.
        Y = []
        cur = 0
        for i in range(len(X)):
            # If i is out of window, forget the value.
            if i >= window:
                cur -= X[i - window]
            cur += X[i]
            Y.append(cur / min(i+1, window))
        return Y

    def show_plot(self, start=None, smooth_window=5, xlabel='iteration', ylabel='loss'):
        L = len(self.data['recon_loss'])
        # We only plot the line after 50 iterations.
        if start is None:
            start = min(L//2, 50)

        X = np.arange(L)[start:]

        # mapping function from numbers to smooth numbers
        def m_f(v): return (X[-len(v[start:]):],
                            self.smooth_num(v, window=smooth_window)[start:])

        def my_plot(key, label, c):
            if key in self.data:
                plt.plot(*m_f(self.data[key]), label=label, c=c)

        # Reconstruction Loss Plot
        plt.subplot(2, 2, 1)
        my_plot('recon_loss', 'Reconstruction Loss', 'blue')
        plt.title("Reconstruction Loss")
        plt.legend()

        # Regularization Loss Plot
        plt.subplot(2, 2, 2)
        my_plot('diff_loss', 'Regularization Loss', 'orange')
        plt.title("Regularization Loss")
        plt.legend()

        # GAN Loss Plot
        plt.subplot(2, 2, 3)
        my_plot('fake_recon_loss', 'Reconstruction Score', 'blue')
        my_plot('fake_sample_loss', 'Random Sample Score', 'red')
        my_plot('discriminator_loss', 'Discriminator Toatal Loss', 'purple')
        plt.title("GAN part Loss")
        plt.legend()

        # Weight Plot
        plt.subplot(2, 2, 4)
        # Well, w_recon seems to be alwaye 1.
        # my_plot('w_recon', 'Reconstruction weight', 'blue')
        my_plot('w_kld', 'Regularization weight', 'orange')
        my_plot('w_dis', 'Reconstruction GAN weight', 'blue')
        my_plot('w_sample', 'Sample GAN weight', 'red')
        plt.legend()
        plt.title("Weight for Each Loss")

        plt.show()

    def get_range_status(self, start=0, end=None):
        # This function is to cut the array.
        # And the reason why this function exist is to cut the initial loss.
        # The value of it is too high and harm the qulity of plot.
        ret = {}
        for k, v in self.data.items():
            target = v[start:end]
            ret[k] = sum(target) / len(target)
        return ret


if __name__ == '__main__':
    logger = Logger(reset=False, file_name='VAE_log.txt')
    logger.load('VAE_log.txt')
    logger.show_plot()

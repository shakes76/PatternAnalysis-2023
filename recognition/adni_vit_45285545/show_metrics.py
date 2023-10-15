import argparse

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def main(args):
    df = pd.read_csv(args.csvfile)
    df.insert(0, 'epoch',  range(1, len(df) + 1))

    sns.set_palette(sns.color_palette('muted'))

    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    fig.tight_layout()

    sns.lineplot(df, x='epoch', y='train_loss', ax=axs[0], label='Training')
    sns.lineplot(df, x='epoch', y='valid_loss', ax=axs[0], label='Validation')

    sns.lineplot(df, x='epoch', y='train_acc', ax=axs[1], label='Training')
    sns.lineplot(df, x='epoch', y='valid_acc', ax=axs[1], label='Validation')

    axs[0].set_title('Cross-Entropy Loss')
    axs[1].set_title('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[1].set_xlabel('Epoch')
    axs[0].set_ylabel('')
    axs[1].set_ylabel('')

    fname = args.csvfile.split('.csv')[0] + '.png'
    fig.savefig(fname, bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csvfile', help='CSV filename containing training metrics')
    args = parser.parse_args()
    main(args)

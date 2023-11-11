import argparse

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def main(args):
    # Load CSV file containing metrics exported by model during training;
    # this doesn't include an "epoch" column, so this is manually added
    df = pd.read_csv(args.csvfile)
    df.insert(0, 'epoch',  range(1, len(df) + 1))

    # Set plot colour palette to whatever you like
    sns.set_palette(sns.color_palette('muted'))

    # Plot training and validation losses and accuracies against epoch number
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))

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

    # Save plot as PNG image with same filename as loaded CSV metrics file
    fig.tight_layout()
    fname = args.csvfile.split('.csv')[0] + '.png'
    fig.savefig(fname, bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csvfile', help='CSV filename containing training metrics')
    args = parser.parse_args()
    main(args)

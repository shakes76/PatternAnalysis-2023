import os
import random
import torch


def get_patients(path: str) -> [str]:
    uids = []
    dire = os.fsdecode(path)
    for file in os.listdir(dire):
        filename = os.fsdecode(file)
        substrings = filename.split("_", 2)
        if substrings[0] not in uids:
            uids.append(substrings[0])
    return uids


def partition(l: [str], n: int):
    random.shuffle(l)
    return [l[i::n] for i in range(n)]


def get_validation_sets(num: int) -> [[str]]:
    """
        Returns a list of lists containing a non-intersecting list of patient id's to be used for validation sets
    """
    partition_ad = partition(get_patients("ADNI_AD_NC_2D/AD_NC/train/AD"), num)
    partition_nc = partition(get_patients("ADNI_AD_NC_2D/AD_NC/train/NC"), num)

    return partition_ad, partition_nc


if __name__ == "__main__":
    ad, nc = get_validation_sets(30)
    print(ad)
    print(nc)

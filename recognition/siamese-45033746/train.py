from dataset import patient_split, SiameseDataSet, compose_transform
import modules


# 256x240

def load() -> (SiameseDataSet, SiameseDataSet):
    train, val = patient_split()
    transform = compose_transform()
    TrainSet = SiameseDataSet(train, transform)
    ValidationSet = SiameseDataSet(val, transform)

    return TrainSet, ValidationSet


if __name__ == '__main__':
    pass

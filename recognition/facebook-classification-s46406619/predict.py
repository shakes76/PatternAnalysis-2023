from dataset import *
from modules import *

model = GCN()
model.load_state_dict(torch.load('C:/Area-51/2023-sem2/COMP3710/PatternAnalysis-2023/recognition/facebook-classification-s46406619/model.pth'))
model.eval()

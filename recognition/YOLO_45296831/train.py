import time
import datetime

from modules import get_device, get_yolo

def main():
    '''
    Trains the YOLO model based on the data.yaml file
    '''
    start_time = time.time()
    print(f"Start time: {time.ctime(start_time)}")

    device = get_device()
    model = get_yolo(device, pretrained=False)

    # Train
    #TODO set train function params
    results = model.train(batch=-1, device=device, model=model, data='C:/Users/clark/OneDrive/Documents/2023/COMP3710/PatternAnalysis-2023/recognition/YOLO_45296831/data.yaml', epochs=80, imgsz=640)

    end_time = time.time()
    print(f"End Time: {time.ctime(end_time)}")
    print(f"Time Taken to Train (H:M:S) : {datetime.timedelta(end_time - start_time)}")

if __name__ == "__main__":
    main()
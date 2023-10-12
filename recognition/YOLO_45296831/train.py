import time
import datetime

from modules import get_device, get_yolo

def main():
    """
    Set up the model and live stream analysis video.
    Available for testing.
    """
    start_time = time.time()
    print(f"Start time: {time.ctime(start_time)}")

    device = get_device()
    model = get_yolo(device, pretrained=False)

    # Train
    #TODO set train function params
    results = model.train(data='dataset.yaml', epochs=0, imgsz=0)

    end_time = time.time()
    print(f"End Time: {time.ctime(end_time)}")
    print(f"Time Taken to Train (H:M:S) : {datetime.timedelta(end_time - start_time)}")

if __name__ == "__main__":
    main()
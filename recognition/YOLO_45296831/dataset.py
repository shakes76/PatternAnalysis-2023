import cv2


rangpur_data_path = "/home/Student/s4529683/COMP3710/"
home_data_path = "C:/Users/clark/OneDrive/Documents/2023/COMP3710/ISIC2018_Task1_Training_GroundTruth_x2"

def mask_to_bp(img_path):
    """
    Converts binary segmentation image into boundary polygon for training
    
    image_path: image to be converted
    
    return: list of polygon co-ordinates
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    polygons = []

    for obj in contours:
        coords = []

        for point in obj:
            coords.append(int(point[0][0])/img.shape[0])
            coords.append(int(point[0][1])/img.shape[1])
        
        polygons.append(coords)

    return polygons




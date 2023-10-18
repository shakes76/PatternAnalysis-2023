import cv2
import os


rangpur_data_path = "/home/Student/s4529683/COMP3710/"
home_data_path = "C:/Users/clark/OneDrive/Documents/2023/COMP3710/ISIC2018_Task1_Training_GroundTruth_x2"

def mask_to_bp(img_path):
    '''
    Converts binary segmentation image into boundary polygon for training
    
    image_path: image to be converted
    return: list of polygon co-ordinates
    '''
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    polygons = []
    for obj in contours:


        for point in obj:
            polygons.append(str(int(point[0][0])/img.shape[1]))
            polygons.append(str(int(point[0][1])/img.shape[0]))
        
        

    return " ".join(polygons)


def main():
    '''
    Loops through all segmentation .png files in a given directory and creates a .txt file
    with the boundaries of the segmentation images.
    '''
    textfile = "ISIC_"

    for filename in os.listdir(home_data_path):
        if filename.endswith('.png'):
            name = str(filename).replace("_segmentation.png","")
            name = name + ".txt"
            with open("COMP3710/test/labels/" + name, 'w') as f:
                f.write(str(0) + " ")
                f.write(mask_to_bp(home_data_path+"/"+filename))

if __name__ == "__main__":
    main()
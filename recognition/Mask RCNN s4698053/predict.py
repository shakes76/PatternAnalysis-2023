from CONFIG import *
from modules import *
from torch.utils.data import DataLoader, random_split
from dataset import ISICDataloader


def check_cuda():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        exit("Warning CUDA not Found. Using CPU")
    return device

def main():
    model = YoloV1(split_size=7, num_boxes=1, num_classes=2)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)


    data = ISICDataloader(classify_file=classify_file, 
                                                 photo_dir=photo_dir, 
                                                 mask_dir=mask_dir,
                                                 mask_empty_dim=image_size)
    
    generator = torch.Generator().manual_seed(torch_seed)
    train_dataset, test_dataset = random_split(data, [train_size, test_size], generator=generator)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test(test_dataloader, model)

def test(test_dataloder, model):
    device = check_cuda()
    model.to(device)
    model.eval()

    pred_boxes, target_bboxes = get_bboxes(
        test_dataloder, model, iou_threshold=0.5, threshold=0.4
    )
    print()

    mean_avg_prec = mean_average_precision(
            pred_boxes, target_bboxes, iou_threshold=0.5, box_format="midpoint"
    )

    count = 0
    IOUs = []
    for pred_box, target_bbox in zip(pred_boxes, target_bboxes):
        if target_bbox[2] == 1.0:
            intersection_over_union()
    print("Mean average precision is:")



if __name__ == "__main__":
    main()
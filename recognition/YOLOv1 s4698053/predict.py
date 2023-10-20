from CONFIG import *
from modules import *
from torch.utils.data import DataLoader, random_split
from dataset import ISICDataloader
import matplotlib.pyplot as plt


def check_cuda():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        exit("Warning CUDA not Found. Using CPU")
    return device

def main():
    model = YoloV1(split_size=1, num_boxes=1, num_classes=2)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    #evaluate(model)
    predict(model)


def predict(model):
    device = check_cuda()
    model.to(device)
    model.eval()
    data = ISICDataloader(classify_file=classify_file, 
                                                 photo_dir=photo_dir, 
                                                 mask_dir=mask_dir,
                                                 mask_empty_dim=image_size)
    
    generator = torch.Generator().manual_seed(torch_seed)
    _, test_dataset = random_split(data, [train_size, test_size], generator=generator)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    count = 0
    for raw_image, raw_target in test_dataset:
        # Only display cancer cells
        if torch.any(raw_target != 0):
            image = raw_image.cpu().numpy().transpose((1, 2, 0))
            #image = np.transpose(raw_image, (2, 0, 1))
            ax = axes[0, count]
            ax.imshow(image)
            
            bbox_true = max(cellboxes_to_boxes(raw_target.unsqueeze(0))[0], key=lambda x: x[1])
            class_id, class_prob, x1, y1, x2, y2 = bbox_true
            x1, x2 = x1 * image_size[1], x2 * image_size[1]
            y1, y2 = y1 * image_size[0], y2 * image_size[0]
            ax.set_title(class_id)
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2))

            ax = axes[1, count]
            ax.imshow(image)
            predictions = model(raw_image.unsqueeze(0))
            bbox_pred = max(cellboxes_to_boxes(predictions)[0], key=lambda x: x[1])
            class_id, class_prob, x1, y1, x2, y2 = bbox_pred
            x1, x2 = x1 * image_size[1], x2 * image_size[1]
            y1, y2 = y1 * image_size[0], y2 * image_size[0]
            ax.set_title(class_id)
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2))

            ax.axis('off')
            
            count += 1

        
        if count > 3 - 1:
                break
    plt.show()


def evaluate(model):
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

    pred_boxes_labelled = []
    target_boxes_labelled = []
    for pred_box, target_bbox in zip(pred_boxes, target_bboxes):
        if target_bbox[2] == 1.0:
            pred_boxes_labelled.append(pred_box[3:])
            target_boxes_labelled.append(target_bbox[3:])
            
    average_IOU = intersection_over_union(torch.Tensor(pred_boxes_labelled), torch.Tensor(target_boxes_labelled))
    print(f"Mean average precision is: {mean_avg_prec}")
    print(f"Mean IOU is: {torch.mean(average_IOU)}")



if __name__ == "__main__":
    main()
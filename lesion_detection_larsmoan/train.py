from yolov7.train import train

opt = {
    'epochs': 300,
    'batch_size': 32,
    'cfg': 'cfg/training/yolov7.yaml',
    'data': 'data/coco.yaml',
    'weights': '',
    'device': '0',
    'name': 'yolov7',
    'img_size': [640, 640],
    'project': 'runs/train',
    'entity': None,
    'entity_name': None,
    'bucket': None,
    'cache_images': False,
}

train(hyp, opt, device=opt['device'])
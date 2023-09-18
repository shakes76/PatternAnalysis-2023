from utils.utils_model import select_model

def load_model(opt, CLASS_NUM):
    model = select_model(opt.model_name, CLASS_NUM, (opt.image_size, opt.image_size), opt.image_channel,
                             opt.pretrained)
    return model
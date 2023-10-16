from train import *

# load the model
unet = UNet3D()
unet.load_state_dict(torch.load("net_paras.pth"))
unet.eval()

# load the data
dataset = NiiImageLoader("semantic_MRs_anon/*",
                         "semantic_labels_anon/*")

test, _ = torch.utils.data.random_split(dataset, [1, 210])

# predict
for X, _ in test:
    X = X.unsqueeze(0).float().to(device)
    pred = unet(X).argmax(1)

# save the output
ni_img = nib.Nifti1Image(pred, func.affine)
nib.save(ni_img, 'output.nii.gz')



from train import *
import matplotlib.pyplot as plt

# load the model
unet = UNet3D()
unet.load_state_dict(torch.load("net_paras.pth", map_location="cpu"))
unet.eval()

# load the data
dataset = NiiImageLoader("semantic_MRs_anon/*",
                         "semantic_labels_anon/*")

# pick a nii image randomly
test, _ = torch.utils.data.random_split(dataset, [1, 210])

# predict
for X, _ in test:
    X = X.unsqueeze(0).float().to(device)
    pred = unet(X).argmax(1).detach().float().numpy()


# save the output
ni_img = nibabel.Nifti1Image(pred, np.eye(4))
nibabel.save(ni_img, 'output.nii.gz')





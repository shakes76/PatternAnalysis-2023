# Code Explain

## Coding Logic
![image](https://github.com/Albert-bc/vq-vae/assets/59477394/5e6f585e-e2c5-4267-87f5-8ba068c54b84)

from https://medium.com/analytics-vidhya/blog-post-explained-creating-images-from-text-using-dall-e-4613376bbf10






## Evaluation Process
In the first generation, the images is colourful, because plt.imshow is used directly to show the image, the default output of matplotlib is RGB 3 channels, it is like pic1. To solve this problem, gray scale is used as follow: plt.imshow(img[:,:,0], cmap='gray', vmin=0, vmax=1, interpolation='none'), then it changed into pic2.

![image](https://github.com/Albert-bc/vq-vae/assets/59477394/84702278-ecd6-4d21-92aa-08fc556e577a)

Pic1.									

![image](https://github.com/Albert-bc/vq-vae/assets/59477394/11d1f8e2-8777-4971-be52-51d9cc006727)

Pic2.

Now, the generated picture is looks reasonably like a brain CT image and SSIM reaches can reach 80%, but it is still blur. Deep into the training code, autocast() is used to improve the speed of training, but the generated picture is not clear enough. Because autocast() will lower the graphic precision although it will accelerate the training speed. In the training, the min loss of using autocast() training 15000 times is 0.006, however if autocast() is not used, the min loss training 15000 is 0.0015. Besides, the matplotlib may also lower the resolution, so save_image inside torchvision.utils is used to take place of imshow(), the 5000 times trained picture is generated like Pic3. 

![image](https://github.com/Albert-bc/vq-vae/assets/59477394/da8a8eaa-1691-4f69-ae94-752b7ebf581c)

Pic3.

It is still blur, but the sharpness of the images is improved and the SSIM reaches 90%. Next, we only need to train more times, so when we train 20000 times, the output is shown in Pic4.

![image](https://github.com/Albert-bc/vq-vae/assets/59477394/5a680481-d186-4513-aa39-07a05730cdc4)

Pic 4.

In this training, it is clear enough and SSIM could reach 95% one picture of all these 64 brain pictures is not shown, which means it could be overfitting. Therefore, we have to adjust the learning rate and track the loss during the training process. Since the first 1000 times training have high loss, we only record the loss after training 1000 epochs. The result is as follows: Pic5 is when learning rate equals 0.01, Pic6 is when learning rate equals 0.001, Pic7 is when learning rate equals 0.0001. The best images they can generate is also shown in Pic 8, 9, 10.

![image](https://github.com/Albert-bc/vq-vae/assets/59477394/a090cf33-0e28-4e51-9d51-cabb5fc1c014) Pic5.	![image](https://github.com/Albert-bc/vq-vae/assets/59477394/6307499f-461a-4cb7-8a9b-0f4602c303de) Pic8. (No_4543 _Loss_0.00182_SSIM_92.36%)     

![image](https://github.com/Albert-bc/vq-vae/assets/59477394/3ec97594-196f-4ddc-9bde-a1d2dc587580) Pic6. ![image](https://github.com/Albert-bc/vq-vae/assets/59477394/dceaab87-88ec-4007-b563-aed41eec277e) Pic9. (No_6726 _Loss_0.00117_SSIM_97.073%)	  

![image](https://github.com/Albert-bc/vq-vae/assets/59477394/126a93ce-776c-4b4b-adf7-d0c07770acbd) Pic7. ![image](https://github.com/Albert-bc/vq-vae/assets/59477394/ae1995b9-23a8-45ea-b83b-ce62e68418fd) Pic10. (No_9693 _Loss_0.001575_SSIM_95.94%)

From the images, we can clearly see that inside 10000 times training, SSIM and loss performance is the best when learning rate equals 0.001, but it still overfitting because 1/64 image is missing (all black). Therefore, we need to adjust the learning rate to make it change in multistep. Code: from torch.optim.lr_scheduler import MultiStepLR is used to adjust the learning rate. However, it is easily getting loss like Pic11. if the learning rate step is not chosen well. After several attempts, we get the best generated image so far is shown in Pic12.

![image](https://github.com/Albert-bc/vq-vae/assets/59477394/9ea36fec-8a6e-4cd7-85ea-7009d848f835)

Pic11.

![image](https://github.com/Albert-bc/vq-vae/assets/59477394/9fd24b25-cbf9-4380-a6a4-0f9ce211c5ad)

Pic12. (1024 x 1024 pixel) (No_14165_img_Loss_0.00107_SSIM_97.37%)

In Pic12., the loss decreases to 0.0010 and SSIM increases to 97.0%. The image now looks quite reasonable clear, and SSIM is quite high. If we continue want to improve the resolution of the generated image, we can delete the line: transforms.Resize((128, 128)). This line only works to make sure the code works functionally on low memory GPU. Then, we can get the output image as clear as the input image


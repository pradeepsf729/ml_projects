import torch
import imageio
import os

def load_images(data_dir,
                *,
                types=['jpeg', 'jpg', 'png'],
                pixels =(256, 256),
                rgb = True):
    all_files = os.listdir(data_dir)
    img_type = lambda file_name: file_name.split('.')[-1]
    all_files = [file_name for file_name in all_files if img_type(file_name) in types]

    # initialize the output tensor.
    if rgb:
        colour_pixel = 3
    else:
        colour_pixel = 1

    batch_t = torch.zeros(len(all_files), colour_pixel, pixels[0], pixels[1], dtype=torch.uint8)

    for i, file_name in enumerate(all_files):
        img_arr = imageio.imread(os.path.join(data_dir, file_name))
        print('got image as shape : ', img_arr.shape, ' for img : ', file_name)
        img_t = torch.from_numpy(img_arr)
        img_t = img_t.permute(2, 0, 1)
        batch_t[i] = img_t
    
    print('loaded ', len(batch_t), ' images , with shape ', batch_t[0].shape)
    return batch_t

def normalize(batch_images):
    for batch_img in batch_images: # for each image in batch.
        n_channels = batch_img.shape[0]
        for c in range(n_channels): # each channel value at one.            
            mean = torch.mean(batch_img[:, c])
            std = torch.std(batch_img[:, c])
            batch_img[:, c] =  (batch_img[:, c] - mean) / std
    print(batch_images[0][:1, :])

res = load_images('image-cats')
print(res[0][:1, :])

normalize(res)
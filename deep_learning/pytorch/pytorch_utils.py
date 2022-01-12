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

def one_hot_encoding_words(text, token_char='\n'):
    all_words = text.replace(token_char, ' ').split()
    punctuations = '.,;:"!?”“_-'
    all_words = [word.strip(punctuations) for word in all_words]

    all_unique_words = sorted(set(all_words))
    word2index_dict = {word: i for i, word in enumerate(all_unique_words)}

    all_lines = text.split('\n')
    one_line = all_lines[0]
    one_line = one_line.strip(punctuations)
    one_line_words = [word.strip(punctuations) for word in one_line.split()]

    if len(one_line) > 0:
        one_hot_tensor = torch.zeros(len(one_line), len(word2index_dict))
        print('Word index size - ', len(word2index_dict))
        for i, word in enumerate(one_line_words):
            word_ind = word2index_dict[word]
            # mark that word index to 1, remaining zeros
            one_hot_tensor[i][word_ind] = 1

        print(one_hot_tensor.shape)
        #print(one_hot_tensor)

# res = load_images('image-cats')
# print(res[0][:1, :])

# res = load_images('image-dogs')
# print(res[0][:1, :])
#normalize(res)

# usaing same novel text
one_hot_encoding_words(open('data/1342-0.txt').read())

# using python source code.
one_hot_encoding_words(open('../basic_nn/nn.py').read(), 'r"[^a-zA-Z0-9_]+"')



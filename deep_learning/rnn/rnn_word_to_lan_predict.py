'''
Neural network created using RNN to predict the language for the given word.
Trained with words from 18 languages.

Output gives top predictions of the language:

$ python predict.py Hinton
(-0.47) Scottish
(-1.52) English
(-3.57) Irish

Training data used : data/names/*.txt
'''

import io
import glob
from nis import cat
import os
import unicodedata
import string
from numpy import float32
import torch
import torch.nn as nn
import random
import math
import time
import matplotlib.pyplot as plt
import sys
from zipfile import ZipFile
import urllib
import urllib.request

all_letters = string.ascii_letters + ' .,;'
n_letters = len(all_letters)

def download_input_data(output_path):
    URL = 'https://download.pytorch.org/tutorial/data.zip'
    FILE = 'data.zip'
    if not os.path.isfile(FILE):
        print ('Downloading {0} and saving as {1} ...'.format(URL, FILE))
        urllib.request.urlretrieve(URL, FILE)

    if not os.path.isdir(output_path):
        print ( 'Unzipping images...' )
        with ZipFile(FILE) as zip_images:
            zip_images.extractall(output_path)
        os.remove(FILE) # zip file removed
        print ( 'Done!' )
    else:
        print('image data already available - [not downloading!!]')

def find_files(path):
    '''
    Get all the files in the given path (recursively)
    '''
    return glob.glob(path)

def unicode_to_ascii(s):
    '''
    Converts all the unicode chars to ascii for the given text
    '''
    ascii_chars = []
    for c in unicodedata.normalize('NFD', s):
        if unicodedata.category(c) != 'Mn' and c in all_letters:
            ascii_chars.append(c)
    return ''.join(ascii_chars)

def read_file_to_ascii(file_name):
    fd = open(file_name, encoding='utf-8')
    text_lines = fd.read().strip().split('\n')
    return [unicode_to_ascii(line) for line in text_lines]

def load_lang_category_data(path):
    category_lines = {}
    all_categories = []
    for file in find_files(path):
        file_name = os.path.basename(file)
        category = os.path.splitext(file_name)[0]
        category_lines[category] = read_file_to_ascii(file)
        all_categories.append(category)
    
    return category_lines, all_categories, len(all_categories)

def line_to_tensor(line):
    '''
    Performing one-hot encoding for a given line for each letter.
    Example for input 'abc'
    tensor will be
    [
     [1.0, 0.0, 0.0, ... all zeros ],
     [0.0, 1.0, 0.0, ... all zeros ],
     [0.0, 0.0, 1.0, ... all zeros ]
    ]

    total letters considered is 57 (all ascii + .,;)

    Returns: Output of the tensor will be len(line) * 1 * len(all_letters)
    line_to_tensor('abc').shape    -> [3, 1, 57]
    line_to_tensor('abcdef').shape -> [6, 1, 57]
    '''
    res = torch.zeros(len(line), 1, n_letters)
    for i, letter in enumerate(line):
        let_ind = all_letters.find(letter)
        res[i][0][let_ind] = 1
    return res

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, ouput_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        # to calculate the hidden state
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)

        # calculate ouput
        self.i2o = nn.Linear(input_size + hidden_size, ouput_size)
        
        # for ouput category calculation
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden_state):
        combined = torch.cat((input, hidden_state), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

def get_category_from_output(output, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def get_random_training_ex(lang_words_map, all_categories):
    # picka random category
    category_ind = random.randint(0, len(all_categories) - 1)
    lang_category = all_categories[category_ind]

    # pick a word/line for that category
    random_word_ind = random.randint(0, len(lang_words_map[lang_category]) - 1)
    line = lang_words_map[lang_category][random_word_ind]

    line_tensor = line_to_tensor(line)
    # tensor for that category
    cat_tensor = torch.tensor([all_categories.index(lang_category)], dtype=torch.long)

    return lang_category, line, cat_tensor, line_tensor

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def train(all_categories, rnn, loss_fn, *,
            learning_rate = 0.005,
            epochs = 100000,
            print_status_for_every = 10000,
            accumulate_loss_for_every = 1000):

    all_losses = []
    current_loss = 0

    start = time.time()

    for epoch in range(1, epochs + 1):
        # get the training example:
        example_entry = get_random_training_ex(lang_to_wrds_map, all_categories)

        category, line = example_entry[0], example_entry[1]
        category_tensor, line_tensor = example_entry[2], example_entry[3]

        # training the example
        hidden = rnn.initHidden()
        rnn.zero_grad()
        num_letters = line_tensor.shape[0]

        # complete rnn forward all the letters with each time 
        # previous hidden state as input
        for i in range(num_letters):
            output, hidden = rnn(line_tensor[i], hidden)

        loss = loss_fn(output, category_tensor)
        loss.backward() # backpropagation

        # update parameters (weights, biases) using learning rate and gradients
        # can we also use optimizer to do it directly.
        for param in rnn.parameters():
            param.data.add_(param.grad.data, alpha=-learning_rate)

        current_loss += loss

        if epoch % print_status_for_every == 0:
            guess, guess_i = get_category_from_output(output, all_categories)
            correct = '✓' if guess == category else '✗ (%s)' % category
            time_since_last_epoch = timeSince(start)
            print('%d %d%% (%s) %.4f %s / %s %s' % 
                (epoch, epoch / epochs * 100, time_since_last_epoch,
                loss, line, guess, correct))

        if epoch % accumulate_loss_for_every == 0:
            all_losses.append(current_loss / accumulate_loss_for_every)
            current_loss = 0

    return all_losses

def evaluate(rnn, line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.shape[0]): # for each letter
        output, hidden = rnn(line_tensor[i], hidden)
    
    return output

def predict(rnn, text, n_top_predictions):
    print('\n> %s' % text)
    with torch.no_grad():
        output = evaluate(rnn, line_to_tensor(text))

        # get top predictions
        topv, topi = output.topk(n_top_predictions, 1, True)
        predictions = []

        for i in range(n_top_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

if __name__ == '__main__':
    download_input_data('data')
    
    r1, r2, r3 = load_lang_category_data('data/data/names/*.txt')

    lang_to_wrds_map = r1
    all_categories = r2
    n_categories = r3
    n_hidden = 128

    rnn = RNN(n_letters, n_hidden, n_categories)
    loss_fn = nn.NLLLoss()

    all_losses = train(all_categories, rnn, loss_fn)
    # plt.figure()
    # plt.plot(all_losses)
    # plt.show()

    if len(sys.argv) == 2:
        text = sys.argv[1]
    else:
        text = 'Hello'
    predict(rnn, text, 3)
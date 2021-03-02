########################################################
# Segmentation framework, test
########################################################

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import xlrd
import time
import cv2
import torch
import torchvision
import os
import tqdm
import sys
import random
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets.folder import make_dataset, VisionDataset
from skimage.util import random_noise
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set your GPU

## Addition data type and check raw data
ext=torchvision.datasets.folder.IMG_EXTENSIONS
torchvision.datasets.folder.IMG_EXTENSIONS=ext+('.raw',)
np.random.seed(0)
torch.manual_seed(0)

class SimpleNet(torch.nn.Module):
    def __init__(self, in_channel, out_channel, flatten_size):
        super(SimpleNet, self).__init__()
        # Encode
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=6)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(6, 12)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.fully_connected1 = self.fully_connected_block(flatten_size, 128)
        self.fully_connected2 = self.fully_connected_block(128, 128)
        self.fully_connected3 = self.fully_connected_block(128, out_channel)
        self.logistic = torch.nn.Sigmoid()

    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
        )
        return block

    def fully_connected_block(self, in_features, out_features):
        block = torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features),
            torch.nn.ReLU(),
        )
        return block

    def padding(self,layer_number):
        padding = torch.nn.ZeroPad2d((layer_number, layer_number, layer_number, layer_number, 0, 0, 0, 0))
        return padding

    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            cs = (bypass.size()[2] - upsampled.size()[2])//2
            ct = (bypass.size()[3] - upsampled.size()[3])//2
            cs_residual = (bypass.size()[2] - upsampled.size()[2])%2
            ct_residual = (bypass.size()[3] - upsampled.size()[3])%2
            bypass = torch.nn.functional.pad(bypass, [-cs-cs_residual, -cs,-ct-ct_residual, -ct])
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        m0, m1, m2, m3 = encode_pool2.shape
        encode_layer3 = encode_pool2.view(-1, m1 * m2 * m3)
        fcn1 = self.fully_connected1(encode_layer3)
        fcn2 = self.fully_connected2(fcn1)
        fcn3 = self.fully_connected3(fcn2)
        return fcn3

class UNet(torch.nn.Module):
    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2,
                                     padding=1, output_padding=1)
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
        )
        return block

    def bottleneck_block(self, in_channels, mid_channel_0, mid_channel_1,out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=3, in_channels=in_channels, out_channels=mid_channel_0, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel_0),
            torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel_0, out_channels=mid_channel_1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel_1),
            torch.nn.ConvTranspose2d(in_channels=mid_channel_1, out_channels= out_channels, kernel_size=kernel_size,
                                     stride=2, padding=1, output_padding=1)
        )
        return block

    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()

        # Encode
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=64)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_maxpool1 = torch.nn.AvgPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(64, 128)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_maxpool2 = torch.nn.AvgPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(128, 256)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_maxpool3 = torch.nn.AvgPool2d(kernel_size=2)

        # Bottleneck
        self.bottleneck = self.bottleneck_block(in_channels = 256, mid_channel_0 = 512, mid_channel_1 = 512,
                                                out_channels = 256)
        # Decode
        self.conv_decode3 = self.expansive_block(512, 256, 128)
        self.conv_decode2 = self.expansive_block(256, 128, 64)
        self.final_layer = self.final_block(128, 64, out_channel)

    def padding(self,layer_number):
        padding = torch.nn.ZeroPad2d((layer_number, layer_number, layer_number, layer_number, 0, 0, 0, 0))
        return padding

    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            cs = (bypass.size()[2] - upsampled.size()[2])//2
            ct = (bypass.size()[3] - upsampled.size()[3])//2
            cs_residual = (bypass.size()[2] - upsampled.size()[2])%2
            ct_residual = (bypass.size()[3] - upsampled.size()[3])%2
            bypass = F.pad(bypass, [-cs-cs_residual, -cs,-ct-ct_residual, -ct])
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)

        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)

        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)

        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)

        # Decode
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3, crop=True)
        cat_layer2 = self.conv_decode3(decode_block3)

        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2, crop=True)
        cat_layer1 = self.conv_decode2(decode_block2)

        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=True)
        final_layer = self.final_layer(decode_block1)
        return final_layer

class gray2color(object):
    def __init__(self):
        pass

    def __call__(self, gray):
        return np.tile(gray, (3, 1, 1)).transpose([1, 2, 0])

class LoadDataset(VisionDataset):
    def __init__(self, image_path, label_path, transform, image_extensions):
        super(LoadDataset, self).__init__(image_path, transform=transform,
                                            target_transform=None)
        self.image_path = image_path
        self.label_path = label_path
        self.transform = transform
        self.image_extensions = image_extensions
        ##
        classes, class_to_idx = self._find_classes(self.image_path)
        self.image_files = make_dataset(self.image_path, class_to_idx, extensions=self.image_extensions, is_valid_file=None)
        self.label_files = make_dataset(self.label_path, class_to_idx, extensions=self.image_extensions, is_valid_file=None)

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        seed = np.random.randint(2147483647)
        random.seed(seed)
        #Read the raw image
        image = image_loader(self.image_files[idx][0])
        label = label_loader(self.label_files[idx][0])
        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)
        return image, label

def read_raw(input_path,width,height,depth):
    if depth <= 8:
        type = np.uint8
    elif depth <= 16:
        type = np.uint16
    else:
        type = np.float32
    f = open(input_path, 'rb')  # only opens the file for reading
    img_arr = np.fromfile(f, dtype=type).reshape(-1,height, width)
    return img_arr

def read_data(dataset_path):
    raw_volume = read_raw(dataset_path, 64, 64, 16)
    # Normalization
    normalised_volume = (raw_volume - np.mean(raw_volume)) / np.std(raw_volume)
    return normalised_volume

def read_label(label_path):
    rb = xlrd.open_workbook(label_path)
    sheet = rb.sheet_by_index(0)
    one_hot_label= []
    single_label = []
    for i in range(0, sheet.nrows):
        label_number = sheet.row_values(i)
        one_hot_label.append(label_number[0:-1])
        single_label.append(int(label_number[-1]))
    return one_hot_label, single_label

def image_loader(path):
    with open(path, 'rb') as fd:
        byte = os.path.getsize(path)
        size = int(np.sqrt(byte))
        img = np.fromfile(fd, dtype=np.uint8, count=(size**2))
        img = img.reshape([size,size])
        img = img.astype(np.float32)
        img = cv2.resize(img, (224, 224))
    # return Image.fromarray(img)
        # Adding the noise
        blur = random_noise(img/255, mode='gaussian',var = 0.01)
    return Image.fromarray(blur*255)

def label_loader(path):
    with open(path, 'rb') as fd:
        byte = os.path.getsize(path)
        size = int(np.sqrt(byte))
        img = np.fromfile(fd, dtype=np.uint8, count=(size**2))
        img = img.reshape([size,size])
        img = img.astype(np.float32)
        img = cv2.resize(img, (224, 224))
    return Image.fromarray(img)

def Num2LabelName(labeldict,num):
    return [k for k, v in labeldict.items() if v == num][0]


def calculate_training_accuracy(label_ground_truth, label_result):
    rounded_label = np.round(label_result)
    label_mse = np.sum(rounded_label==label_ground_truth)/(label_result.shape[0]*label_result.shape[1])
    return label_mse


## Use simplenet to classify human body images
if __name__ == '__main__':
    # Read the data and label
    train_data_path = 'D:/Datastack/im_sub_4/segmentation1/training_image'
    train_label_path = 'D:/Datastack/im_sub_4/segmentation1/training_label'

    body_transform = torchvision.transforms.Compose([
        gray2color(),
        torchvision.transforms.ToTensor()])
    train_data = LoadDataset(image_path = train_data_path,
                             label_path = train_label_path,
                             transform = body_transform,
                             image_extensions=torchvision.datasets.folder.IMG_EXTENSIONS)

    train_data_loader = torch.utils.data.DataLoader(train_data,
                                                    batch_size=5,
                                                    shuffle=True,
                                                    worker_init_fn=lambda x: np.random.seed(0))

    # Training
    net = UNet(in_channel=3,out_channel=3).to(device)

    Nepoch = 10
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
    minimum_loss = 1e10
    loss_index = 0
    model_path = 'model_weights/unet_best_color_noised_test.pth'
    for epoch in tqdm.tqdm(range(Nepoch)):  # loop over the dataset multiple times
        for i, data in enumerate(train_data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)  # to GPU
            labels = labels.to(device)  # to GPU
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if loss.item() < minimum_loss:
            loss_index = epoch
            minimum_loss = loss.item()
            torch.save(net.state_dict(), model_path)
        print('[%d] loss: %.10f' % (epoch + 1, loss.item()))
        time.sleep(0.01)
    print("The best accuracy is in {}th epoch".format(loss_index+1))

    ## Validation
    net = UNet(in_channel=3,out_channel=3).to(device)
    model_path = 'model_weights/unet_best_color_noised_test.pth'
    net.load_state_dict(torch.load(model_path))
    #
    # # Validation on training dataset
    # test_data_path = 'D:/Datastack/im_sub_4/segmentation2/training_image'
    # test_label_path = 'D:/Datastack/im_sub_4/segmentation2/training_label'
    # Test_transform = torchvision.transforms.Compose([
    #     gray2color(),
    #     torchvision.transforms.ToTensor()
    # ])
    # test_data = LoadDataset(image_path=test_data_path,
    #                         label_path=test_label_path,
    #                         transform=body_transform,
    #                         image_extensions=torchvision.datasets.folder.IMG_EXTENSIONS)
    #
    # test_data_loader = torch.utils.data.DataLoader(test_data,
    #                                                batch_size=1,
    #                                                shuffle=True)
    # test_iter = iter(test_data_loader)
    # net.eval()
    # ImageN = len(test_data.image_files)
    # set_accuracy = 0
    # for n in tqdm.tqdm(range(ImageN)):
    #     testim, testlb = test_iter.next()
    #     ans = net(testim.to(device))
    #     ground_truth = np.squeeze(testlb.cpu().numpy())[0, :, :]
    #     result = np.squeeze(ans.cpu().detach().numpy())[0, :, :]
    #     input_image = np.squeeze(testim.cpu().numpy())[0, :, :]
    #     plt.figure('Compare')
    #     plt.subplot(1, 3, 1), plt.imshow(input_image, 'gray'), plt.gca().set_title('input image')
    #     plt.subplot(1, 3, 2), plt.imshow(result,'gray'),plt.gca().set_title('result')
    #     plt.subplot(1, 3, 3), plt.imshow(ground_truth, 'gray'), plt.gca().set_title('ground truth')
    #     plt.pause(0.1)
    #     frame_accuracy = calculate_training_accuracy(np.squeeze(testlb.cpu().numpy())[0, :, :],
    #                                                  np.squeeze(ans.cpu().detach().numpy())[0, :, :])
    #     set_accuracy += frame_accuracy
    # print('training accuracy is  {}'.format(set_accuracy / ImageN))

    # Validation on test dataset
    test_data_path = 'D:/Datastack/im_sub_4/segmentation1/test_image'
    test_label_path = 'D:/Datastack/im_sub_4/segmentation1/test_label'
    Test_transform = torchvision.transforms.Compose([
        gray2color(),
        torchvision.transforms.ToTensor()
    ])
    test_data = LoadDataset(image_path = test_data_path,
                            label_path = test_label_path,
                            transform = body_transform,
                            image_extensions=torchvision.datasets.folder.IMG_EXTENSIONS)

    test_data_loader = torch.utils.data.DataLoader(test_data,
                                                   batch_size=1,
                                                   shuffle=True)
    test_iter = iter(test_data_loader)
    net.eval()
    ImageN = len(test_data.image_files)
    set_accuracy = 0
    for n in tqdm.tqdm(range(ImageN)):
        testim, testlb = test_iter.next()
        ans = net(testim.to(device))
        ground_truth = np.squeeze(testlb.cpu().numpy())[0,:,:]
        result = np.squeeze(ans.cpu().detach().numpy())[0,:,:]
        input_image = np.squeeze(testim.cpu().numpy())[0,:,:]
        plt.figure('Compare')
        plt.subplot(1, 3, 1), plt.imshow(input_image, 'gray'), plt.gca().set_title('input image')
        plt.subplot(1, 3, 2), plt.imshow(result,'gray'),plt.gca().set_title('result')
        plt.subplot(1, 3, 3), plt.imshow(ground_truth, 'gray'), plt.gca().set_title('ground truth')
        plt.pause(0.1)
        frame_accuracy = calculate_training_accuracy(np.squeeze(testlb.cpu().numpy())[0,:,:],np.squeeze(ans.cpu().detach().numpy())[0,:,:])
        set_accuracy += frame_accuracy
    print('Validation accuracy is  {}'.format(set_accuracy/ImageN))
    plt.pause(0.1)

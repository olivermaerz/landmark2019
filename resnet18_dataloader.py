import torch
from torch.utils.data import Dataset, DataLoader
import os
import time
import copy
import pandas as pd
import unittest
from PIL import Image


import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LandMarkDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = data_dir
        self.transform = transform
        self.num_classes = self.df['landmark_id'].nunique()
        self.min_class_id = int(self.df['landmark_id'].min())
        self.max_class_id = int(self.df['landmark_id'].max())
        

    def __getitem__(self, idx):
        try:
            img_id = self.df.iloc[idx]['id'] + '.jpg'
            label = self.df.iloc[idx]['landmark_id']
            a, b, c, *_ = img_id
            img_name = os.path.join(self.root_dir, a, b, c, img_id)
            image = Image.open(img_name)
            if self.transform:
                image = self.transform(image)
            return image, label
        except:
            return None
        

    def __len__(self):
        return len(self.df)


    
def filtered_collate_fn(batch):
  # Skip errors. __get_item__ returns None if catches an exception.
  return torch.utils.data.dataloader.default_collate([x for x in batch if x is not None])



normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

dset = LandMarkDataset(csv_file='/home/ubuntu/image-retrieval/train.csv', data_dir='/home/ubuntu/image-retrieval/train', transform = transform)
loader = DataLoader(dset, num_workers=8, batch_size=64, shuffle=True, collate_fn=filtered_collate_fn)



def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            st_batch = time.time()
            for i_batch, (images, labels) in enumerate(loader):
                if (i_batch+1) % 1000 == 0:
                    print('Batch {} with time taken {:.4f} seconds'.format(i_batch, time.time() - st_batch))
                    st_batch = time.time()
                inputs = images.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(dset)
            epoch_acc = running_corrects.double() / len(dset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


num_classes = dset.num_classes
resnet_model18 = torchvision.models.resnet18(pretrained=True)
num_ftrs = resnet_model18.fc.in_features
resnet_model18.fc = nn.Linear(num_ftrs, num_classes)
resnet_model18 = resnet_model18.to(device)


criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(resnet_model18.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


resnet_model18_ft = train_model(resnet_model18, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=1)


# transformations for training and validation

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),             # resizes images to 224x224 pixels
    transforms.RandomHorizontalFlip(),         # randomly flips images horizontally for data augmentation
    transforms.ToTensor(),                     # converts PIL Image to a PyTorch tensor (scales pixels to [0,1])
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # normalizes using ImageNet statistics
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),             # resizes images for consistency
    transforms.ToTensor(),                     # converts images to tensor, pixel values now between [0,1].
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # uses same normalization as training
                         std=[0.229, 0.224, 0.225])
])

# assigns labels to images based on subfolder names (= class names)
train_dataset = ImageFolder(root=train_dir, transform=train_transforms)
val_dataset   = ImageFolder(root=test_dir, transform=val_transforms)

# data loaders to handle batching and shuffling
train_loader = DataLoader(train_dataset, batch_size= batch, shuffle=True, num_workers=2) # shuffle in order not to get overfitting
val_loader   = DataLoader(val_dataset, batch_size= batch, shuffle=False, num_workers=2) #num_workers = 2: 2 subprocesses to speed things up

#download model with pretrained weights
model = timm.create_model("efficientnet_b0", pretrained=True) 

#adapts the model to classify images into NumClasses instead of the default 1000 ImageNet categories

in_features = model.classifier.in_features             # number of input features for the classifier layer
model.classifier = nn.Linear(in_features, NumClasses)  # replaces with a new linear layer

model = model.to(device)


#loss function
criterion = nn.CrossEntropyLoss()

#optimizer updates model weights during training
optimizer = optim.Adam(model.parameters(), lr=LearningRate, weight_decay=WeightDecay)

# learning rate schedule, reduces LR after every 5 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=Gamma)
# after every 5 epochs, the LR is multiplied by gamma=0.1

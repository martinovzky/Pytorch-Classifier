#we train over 10 epochs

for epochs in range(number_epochs):
  model.train() #set the model to training mode: activates dropout (prevents overfitting), batchnorm, etc.
  running_loss = 0.0 #loss per epoch
  correct_samples = 0  #i.e. predicted label matches with real label
  total_sampled = 0 

  for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device) #moves these batches (tensors) them where the model is 
    optimizer.zero_grad() #sets the gradients from previous steps to 0 

    output = model(images) #forward pass, computes predictions (confidence scocres, not probabilities), and stores them in a tensor
    loss = criterion(output, labels) #computes loss between label predictions and actual labels
    
    #backprobagation : computes gradients of loss wrt the model's weights,this traverses the model backwards and probagates the outpout back to the input
    loss.backward()  

    optimizer.step() #updates model weights

    # model performance

    running_loss += loss.item() * images.size(0)       #acumulated loss (loss per batch * number of images)
    _, preds = torch.max(output, 1)  #gets predicted class (preds) by taking the sample's max confidence score (_, is then ignored in order to later compute element-wise comparaison)
    correct_samples += (preds == labels).sum().item()  #counts correct predictions
    total_sampled += labels.size(0)                    #counts total samples

  epoch_loss = running_loss / total_sampled            #average loss for the epoch
  epoch_acc  = correct_samples / total_sampled         #accuracy for the epoch

  print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")

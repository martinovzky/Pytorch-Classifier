#we train over 15 epochs

train_losses = []
val_losses = [] 

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

    running_loss += loss.item() * images.size(0)  #acumulated loss (loss per batch * number of images)
    _, preds = torch.max(output, 1)  #gets predicted class (preds) by taking the sample's max confidence score (_, is then ignored in order to later compute element-wise comparaison)
    correct_samples += (preds == labels).sum().item()  #counts correct predictions
    total_sampled += labels.size(0)  #counts total samples

  epoch_loss = running_loss / total_sampled  #average loss for the epoch
  train_losses.append(epoch_loss)

  epoch_acc  = correct_samples / total_sampled         #accuracy for the epoch

  print(f"Epoch {epochs+1}/{number_epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")



  model.eval() #sets to validation mode
  val_loss = 0.0
  val_correct = 0
  val_total = 0

  #disables gradient calculations for validation to save memory and computation

  with torch.no_grad():
    for images, labels in val_loader:
      images, labels = images.to(device), labels.to(device)
      output = model(images)           #forward pass
      loss = criterion(output, labels) #validation loss

      val_loss += loss.item() * images.size(0)  #acumulated loss (loss per batch * number of images)
      _, preds = torch.max(output,1)
      val_correct += (preds == labels).sum().item() # item() converts tensor to a scalar
      val_total += labels.size(0)

  val_loss = val_loss / val_total #average validation loss
  val_losses.append(val_loss)
  val_acc = val_correct / val_total #average validation accuracy

  print(f"Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.4f}")



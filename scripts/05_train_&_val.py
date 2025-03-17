#we train over 8 epochs

train_losses = []
val_losses = [] 

for epochs in range(number_epochs):
  model.train() #set the model to training mode: activates dropout (prevents overfitting), batchnorm, etc.
  running_loss = 0.0 #loss per epoch
  correct_samples = 0  #i.e. predicted label matches with real label
  total_sampled = 0

  for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device) 
    optimizer.zero_grad() #sets the gradients from previous steps to 0

    output = model(images) #forward pass
    loss = criterion(output, labels) #loss 

    #backprop
    loss.backward()

    optimizer.step() #updates model weights

    #model performance

    running_loss += loss.item() * images.size(0)  #acumulated loss (loss per batch * number of images)
    _, preds = torch.max(output, 1)  #gets predicted class (preds) by taking the sample's max confidence score (_, is then ignored in order to later compute element-wise comparaison)
    correct_samples += (preds == labels).sum().item()  
    total_sampled += labels.size(0)  
    
  epoch_loss = running_loss / total_sampled 
  train_losses.append(epoch_loss)

  epoch_acc  = correct_samples / total_sampled        

  print(f"Epoch {epochs+1}/{number_epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")



  model.eval() 
  val_loss = 0.0
  val_correct = 0
  val_total = 0

  #disables gradient calculations for validation to save memory and computation

  with torch.no_grad():
    for images, labels in val_loader:
      images, labels = images.to(device), labels.to(device)
      output = model(images)           
      loss = criterion(output, labels) 

      val_loss += loss.item() * images.size(0)  #acumulated loss (loss per batch * number of images)
      _, preds = torch.max(output,1)
      val_correct += (preds == labels).sum().item() #item() converts tensor to a scalar
      val_total += labels.size(0)

  val_loss = val_loss / val_total 
  val_losses.append(val_loss)
  val_acc = val_correct / val_total 

  print(f"Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.4f}")



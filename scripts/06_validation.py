model.eval() #set to validation mode
val_loss = 0.0
val_correct = 0
val_total = 0 

#disable gradient calculations for validation to save memory and computation

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
val_acc = val_correct / val_total #average validation accuracy

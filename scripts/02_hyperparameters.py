batch = 32               # of images / training batch
number_epochs = 8       # tot number of training epochs (full passes over the dataset)
LearningRate = 1e-3      # learning rate for the optimizer (how much should model weights be updated after each batch)
Gamma = 0.1              # factor by which the learning rate will be reduced, helpw the model converge more smoothly and avoid overshooting local minima
NumClasses = 67          # MIT‑67 dataset has 67 scene categories
device = "cuda" if torch.cuda.is_available() else "cpu"


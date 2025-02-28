batch = 32               # of images / training batch
epochs = 10              # tot number of training epochs (full passes over the dataset)
LearningRate = 1e-3      # learning rate for the optimizer (how much should model weights be updated after each batch )
NumClasses = 67          # MIT‑67 dataset has 67 scene categories
device = "cuda" if torch.cuda.is_available() else "cpu"

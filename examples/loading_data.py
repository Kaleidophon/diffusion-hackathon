from torch.utils.data import DataLoader

from dataset.sprites_dataset import SpritesDataset

# This is a torch.utils.data.Dataset object
dataset = SpritesDataset()

# Using it, we could make a simple dataloader:
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Now we can iterate over the dataloader to get batches of data
for batch in dataloader:
    print(batch.shape)  # torch.Size([32, 3, 16, 16])

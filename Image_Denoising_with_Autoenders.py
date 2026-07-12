import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + 0.5 * torch.randn_like(x)),  # Add Gaussian noise
    transforms.Lambda(lambda x: torch.clamp(x, 0., 1.)) 
])

noisy_images=datasets.MNIST(root="./data",train=True,transform=transform,download=True)
clean_images=datasets.MNIST(root="./data",train=True,transform=transforms.ToTensor(),download=True)

noisy_image=DataLoader(noisy_images,batch_size=128,shuffle=False)
clean_image=DataLoader(clean_images,batch_size=128,shuffle=False)

data_loader=zip(noisy_image,clean_image)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.encoder=nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU()
        )
        self.decoder=nn.Sequential(
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,28*28),
            nn.Sigmoid()
        )
    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x.view(-1,1,28,28)

model=AutoEncoder()
criterion=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

for epoch in range(5):
    data_loader=zip(noisy_image,clean_image)
    for (noisy_imgs,_),(clean_imgs,_) in data_loader:
        outputs=model(noisy_imgs)
        loss=criterion(outputs,clean_imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"loss is {loss} in epoch {epoch+1}")


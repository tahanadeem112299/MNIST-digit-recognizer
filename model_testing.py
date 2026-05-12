import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam 
from google.colab import files
from PIL import Image
import matplotlib.pyplot as plt 

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
upload=files.upload()
file=list(upload.keys())[0]

image=Image.open(file)

plt.imshow(image,cmap='gray')
plt.title("Image")
plt.show()

transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x:1-x),
    transforms.Normalize((0.5,),(0.5,))
])

class CNN(nn.Module): 
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(1,32,3,1,1)
        self.norm1=nn.BatchNorm2d(32)

        self.conv2=nn.Conv2d(32,64,3,1,1)
        self.norm2=nn.BatchNorm2d(64)

        self.conv3=nn.Conv2d(64,128,3,1,1)
        self.norm3=nn.BatchNorm2d(128)

        self.relu=nn.ReLU()
        self.pool=nn.MaxPool2d(2,2)

        self.dropout=nn.Dropout(0.25)

        self.fc1=nn.Linear(128*3*3, 512)
        self.fc2=nn.Linear(512,10)
    def forward(self,x):
        #block1
        x=self.pool(self.relu(self.norm1(self.conv1(x))))

        #block2
        x=self.pool(self.relu(self.norm2(self.conv2(x))))

        #block3
        x=self.pool(self.relu(self.norm3(self.conv3(x))))

        x=x.view(x.size(0),-1)
        x=self.dropout(x)

        x=self.relu(self.fc1(x))
        x=self.fc2(x)
        return x
model=CNN().to(device)
model.load_state_dict(torch.load("own_mnist_model.pth",map_location=device))
model.eval()
image=transform(image).unsqueeze(0).to(device)
with torch.no_grad():
    output=model(image)
    __, prediction= torch.max(output,1)
print("The output is: ", prediction.item())
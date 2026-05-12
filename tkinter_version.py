import torch 
import torch.nn as nn 
from torchvision import transforms
from PIL import Image, ImageOps
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        x=self.pool(self.relu(self.norm1(self.conv1(x))))
        x=self.pool(self.relu(self.norm2(self.conv2(x))))
        x=self.pool(self.relu(self.norm3(self.conv3(x))))

        x=x.view(x.size(0),-1)
        x=self.dropout(x)

        x=self.relu(self.fc1(x))
        x=self.fc2(x)

        return x

model=CNN().to(device)
model.load_state_dict(torch.load("own_mnist_model.pth",map_location=device))
model.eval()

root=tk.Tk()
root.title("MNIST Digit Predictor")
root.geometry("400x500")

image_label=tk.Label(root)
image_label.pack(pady=20)

result_label=tk.Label(root,text="",font=("Arial",20))
result_label.pack(pady=20)

def upload_image():
    file=filedialog.askopenfilename()

    if file:
        image=Image.open(file)

        display_image=image.resize((200,200))
        photo=ImageTk.PhotoImage(display_image)

        image_label.config(image=photo)
        image_label.image=photo

        img=transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output=model(img)
            __,prediction=torch.max(output,1)

        result_label.config(text=f"Prediction: {prediction.item()}")

button=tk.Button(root,text="Upload Image",command=upload_image,font=("Arial",15))
button.pack(pady=20)

root.mainloop()
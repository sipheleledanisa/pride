import torch
import cv2
import torchvision.transforms as transforms

import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter as writer
import matplotlib.image as mping
plt.rcParams['animation.ffmpeg_path'] = "C:\\ffmpeg-2024-02-15-git-a2cfd6062c-full_build\\bin\\ffmpeg.exe"



class DeepAutoencoder(torch.nn.Module): 
    def __init__(self): 
        super().__init__()         
        self.encoder = torch.nn.Sequential( 
            torch.nn.Linear(31488, 128), 
            torch.nn.ReLU(), 
            torch.nn.Linear(128, 64), 
            torch.nn.ReLU() 
        ) 
          
        self.decoder = torch.nn.Sequential( 
            torch.nn.Linear(64, 128), 
            torch.nn.ReLU(), 
            torch.nn.Linear(128, 31488), 
            torch.nn.Sigmoid() 
        ) 
  
    def forward(self, x): 
        encoded = self.encoder(x) 
        decoded = self.decoder(encoded) 
        return decoded 



img = cv2.imread('pride.png') 

model = DeepAutoencoder()
optim = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = torch.nn.MSELoss()

N=1000

#data = transform_norm(img)#.flatten()
data = torch.from_numpy(img)/255.0
full_history = []

# Here, we use enumerate(training_loader) instead of
# iter(training_loader) so that we can track the batch
# index and do some intra-epoch reporting
for i in range(N):
    # Every data instance is an input + label pair
    

    # Zero your gradients for every batch!
    optim.zero_grad()

    # Make predictions for this batch
    outputs = model(data.flatten().unsqueeze(0)).squeeze()
    outputs = outputs.reshape(img.shape)
    full_history.append(outputs.detach().numpy())
    


    # Compute the loss and its gradients
    loss = loss_fn(outputs, data)
    loss.backward()

    # Adjust learning weights
    optim.step()
    if i%50==0:
        print(loss.item())


# img_mod= outputs.detach().numpy()#*255

# plt.imshow(cv2.cvtColor(img_mod, cv2.COLOR_BGR2RGB))
# plt.show()

# Animating function.
def animate(i):
    """
    This will animate the results using 'i' as the frame index.
    We access needed variables through their global scope here, because it is not clear how to pass these through matplotlib's FuncAnimation.
    """
    ax.clear()
    #Make a 3D plot
    ax.set_title('Rainbow')
    # Plot distribution for context.
    img_item = full_history[i]
    img_item = cv2.cvtColor(img_item, cv2.COLOR_BGR2RGB)

    
    # Plot iterations with points using the full history
    ax.imshow(img_item)
    ax.grid(False)

# Initialize figure for 3D plotting   
fig, ax = plt.subplots(1, 1,figsize=(10,10))
# Do animation computations
ani = FuncAnimation(fig, animate, interval=50,frames=1000)
plt.show()

# If one wants to save, uncomment.
FFwriter = writer(fps=10)
ani.save('C:\\Users\\siphe\\OneDrive\\Documents\\Python Scripts\pride\\animation.mp4', writer = FFwriter)




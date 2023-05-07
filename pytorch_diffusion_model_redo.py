import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
device = torch.device('cuda')
from matplotlib import pyplot as plt

dataset = torchvision.datasets.MNIST(root = 'mnist/', train = True, download = True, transform = torchvision.transforms.ToTensor())

#print(device)


#DATASET

train_dataloader = DataLoader(dataset, batch_size = 8, shuffle = True)
x, y = next(iter(train_dataloader))


#CORRUPTION

def corrupt(x, amount):
	noise = torch.rand_like(x)
	amount = amount.view(-1, 1, 1, 1)
	noisy_x = x*(1-amount) + noise*amount
	return noisy_x


class BasicUNet(nn.Module):
	def __init__(self, in_channels = 1, out_channels = 1):
		super().__init__()
		self.down_layers = torch.nn.ModuleList([
		nn.Conv2d(in_channels, 32, kernel_size = 5, padding = 2),
		nn.Conv2d(32, 64, kernel_size = 5, padding = 2),
		nn.Conv2d(64, 64, kernel_size = 5, padding = 2)
		])
		self.up_layers = torch.nn.ModuleList([
		nn.Conv2d(64, 64, kernel_size = 5, padding = 2),
		nn.Conv2d(64, 32, kernel_size = 5, padding = 2),
		nn.Conv2d(32, out_channels, kernel_size = 5, padding = 2)
		])

		"""self.down_layers = torch.nn.ModuleList([
			nn.Conv2d(in_channels, 32, kernel_size = 5, padding = 2),
			nn.Conv2d(32, 64, kernel_size = 5, padding = 2),
			nn.Conv2d(64, 128, kernel_size = 5, padding = 2),
			nn.Conv2d(128, 128, kernel_size = 5, padding =2)])
		self.up_layers = torch.nn.ModuleList([
			nn.Conv2d(128, 128, kernel_size = 5, padding = 2),
			nn.Conv2d(128, 64, kernel_size = 5, padding = 2),
			nn.Conv2d(64, 32, kernel_size = 5, padding = 2),
			nn.Conv2d(32, out_channels, kernel_size = 5, padding = 2)])"""

		self.act = nn.SiLU()  #ACTIVATION FUNCTION
		self.downscale = nn.MaxPool2d(2)
		self.upscale = nn.Upsample(scale_factor = 2)


	def forward(self, x):
		h = []   #FOR SKIPPING STEPS IN THE BACKWARD PROCESS
		for i, l in enumerate(self.down_layers):
			x = self.act(l(x))
			if i<2:
				h.append(x)
				x = self.downscale(x)

		for i, l in enumerate(self.up_layers):
			if i>0:
				x = self.upscale(x)
				x += h.pop()
				x = self.act(l(x))
		return x





n_epochs = 3

net = BasicUNet()
net.to(device)

loss_fn = nn.MSELoss()

opt = torch.optim.Adam(net.parameters())

losses = []

for epoch in range(n_epochs):
	for x, y in train_dataloader:
		x = x.to(device)
		noise_amount = torch.rand(x.shape[0]).to(device)
		
		noisy_x = corrupt(x, noise_amount)

		pred = net(noisy_x)

		loss = loss_fn(pred, x)

		opt.zero_grad()
		loss.backward()
		opt.step()

		losses.append(loss.item())

	avg_loss = sum(losses[-len(train_dataloader):])/len(train_dataloader)
	print(f'Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}')

plt.plot(losses)
plt.ylim(0, 0.1);
plt.show()


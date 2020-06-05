
import torch
import torch.nn as nn

class LinearNet(nn.Module):

	def __init__(self):
		super(LinearNet, self).__init__()
		self.linear_4 = nn.Linear(in_features = 28*28, out_features = 64, bias = True)
		self.reLU_7 = nn.ReLU(inplace = False)
		self.linear_6 = nn.Linear(in_features = 64, out_features = 128, bias = True)
		self.reLU_8 = nn.ReLU(inplace = False)
		self.linear_5 = nn.Linear(in_features = 128, out_features = 10, bias = True)

	def forward(self, x_para_1):
		x_reshape_3 = torch.reshape(x_para_1,shape = (-1, 28*28))
		x_linear_4 = self.linear_4(x_reshape_3)
		x_reLU_7 = self.reLU_7(x_linear_4)
		x_linear_6 = self.linear_6(x_reLU_7)
		x_reLU_8 = self.reLU_8(x_linear_6)
		x_linear_5 = self.linear_5(x_reLU_8)
		return x_linear_5


import torch
import torch.nn as nn

class ExampleNet(nn.Module):

	def __init__(self):
		super(ExampleNet, self).__init__()
		self.conv2d_4 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True)
		self.reLU_14 = nn.ReLU(inplace = False)
		self.maxPool2D_5 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, return_indices = False, ceil_mode = False)
		self.conv2d_6 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True)
		self.reLU_15 = nn.ReLU(inplace = False)
		self.maxPool2D_7 = nn.MaxPool2d(kernel_size = 2, stride = None, padding = 0, dilation = 1, return_indices = False, ceil_mode = False)
		self.conv2d_16 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True)
		self.reLU_17 = nn.ReLU(inplace = False)
		self.linear_9 = nn.Linear(in_features = 7*7*32, out_features = 120, bias = True)
		self.reLU_12 = nn.ReLU(inplace = False)
		self.linear_10 = nn.Linear(in_features = 120, out_features = 84, bias = True)
		self.reLU_13 = nn.ReLU(inplace = False)
		self.linear_11 = nn.Linear(in_features = 84, out_features = 10, bias = True)

	def forward(self, x_para_2):
		x_conv2d_4 = self.conv2d_4(x_para_2)
		x_reLU_14 = self.reLU_14(x_conv2d_4)
		x_maxPool2D_5 = self.maxPool2D_5(x_reLU_14)
		x_conv2d_6 = self.conv2d_6(x_maxPool2D_5)
		x_reLU_15 = self.reLU_15(x_conv2d_6)
		x_maxPool2D_7 = self.maxPool2D_7(x_reLU_15)
		x_conv2d_16 = self.conv2d_16(x_maxPool2D_7)
		x_reLU_17 = self.reLU_17(x_conv2d_16)
		x_reshape_8 = torch.reshape(x_reLU_17,shape = (-1, 7*7*32))
		x_linear_9 = self.linear_9(x_reshape_8)
		x_reLU_12 = self.reLU_12(x_linear_9)
		x_linear_10 = self.linear_10(x_reLU_12)
		x_reLU_13 = self.reLU_13(x_linear_10)
		x_linear_11 = self.linear_11(x_reLU_13)
		return x_linear_11

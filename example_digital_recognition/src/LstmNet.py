
import torch
import torch.nn as nn

class LstmNet(nn.Module):

	def __init__(self):
		super(LstmNet, self).__init__()
		self.lstm_3 = nn.LSTM(input_size = 28, hidden_size = 64, num_layers = 2, bias = True, batch_first = True)
		self.linear_4 = nn.Linear(in_features = 64, out_features = 10, bias = True)

	def forward(self, x_para_1):
		x_reshape_8 = torch.reshape(x_para_1,shape = (-1,28,28))
		x_lstm_3 = self.lstm_3(x_reshape_8)
		x_slice_9 = x_lstm_3[0]
		x_linear_4 = self.linear_4(x_slice_9)
		x_slice_7 = x_linear_4[:, -1, :]
		return x_slice_7

import torch.nn as nn


class Classifier(nn.Module):
	def __init__(self):
		super().__init__()
		self.encoder = nn.Sequential(
			nn.Linear(65, 64),#[BS 65]->[BS 128]
			nn.LeakyReLU(),
			nn.Linear(64, 32),  # [BS 128 40]->[BS 128 160]
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(32, 32),  # [BS 128 40]->[BS 128 160]
			nn.LeakyReLU(),
			nn.Linear(32, 32),  # [BS 128 40]->[BS 128 160]
			nn.TransformerEncoderLayer(d_model=32, dim_feedforward=64, nhead=4, batch_first=True),#[BS  65] SELF-ATTENION + 2 LINER  RESNET
		)
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(32, 128),
			nn.BatchNorm1d(128),
			nn.LeakyReLU(),
			nn.Linear(128, 12),
		)

	def forward(self, input):
		mid = self.encoder(input)
		mid = mid.mean(dim=1)#[BS 128]
		out = self.classifier(mid)
		return out
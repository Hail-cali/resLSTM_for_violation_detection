#https://github.com/pranoyr/cnn-lstm.git
# train &  validate setting

import pandas as pd
import numpy as np
from utils.data_loader import *
from models.feature_net import *
import torch.nn as nn
import torch.optim as optim
import tensorboardX
from opts import parse_opts
from torch.utils import data
from opts import parse_opts
from set_train import *
from set_validate import *
from utils.make_plot import *


def resume_model(opt, model, optimizer):
	""" Resume model
	"""
	checkpoint = torch.load(opt.resume_path)
	model.load_state_dict(checkpoint['state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	print("Model Restored from Epoch {}".format(checkpoint['epoch']))
	start_epoch = checkpoint['epoch'] + 1
	return start_epoch


def main():
	# DPATH = '../dataset'
	DPATH = '../data'
	opt = parse_opts()
	# device = torch.device(f"cuda:{opt.gpu}" if opt.use_cuda else "cpu")

	# tensorboard
	summary_writer = tensorboardX.SummaryWriter(log_dir='../tf_logs')

	device = 'cuda'
	print(device, 'use')
	# use loader

	if DPATH == '../data':
		loader = DataLoader(path=DPATH, img_resize=True)
	else:
		loader = DataLoader(path=DPATH)
	# loader = DataLoader(path=DPATH, test_mode=True)

	# data set
	X, y = loader.make_frame(mode='extract', fc_layers=200, device=device)
	total_data = myDataset(x=X, y=y)
	train, val = data.random_split(total_data,
								   [int(len(total_data) * 0.8), len(total_data) - int(len(total_data) * 0.8)])

	print(type(train))
	train_loader = data.DataLoader(train, batch_size=30, shuffle=True)
	val_loader = data.DataLoader(val, batch_size=30, shuffle=True)

	# set model
	model = ResLSTM()
	model.to(device)

	# USE_CUDA = torch.cuda.is_available()
	# DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

	# optimizer
	criterion = nn.CrossEntropyLoss()
	# criterion = nn.BCEWithLogitsLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)

	if opt.resume_path:
		start_epoch = resume_model(opt, model, optimizer)
	else:
		start_epoch = 1
	# start training

	train_loss_l = []
	val_loss_l = []
	train_acc_l = []
	val_acc_l = []
	plot_epoch_l = []


	for epoch in range(start_epoch, opt.n_epochs + 1):
			train_loss, train_acc = train_epoch(
				model, train_loader, criterion, optimizer, epoch, opt.log_interval, device)
			val_loss, val_acc = val_epoch(
				model, val_loader, criterion, device)

			# saving weights to checkpoint
			if (epoch) % opt.save_interval == 0:
				# scheduler.step(val_loss)
				# write summary
				summary_writer.add_scalar(
					'losses/train_loss', train_loss, global_step=epoch)
				summary_writer.add_scalar(
					'losses/val_loss', val_loss, global_step=epoch)
				summary_writer.add_scalar(
					'acc/train_acc', train_acc * 100, global_step=epoch)
				summary_writer.add_scalar(
					'acc/val_acc', val_acc * 100, global_step=epoch)

				train_loss_l.append(train_loss)
				val_loss_l.append(val_loss)
				train_acc_l.append(train_acc)
				val_acc_l.append(val_acc)
				plot_epoch_l.append(epoch)

				state = {'epoch': epoch, 'state_dict': model.state_dict(),
						 'optimizer_state_dict': optimizer.state_dict()}
				torch.save(state, os.path.join('../snapshots', f'{opt.model}-Epoch-{epoch}-Loss-{val_loss}.pth'))
				print("Epoch {} model saved!\n".format(epoch))

	if DPATH[3:] == 'data':
		dataset_name = 'SCFD'
	else:
		dataset_name = 'Aihub'

	name = f'A_{opt.model}_4_{dataset_name}'
	summary_writer.close()
	loss_plot(train_loss_l, val_loss_l, plot_epoch_l, name)
	acc_plot(train_acc_l, val_acc_l, plot_epoch_l, name)

if __name__ == "__main__":
	main()


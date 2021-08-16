import pandas as pd
import numpy as np
from utils.data_loader import *
from models.feature_net import *
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import tensorboardX
from opts import parse_opts
DPATH = '../data/fight'



loader = DataLoader(path=DPATH)
print(loader.file_list)
total_frame = loader.make_frame(mode='train')

#print(f'total_frame len: {len(total_frame)}')
#print([len(frames) for frames in total_frame])
#print([frames[-1].shape for frames in total_frame])


# get_global_opt
opt = parse_opts()

# default cpu (gpu=0)
model = FeatureNet()
device = torch.device(f"cuda:{opt.gpu}" if opt.use_cuda else "cpu")
model.to(device)

# tensorboard
summary_writer = tensorboardX.SummaryWriter(log_dir='../tf_logs')

# optimizer
criterion = nn.CrossEntropyLoss(reduction='sum')
# optimizer = optim.SGD(model.parameters(), lr=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.001)



# sample settings
sample = total_frame[10]
#x = model.forward(sample)
#print(x)




epoch = 10
y = torch.LongTensor([1])
#print(f'result : {summary(model)}')
model.train()
for t in range(epoch):

    y_pred = model.forward(sample)
    loss = criterion(y_pred, y)

    if t % 2000 == 1999:
        print(f'step {t} | loss : {loss.item()}')

    print(f'step {t} || loss : {loss.item()}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# # start training
# 	for epoch in range(start_epoch, opt.n_epochs + 1):
# 		train_loss, train_acc = train_epoch(
# 			model, train_loader, criterion, optimizer, epoch, opt.log_interval, device)
# 		val_loss, val_acc = val_epoch(
# 			model, val_loader, criterion, device)
#
# 		# saving weights to checkpoint
# 		if (epoch) % opt.save_interval == 0:
# 			# scheduler.step(val_loss)
# 			# write summary
# 			summary_writer.add_scalar(
# 				'losses/train_loss', train_loss, global_step=epoch)
# 			summary_writer.add_scalar(
# 				'losses/val_loss', val_loss, global_step=epoch)
# 			summary_writer.add_scalar(
# 				'acc/train_acc', train_acc * 100, global_step=epoch)
# 			summary_writer.add_scalar(
# 				'acc/val_acc', val_acc * 100, global_step=epoch)
#
# 			state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
# 			torch.save(state, os.path.join('snapshots', f'{opt.model}-Epoch-{epoch}-Loss-{val_loss}.pth'))
# 			print("Epoch {} model saved!\n".format(epoch))
#
#





print()
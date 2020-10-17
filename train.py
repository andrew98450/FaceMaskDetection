import torch
from model import *

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_set, 
    batch_size=100,
    shuffle=True)

facemasknet = FaceMaskNet()
facemasknet = facemasknet.cuda()

epoch = 5
optim = torch.optim.Adam(facemasknet.parameters(), lr=0.001)
loss_f = torch.nn.NLLLoss()

for i in range(1, epoch + 1):
    acc = 0.0
    loss = 0.0
    step = 0
    total_size = 0
    batch_size = train_dataloader.batch_size
    for inputs, labels in train_dataloader:
        
        inputs = inputs.float().cuda()
        labels = labels.long().cuda()
        
        outputs = torch.log(facemasknet(inputs))
        cross_loss = loss_f(outputs, labels)
        
        optim.zero_grad()
        cross_loss.backward()
        optim.step()
        
        acc += torch.argmax(outputs, 1).eq(labels).sum().item()
        loss += cross_loss.item()
        
        step += 1
        total_size += batch_size
        
    acc /= total_size
    loss /= step
    print("[+] Epoch: %d Acc: %.4f Loss: %.4f" % (i, acc, loss))

facemasknet = facemasknet.cpu()

torch.save(facemasknet, "facemask_model.pth")
        
        
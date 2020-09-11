import numpy as np
import torch 
import torch.nn as nn
from netmodel import *
import argparse
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from multiprocessing import Process
import multiprocessing



def train_net(args, model, device, train, optimizer, epoch):
    model.train()
    n_example = 0
    example = 0

    for i in train:
        example += len(i[0])

    
    for batch_idx in range(0, len(train)):
        data, target = train[batch_idx][0], train[batch_idx][1]
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        n_example += len(data)
    
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, n_example, example,
                100 * n_example / example, loss.item()))


def test_net(model, device, test):
    model.eval()
    n_example = 0
    example = 0
    for i in test:
        example += len(i[0])

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for t in test:
            data, target = t[0], t[1]
            output = model(data)
            test_loss = F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim = 1, keepdim = True)
            
            h = torch.eq(pred, target)

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, example,
        100. * correct / example))
    


def task_0(args, device, train_loader, test_loader):
    # train 0 as negative, 1 as positive
    print("Preparing data for task 0...")
    train = []
    for batch_idx, (data, target) in enumerate(train_loader):
        
        flag0 = True
        flag1 = True
        ta = torch.empty([0])
        da = torch.empty([0])
        for i in range(0, len(target)):
            if target[i]==0:
                target[i] = 1
                t=target[i].numpy().tolist()
                d=data[i].numpy().tolist()
                
                if flag0 and flag1:
                    ta = torch.tensor([t])
                    da = torch.tensor([d])
                    flag0 = False
                else:
                    ta = torch.cat((ta, torch.tensor([t])), 0)
                    da = torch.cat((da, torch.tensor([d])), 0)

            elif target[i]==1:
                target[i] = 0
                t=target[i].numpy().tolist()
                d=data[i].numpy().tolist()
                
                if flag0 and flag1:
                    ta = torch.tensor([t])
                    da = torch.tensor([d])
                    flag1 = False
                else:
                    ta = torch.cat((ta, torch.tensor([t])), 0)
                    da = torch.cat((da, torch.tensor([d])), 0)

        
        if da.size() == torch.Size([0]):
            continue
        train.append([da, ta])


    test = []
    for batch_idx, (data, target) in enumerate(test_loader):
        
        flag0 = True
        flag1 = True
        ta = torch.empty([0])
        da = torch.empty([0])
        for i in range(0, len(target)):
            if target[i]==0:
                target[i] = 1
                t=target[i].numpy().tolist()
                d=data[i].numpy().tolist()
                
                if flag0 and flag1:
                    ta = torch.tensor([t])
                    da = torch.tensor([d])
                    flag0 = False
                else:
                    ta = torch.cat((ta, torch.tensor([t])), 0)
                    da = torch.cat((da, torch.tensor([d])), 0)

            # elif target[i]==0:
            #     target[i] = 1
            #     t=target[i].numpy().tolist()
            #     d=data[i].numpy().tolist()
                
            #     if flag0 and flag1 :
            #         ta = torch.tensor([t])
            #         da = torch.tensor([d])
            #         flag1 = False
            #     else:
            #         ta = torch.cat((ta, torch.tensor([t])), 0)
            #         da = torch.cat((da, torch.tensor([d])), 0)

        if da.size() == torch.Size([0]):   
            continue
        test.append([da, ta])

    
    print("Data for task 0 prepared!")
    print("Now begin task 0...")

    # train/test: [batch_idx] [0=data, 1=target] [tensor/label]

    model = LeNet5(outdim=30).to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        print("Now Train f_0...")
        train_net(args, model, device, train, optimizer, epoch)
        
        print("Now Test f_0...")
        test_net(model, device, test)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "./nnmodels/f_0.pt")

    return 


def task_1(args, device, train_loader, test_loader):
    # train 1 as positive, 0 as negative
    print("Preparing data for task 1...")
    train = []
    for batch_idx, (data, target) in enumerate(train_loader):
        
        flag0 = True
        flag1 = True
        ta = torch.empty([0])
        da = torch.empty([0])
        for i in range(0, len(target)):
            if target[i]==1:
                target[i] = 1
                t=target[i].numpy().tolist()
                d=data[i].numpy().tolist()
                
                if flag0 and flag1:
                    ta = torch.tensor([t])
                    da = torch.tensor([d])
                    flag0 = False
                else:
                    ta = torch.cat((ta, torch.tensor([t])), 0)
                    da = torch.cat((da, torch.tensor([d])), 0)

            elif target[i]==0:
                target[i] = 0
                t=target[i].numpy().tolist()
                d=data[i].numpy().tolist()
                
                if flag0 and flag1:
                    ta = torch.tensor([t])
                    da = torch.tensor([d])
                    flag1 = False
                else:
                    ta = torch.cat((ta, torch.tensor([t])), 0)
                    da = torch.cat((da, torch.tensor([d])), 0)

        
        if da.size() == torch.Size([0]):
            continue
        train.append([da, ta])


    test = []
    for batch_idx, (data, target) in enumerate(test_loader):
        
        flag0 = True
        flag1 = True
        ta = torch.empty([0])
        da = torch.empty([0])
        for i in range(0, len(target)):
            if target[i]==1:
                target[i] = 0
                t=target[i].numpy().tolist()
                d=data[i].numpy().tolist()
                
                if flag0 and flag1:
                    ta = torch.tensor([t])
                    da = torch.tensor([d])
                    flag0 = False
                else:
                    ta = torch.cat((ta, torch.tensor([t])), 0)
                    da = torch.cat((da, torch.tensor([d])), 0)

            # elif target[i]==0:
            #     target[i] = 1
            #     t=target[i].numpy().tolist()
            #     d=data[i].numpy().tolist()
                
            #     if flag0 and flag1 :
            #         ta = torch.tensor([t])
            #         da = torch.tensor([d])
            #         flag1 = False
            #     else:
            #         ta = torch.cat((ta, torch.tensor([t])), 0)
            #         da = torch.cat((da, torch.tensor([d])), 0)

        if da.size() == torch.Size([0]):   
            continue
        test.append([da, ta])

    
    print("Data for task 1 prepared!")
    print("Now begin task 1...")

    # train/test: [batch_idx] [0=data, 1=target] [tensor/label]

    model = LeNet5(outdim=2).to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        print("Now Train f_1...")
        train_net(args, model, device, train, optimizer, epoch)
        
        print("Now Test f_1...")
        test_net(model, device, test)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "./nnmodels/f_1.pt")

    return 



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LeNet5 Binary Classifier')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    print("Now Loading Data...")

    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./Data/', download=True, train=True, 
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./Data/', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    print("Data Loaded!")

    
    # p0 = Process(target=task_0, args=(args, device, train_loader, test_loader))
    task_0(args, device, train_loader, test_loader)
    # p1 = Process(target=task_1, args=(args, device, train_loader, test_loader))
    task_1(args, device, train_loader, test_loader)
    # p0.start()
    # p1.start()
    # p0.join()
    # p1.join()

    
    

    
    
    



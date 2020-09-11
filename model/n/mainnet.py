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
import os
import matplotlib
import matplotlib.pyplot as plt



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
        loss = F.nll_loss(output, target)
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
            test_loss = F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim = 1, keepdim = True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data)
    acc = 100. *correct / example

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, example,
        100. * correct / example))

    return test_loss, acc
    


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
            if target[i] == 0:
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

            elif target[i] == 1:
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
            if target[i] == 0:
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

            elif target[i] == 1:
                target[i] = 0
                t=target[i].numpy().tolist()
                d=data[i].numpy().tolist()
                
                if flag0 and flag1 :
                    ta = torch.tensor([t])
                    da = torch.tensor([d])
                    flag1 = False
                else:
                    ta = torch.cat((ta, torch.tensor([t])), 0)
                    da = torch.cat((da, torch.tensor([d])), 0)

        if da.size() == torch.Size([0]):   
            continue
        test.append([da, ta])

    
    print("Data for task 0 is ready!")
    print("Now begin task 0...")

    # train/test: [batch_idx] [0=data, 1=target] [tensor/label]

    model = LeNet5().to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        print("Now Train f_0...")
        train_net(args, model, device, train, optimizer, epoch)
        
        print("Now Test f_0...")
        tl, acc = test_net(model, device, test)
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
            if target[i] == 1:
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

            elif target[i] == 0:
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
            if target[i] == 1:
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

            elif target[i] == 0:
                target[i] = 0
                t=target[i].numpy().tolist()
                d=data[i].numpy().tolist()
                
                if flag0 and flag1 :
                    ta = torch.tensor([t])
                    da = torch.tensor([d])
                    flag1 = False
                else:
                    ta = torch.cat((ta, torch.tensor([t])), 0)
                    da = torch.cat((da, torch.tensor([d])), 0)

        if da.size() == torch.Size([0]):   
            continue
        test.append([da, ta])

    
    print("Data for task 1 is ready!")
    print("Now begin task 1...")

    # train/test: [batch_idx] [0=data, 1=target] [tensor/label]

    model = LeNet5().to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        print("Now Train f_1...")
        train_net(args, model, device, train, optimizer, epoch)
        
        print("Now Test f_1...")
        tl, acc = test_net(model, device, test)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "./nnmodels/f_1.pt")

    return 


def task_2(args, device, train_loader, test_loader):
    testing = []
    print("Preparing data for task 2...")
    for batch_idx, (data, target) in enumerate(train_loader):
        
        flag0 = True
        flag1 = True
        flag2 = True
        ta = torch.empty([0]) # target tensor
        da = torch.empty([0]) # image vector tensor
        for i in range(0, len(target)):
            if target[i] == 1:
                target[i] = 0
                t=target[i].numpy().tolist()
                d=data[i].numpy().tolist()
                
                if flag0 and flag1 and flag2:
                    ta = torch.tensor([t])
                    da = torch.tensor([d])
                    flag0 = False
                else:
                    ta = torch.cat((ta, torch.tensor([t])), 0)
                    da = torch.cat((da, torch.tensor([d])), 0)

            elif target[i] == 0:
                target[i] = 0
                t=target[i].numpy().tolist()
                d=data[i].numpy().tolist()
                
                if flag0 and flag1 :
                    ta = torch.tensor([t])
                    da = torch.tensor([d])
                    flag1 = False
                else:
                    ta = torch.cat((ta, torch.tensor([t])), 0)
                    da = torch.cat((da, torch.tensor([d])), 0)

            elif target[i] == 2:
                target[i] = 1
                t=target[i].numpy().tolist()
                d=data[i].numpy().tolist()
                
                if flag0 and flag1 and flag2:
                    ta = torch.tensor([t])
                    da = torch.tensor([d])
                    flag2 = False
                else:
                    ta = torch.cat((ta, torch.tensor([t])), 0)
                    da = torch.cat((da, torch.tensor([d])), 0)

        if da.size() == torch.Size([0]):   
            continue
        testing.append([da, ta])

    test2 = []
    test0 = []
    test1 = []
    for batch_idx, (data, target) in enumerate(test_loader):
        
        flag0 = True
        flag1 = True
        flag2 = True
        ta2 = torch.empty([0])
        ta0 = torch.empty([0])
        ta1 = torch.empty([0]) # target tensor
        da = torch.empty([0]) # image vector tensor
        for i in range(0, len(target)):
            if target[i] == 1:
                target[i] = 0
                t2 = target[i].numpy().tolist()
                t0 = target[i].numpy().tolist()
                target[i] = 1
                t1 = target[i].numpy().tolist()
                d=data[i].numpy().tolist()
                
                if flag0 and flag1 and flag2:
                    ta1 = torch.tensor([t1])
                    ta2 = torch.tensor([t2])
                    ta0 = torch.tensor([t0])
                    da = torch.tensor([d])
                    flag1 = False
                else:
                    ta1 = torch.cat((ta1, torch.tensor([t1])), 0)
                    ta2 = torch.cat((ta2, torch.tensor([t2])), 0)
                    ta0 = torch.cat((ta0, torch.tensor([t0])), 0)
                    da = torch.cat((da, torch.tensor([d])), 0)

            elif target[i] == 0:
                target[i] = 0
                t1 = target[i].numpy().tolist()
                t2 = target[i].numpy().tolist()
                target[i] = 1
                t0 = target[i].numpy().tolist()
                d=data[i].numpy().tolist()
                
                if flag0 and flag1 and flag2:
                    ta1 = torch.tensor([t1])
                    ta2 = torch.tensor([t2])
                    ta0 = torch.tensor([t0])
                    da = torch.tensor([d])
                    flag0 = False
                else:
                    ta1 = torch.cat((ta1, torch.tensor([t1])), 0)
                    ta2 = torch.cat((ta2, torch.tensor([t2])), 0)
                    ta0 = torch.cat((ta0, torch.tensor([t0])), 0)
                    da = torch.cat((da, torch.tensor([d])), 0)

            elif target[i] == 2:
                target[i] = 1
                t2 = target[i].numpy().tolist()
                target[i] = 0
                t0 = target[i].numpy().tolist()
                t1 = target[i].numpy().tolist()
                d=data[i].numpy().tolist()
                
                if flag0 and flag1 and flag2:
                    ta1 = torch.tensor([t1])
                    ta2 = torch.tensor([t2])
                    ta0 = torch.tensor([t0])
                    da = torch.tensor([d])
                    flag2 = False
                else:
                    ta1 = torch.cat((ta1, torch.tensor([t1])), 0)
                    ta2 = torch.cat((ta2, torch.tensor([t2])), 0)
                    ta0 = torch.cat((ta0, torch.tensor([t0])), 0)
                    da = torch.cat((da, torch.tensor([d])), 0)

        if da.size() == torch.Size([0]):   
            continue
        test2.append([da, ta2])
        test0.append([da, ta0])
        test1.append([da, ta1])
    
    # load trained model
    print("Data for task 2 is ready!")

    print("Seems there's something new...")

    training2 = [] # dataset for initializing f_2
    training1 = [] # dataset for updating f_1
    training0 = [] # dataset for updating f_0

    for iters in range(0, 5):
        print("Now we detect the novel class iteratively...")
        print("Iteration ", iters, "...")
        for k in range(0, len(testing)):
            f_0 = LeNet5()
            f_1 = LeNet5()
            checkpoint0 = torch.load("./nnmodels/f_0.pt")
            checkpoint1 = torch.load("./nnmodels/f_1.pt")
            f_0.load_state_dict(checkpoint0)
            f_1.load_state_dict(checkpoint1)

            x = testing[k][0]

            y_0 = f_0(x) # test x on f_0
            y_1 = f_1(x) # test x on f_1

            mg_0 = y_0[:,1]-y_0[:,0] # get margin vector
            mg_1 = y_1[:,1]-y_1[:,0] # get margin vector
            
            conf_0 = mg_0.argmax() # get the index of the most confident sample for further training
            conf_1 = mg_1.argmax() # get the index of the most confident sample for further training
            
            y_hat_0 = y_0.argmax(dim = 1, keepdim = True) # get prediction result
            y_hat_1 = y_1.argmax(dim = 1, keepdim = True) # get prediction result

            h = torch.eq(y_hat_0, y_hat_1) # do XOR of the results to find samples that belong to novel class
            

            ta2 = torch.empty([0])
            ta0 = torch.empty([0])
            ta1 = torch.empty([0])
            da = torch.empty([0])
            flag1 = True
            flag0 = True
            flag2 = True
            for i in range(0, len(h)):
                if h[i]:
                    testing[k][1][i] = 1
                    t2 = testing[k][1][i].numpy().tolist()
                    testing[k][1][i] = 0
                    t0 = testing[k][1][i].numpy().tolist()
                    t1 = testing[k][1][i].numpy().tolist()
                    d = x[i].numpy().tolist()
                    if flag2 and flag1 and flag0:
                        ta2 = torch.tensor([t2])
                        ta0 = torch.tensor([t0])
                        ta1 = torch.tensor([t1])
                        da = torch.tensor([d])
                        flag2 = False
                    else:
                        ta2 = torch.cat((ta2, torch.tensor([t2])), 0)
                        ta0 = torch.cat((ta0, torch.tensor([t0])), 0)
                        ta1 = torch.cat((ta1, torch.tensor([t1])), 0)
                        da = torch.cat((da, torch.tensor([d])), 0)
                
                elif conf_0 == i:
                    testing[k][1][i] = 0
                    t2 = testing[k][1][i].numpy().tolist()
                    t1 = testing[k][1][i].numpy().tolist()
                    testing[k][1][i] = 1
                    t0 = testing[k][1][i].numpy().tolist()
                    d = x[i].numpy().tolist()
                    if flag1 and flag0 and flag2:
                        ta2 = torch.tensor([t2])
                        ta0 = torch.tensor([t0])
                        ta1 = torch.tensor([t1])
                        da = torch.tensor([d])
                        flag0 = False
                    else:
                        ta2 = torch.cat((ta2, torch.tensor([t2])), 0)
                        ta0 = torch.cat((ta0, torch.tensor([t0])), 0)
                        ta1 = torch.cat((ta1, torch.tensor([t1])), 0)
                        da = torch.cat((da, torch.tensor([d])), 0)

                elif conf_1 == i:
                    testing[k][1][i] = 0
                    t2 = testing[k][1][i].numpy().tolist()
                    t0 = testing[k][1][i].numpy().tolist()
                    testing[k][1][i] = 1
                    t1 = testing[k][1][i].numpy().tolist()
                    d = x[i].numpy().tolist()
                    if flag1 and flag0 and flag2:
                        ta2 = torch.tensor([t2])
                        ta0 = torch.tensor([t0])
                        ta1 = torch.tensor([t1])
                        da = torch.tensor([d])
                        flag1 = False
                    else:
                        ta2 = torch.cat((ta2, torch.tensor([t2])), 0)
                        ta0 = torch.cat((ta0, torch.tensor([t0])), 0)
                        ta1 = torch.cat((ta1, torch.tensor([t1])), 0)
                        da = torch.cat((da, torch.tensor([d])), 0)


            if da.size() == torch.Size([0]):   
                continue
            training2.append([da, ta2])
            training0.append([da, ta0])
            training1.append([da, ta1])

        
        print("Aha! I'm detecting the novel class!")

        print("It's time to specify a new classifier for the novel class...")

        # train/test: [batch_idx] [0=data, 1=target] [tensor/label]

        optimizer0 = optim.Adadelta(f_0.parameters(), lr=args.lr)

        optimizer1 = optim.Adadelta(f_1.parameters(), lr=args.lr)

        scheduler0 = StepLR(optimizer0, step_size=1, gamma=args.gamma)

        scheduler1 = StepLR(optimizer1, step_size=1, gamma=args.gamma)

        for epoch in range(1, args.epochs + 1):
            print("Now Update f_0...")
            train_net(args, f_0, device, training0, optimizer0, epoch)
            tl, acc = test_net(f_0, device, test0)
            scheduler0.step()

        for epoch in range(1, args.epochs + 1):    
            print("Now Update f_1...")
            train_net(args, f_1, device, training1, optimizer1, epoch)
            tl, acc = test_net(f_1, device, test1)
            scheduler1.step()
            

        if args.save_model:
            torch.save(f_0.state_dict(), "./nnmodels/f_0.pt")
            torch.save(f_1.state_dict(), "./nnmodels/f_1.pt")


    try:
        f_2 = LeNet5()
        checkpoint = torch.load("./nnmodels/f_2.pt")
        f_2.load_state_dict(checkpoint)
    except:
        f_2 = LeNet5().to(device)
        optimizer = optim.Adadelta(f_2.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    for epoch in range(1, args.epochs + 1):
        print("Now Train f_2...")
        train_net(args, f_2, device, training2, optimizer, epoch)
        tl, acc = test_net(f_2, device, test2)
        scheduler.step()

    if args.save_model:
        torch.save(f_2.state_dict(), "./nnmodels/f_2.pt")

    
    return 


def task_3(args, device, train_loader, test_loader):
    testing = []
    print("Preparing data for task ３...")
    for batch_idx, (data, target) in enumerate(train_loader):
        
        flag0 = True
        flag1 = True
        flag2 = True
        ta = torch.empty([0]) # target tensor
        da = torch.empty([0]) # image vector tensor
        for i in range(0, len(target)):
            if target[i] == 1:
                target[i] = 0
                t=target[i].numpy().tolist()
                d=data[i].numpy().tolist()
                
                if flag0 and flag1 and flag2:
                    ta = torch.tensor([t])
                    da = torch.tensor([d])
                    flag0 = False
                else:
                    ta = torch.cat((ta, torch.tensor([t])), 0)
                    da = torch.cat((da, torch.tensor([d])), 0)

            elif target[i] == 0:
                target[i] = 0
                t=target[i].numpy().tolist()
                d=data[i].numpy().tolist()
                
                if flag0 and flag1 :
                    ta = torch.tensor([t])
                    da = torch.tensor([d])
                    flag1 = False
                else:
                    ta = torch.cat((ta, torch.tensor([t])), 0)
                    da = torch.cat((da, torch.tensor([d])), 0)

            elif target[i] == 3:
                target[i] = 1
                t=target[i].numpy().tolist()
                d=data[i].numpy().tolist()
                
                if flag0 and flag1 and flag2:
                    ta = torch.tensor([t])
                    da = torch.tensor([d])
                    flag2 = False
                else:
                    ta = torch.cat((ta, torch.tensor([t])), 0)
                    da = torch.cat((da, torch.tensor([d])), 0)

        if da.size() == torch.Size([0]):   
            continue
        testing.append([da, ta])

    test2 = []
    test0 = []
    test1 = []
    for batch_idx, (data, target) in enumerate(test_loader):
        
        flag0 = True
        flag1 = True
        flag2 = True
        ta2 = torch.empty([0])
        ta0 = torch.empty([0])
        ta1 = torch.empty([0]) # target tensor
        da = torch.empty([0]) # image vector tensor
        for i in range(0, len(target)):
            if target[i] == 1:
                target[i] = 0
                t2 = target[i].numpy().tolist()
                t0 = target[i].numpy().tolist()
                target[i] = 1
                t1 = target[i].numpy().tolist()
                d=data[i].numpy().tolist()
                
                if flag0 and flag1 and flag2:
                    ta1 = torch.tensor([t1])
                    ta2 = torch.tensor([t2])
                    ta0 = torch.tensor([t0])
                    da = torch.tensor([d])
                    flag1 = False
                else:
                    ta1 = torch.cat((ta1, torch.tensor([t1])), 0)
                    ta2 = torch.cat((ta2, torch.tensor([t2])), 0)
                    ta0 = torch.cat((ta0, torch.tensor([t0])), 0)
                    da = torch.cat((da, torch.tensor([d])), 0)

            elif target[i] == 0:
                target[i] = 0
                t1 = target[i].numpy().tolist()
                t2 = target[i].numpy().tolist()
                target[i] = 1
                t0 = target[i].numpy().tolist()
                d=data[i].numpy().tolist()
                
                if flag0 and flag1 and flag2:
                    ta1 = torch.tensor([t1])
                    ta2 = torch.tensor([t2])
                    ta0 = torch.tensor([t0])
                    da = torch.tensor([d])
                    flag0 = False
                else:
                    ta1 = torch.cat((ta1, torch.tensor([t1])), 0)
                    ta2 = torch.cat((ta2, torch.tensor([t2])), 0)
                    ta0 = torch.cat((ta0, torch.tensor([t0])), 0)
                    da = torch.cat((da, torch.tensor([d])), 0)

            elif target[i] == 3:
                target[i] = 1
                t2 = target[i].numpy().tolist()
                target[i] = 0
                t0 = target[i].numpy().tolist()
                t1 = target[i].numpy().tolist()
                d=data[i].numpy().tolist()
                
                if flag0 and flag1 and flag2:
                    ta1 = torch.tensor([t1])
                    ta2 = torch.tensor([t2])
                    ta0 = torch.tensor([t0])
                    da = torch.tensor([d])
                    flag2 = False
                else:
                    ta1 = torch.cat((ta1, torch.tensor([t1])), 0)
                    ta2 = torch.cat((ta2, torch.tensor([t2])), 0)
                    ta0 = torch.cat((ta0, torch.tensor([t0])), 0)
                    da = torch.cat((da, torch.tensor([d])), 0)

        if da.size() == torch.Size([0]):   
            continue
        test2.append([da, ta2])
        test0.append([da, ta0])
        test1.append([da, ta1])

    # load trained model
    print("Data for task 3 is ready!")

    print("Seems there's something new...")

    training2 = [] # dataset for initializing f_2
    training1 = [] # dataset for updating f_1
    training0 = [] # dataset for updating f_0

    for iters in range(0, 5):
        print("Now we detect the novel class iteratively...")
        print("Iteration ", iters, "...")
        for k in range(0, len(testing)):
            f_0 = LeNet5()
            f_1 = LeNet5()
            checkpoint0 = torch.load("./nnmodels/f_0.pt")
            checkpoint1 = torch.load("./nnmodels/f_1.pt")
            f_0.load_state_dict(checkpoint0)
            f_1.load_state_dict(checkpoint1)

            x = testing[k][0]

            y_0 = f_0(x) # test x on f_0
            y_1 = f_1(x) # test x on f_1

            mg_0 = y_0[:,1]-y_0[:,0] # get margin vector
            mg_1 = y_1[:,1]-y_1[:,0] # get margin vector
            
            conf_0 = mg_0.argmax() # get the index of the most confident sample for further training
            conf_1 = mg_1.argmax() # get the index of the most confident sample for further training
            
            y_hat_0 = y_0.argmax(dim = 1, keepdim = True) # get prediction result
            y_hat_1 = y_1.argmax(dim = 1, keepdim = True) # get prediction result

            h = torch.eq(y_hat_0, y_hat_1) # do XOR of the results to find samples that belong to novel class
            

            ta2 = torch.empty([0])
            ta0 = torch.empty([0])
            ta1 = torch.empty([0])
            da = torch.empty([0])
            flag1 = True
            flag0 = True
            flag2 = True
            for i in range(0, len(h)):
                if h[i]:
                    testing[k][1][i] = 1
                    t2 = testing[k][1][i].numpy().tolist()
                    testing[k][1][i] = 0
                    t0 = testing[k][1][i].numpy().tolist()
                    t1 = testing[k][1][i].numpy().tolist()
                    d = x[i].numpy().tolist()
                    if flag2 and flag1 and flag0:
                        ta2 = torch.tensor([t2])
                        ta0 = torch.tensor([t0])
                        ta1 = torch.tensor([t1])
                        da = torch.tensor([d])
                        flag2 = False
                    else:
                        ta2 = torch.cat((ta2, torch.tensor([t2])), 0)
                        ta0 = torch.cat((ta0, torch.tensor([t0])), 0)
                        ta1 = torch.cat((ta1, torch.tensor([t1])), 0)
                        da = torch.cat((da, torch.tensor([d])), 0)
                
                elif conf_0 == i:
                    testing[k][1][i] = 0
                    t2 = testing[k][1][i].numpy().tolist()
                    t1 = testing[k][1][i].numpy().tolist()
                    testing[k][1][i] = 1
                    t0 = testing[k][1][i].numpy().tolist()
                    d = x[i].numpy().tolist()
                    if flag1 and flag0 and flag2:
                        ta2 = torch.tensor([t2])
                        ta0 = torch.tensor([t0])
                        ta1 = torch.tensor([t1])
                        da = torch.tensor([d])
                        flag0 = False
                    else:
                        ta2 = torch.cat((ta2, torch.tensor([t2])), 0)
                        ta0 = torch.cat((ta0, torch.tensor([t0])), 0)
                        ta1 = torch.cat((ta1, torch.tensor([t1])), 0)
                        da = torch.cat((da, torch.tensor([d])), 0)

                elif conf_1 == i:
                    testing[k][1][i] = 0
                    t2 = testing[k][1][i].numpy().tolist()
                    t0 = testing[k][1][i].numpy().tolist()
                    testing[k][1][i] = 1
                    t1 = testing[k][1][i].numpy().tolist()
                    d = x[i].numpy().tolist()
                    if flag1 and flag0 and flag2:
                        ta2 = torch.tensor([t2])
                        ta0 = torch.tensor([t0])
                        ta1 = torch.tensor([t1])
                        da = torch.tensor([d])
                        flag1 = False
                    else:
                        ta2 = torch.cat((ta2, torch.tensor([t2])), 0)
                        ta0 = torch.cat((ta0, torch.tensor([t0])), 0)
                        ta1 = torch.cat((ta1, torch.tensor([t1])), 0)
                        da = torch.cat((da, torch.tensor([d])), 0)


            if da.size() == torch.Size([0]):   
                continue
            training2.append([da, ta2])
            training0.append([da, ta0])
            training1.append([da, ta1])

        
        print("Aha! I'm detecting the novel class!")

        print("It's time to specify a new classifier for the novel class...")

        # train/test: [batch_idx] [0=data, 1=target] [tensor/label]

        optimizer0 = optim.Adadelta(f_0.parameters(), lr=args.lr)

        optimizer1 = optim.Adadelta(f_1.parameters(), lr=args.lr)

        scheduler0 = StepLR(optimizer0, step_size=1, gamma=args.gamma)

        scheduler1 = StepLR(optimizer1, step_size=1, gamma=args.gamma)

        for epoch in range(1, args.epochs + 1):
            print("Now Update f_0...")
            train_net(args, f_0, device, training0, optimizer0, epoch)
            tl, acc = test_net(f_0, device, test0)
            scheduler0.step()

        for epoch in range(1, args.epochs + 1):    
            print("Now Update f_1...")
            train_net(args, f_1, device, training1, optimizer1, epoch)
            tl, acc = test_net(f_1, device, test1)
            scheduler1.step()
            

        if args.save_model:
            torch.save(f_0.state_dict(), "./nnmodels/f_0.pt")
            torch.save(f_1.state_dict(), "./nnmodels/f_1.pt")


    try:
        f_3 = LeNet5()
        checkpoint = torch.load("./nnmodels/f_3.pt")
        f_3.load_state_dict(checkpoint)
    except:
        f_3 = LeNet5().to(device)
        optimizer = optim.Adadelta(f_3.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    for epoch in range(1, args.epochs + 1):
        print("Now Train f_3...")
        train_net(args, f_3, device, training2, optimizer, epoch)
        tl, acc = test_net(f_3, device, test2)
        scheduler.step()

    if args.save_model:
        torch.save(f_3.state_dict(), "./nnmodels/f_3.pt")

    
    return 





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LeNet5 Binary Classifier')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
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

    # Firstly check model existence
    try: 
        print("Check if there exists trained f_0, f_1...")
        f_0 = LeNet5()
        f_1 = LeNet5()
        checkpoint0 = torch.load("./nnmodels/f_0.pt")
        checkpoint1 = torch.load("./nnmodels/f_1.pt")
        f_0.load_state_dict(checkpoint0)
        f_1.load_state_dict(checkpoint1)
        print("f_0, f_1 loaded successfully!")
    # No model avaliable, learn from scartch
    except: 
        print("There's no model avaliable, thus learn-from-scratch...")
        
        p0 = Process(target=task_0, args=(args, device, train_loader, test_loader))
        p1 = Process(target=task_1, args=(args, device, train_loader, test_loader))
        # Comment the above two lines and uncomment the following two lines to disable multiprocessing
        #task_0(args, device, train_loader, test_loader)
        #task_1(args, device, train_loader, test_loader)
        p0.start()
        p1.start()
        p0.join()
        p1.join()

    # Detecting 2 as novel object
    task_2(args, device, train_loader, test_loader)

    # Detecting 3 as novel object
    # task_3(args, device, train_loader, test_loader)


    



    
    

    
    
    



from __future__ import print_function
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    correct_count = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)[0]
        batch_loss = F.nll_loss(output, target)
        total_loss += batch_loss.item()
        batch_loss.backward()
        optimizer.step()

        # Calculate accuracy
        # get the index of the max log-probability
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred))

        correct_count += correct.sum().item()

        if batch_idx % args['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), batch_loss.item()))

    return {
        "acc": correct_count / len(train_loader.dataset),
        "loss": total_loss
    }


def test(args, model, device, test_loader, held_out, control):
    model.eval()
    total_loss = 0
    correct_count = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            total_loss += F.nll_loss(output, target, reduction='sum').item()

            # Calculate accuracy
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred))

            correct_count += correct.sum().item()

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        total_loss,
        correct_count, len(test_loader.dataset), 100. * correct_count / len(test_loader.dataset)
        ))

    return {
        "acc": correct_count / len(test_loader.dataset),
        "loss": total_loss
    }


def run(train_loader_fn, test_loader_fn, model_fn, args):
    num_classes = 10
    args['use_cuda'] = not args['no_cuda'] and torch.cuda.is_available()

    train_results, test_results = [], []

    device = torch.device("cuda" if args['use_cuda'] else "cpu")
    model = model_fn(num_classes).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])

    train_loader = train_loader_fn(args)
    test_loader = test_loader_fn(args)
    for epoch in range(1, args['epochs'] + 1):
        train_results.append(train(args, model, device, train_loader, optimizer, epoch))
        test_results.append(
            test(args, model, device, test_loader, train_loader.dataset.held_out, train_loader.dataset.control))

    return {"train_results": train_results, "test_results": test_results}

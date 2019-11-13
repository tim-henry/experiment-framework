from __future__ import print_function
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import experiments.get_weight


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    num_loss = 0
    col_loss = 0
    loc_loss = 0
    num_correct_count = 0
    col_correct_count = 0
    loc_correct_count = 0
    correct_count = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        num_target, col_target, loc_target = target[:, 0], target[:, 1], target[:, 2]
        optimizer.zero_grad()

        num_output, col_output, loc_output = model(data)
        batch_num_loss = F.nll_loss(num_output, num_target)
        batch_col_loss = F.nll_loss(col_output, col_target)
        batch_loc_loss = F.nll_loss(loc_output, loc_target)  # TODO 3 way weight
        num_loss += batch_num_loss.item()
        col_loss += batch_col_loss.item()
        loc_loss += batch_loc_loss.item()
        loss = batch_num_loss + batch_col_loss + batch_loc_loss
        loss.backward()
        optimizer.step()
        # Calculate accuracy
        # get the index of the max log-probability
        pred = torch.cat((num_output.argmax(dim=1, keepdim=True),
                          col_output.argmax(dim=1, keepdim=True),
                          loc_output.argmax(dim=1, keepdim=True)), 1)
        num_correct = pred.eq(target.view_as(pred))[:, 0]
        col_correct = pred.eq(target.view_as(pred))[:, 1]
        loc_correct = pred.eq(target.view_as(pred))[:, 2]
        correct = num_correct * col_correct * loc_correct  # all must be correct

        num_correct_count += num_correct.sum().item()
        col_correct_count += col_correct.sum().item()
        loc_correct_count += loc_correct.sum().item()
        correct_count += correct.sum().item()

        if batch_idx % args['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    total_loss = num_loss + col_loss + loc_loss

    return {
        "class_1_name": train_loader.dataset.class_names[0],
        "class_2_name": train_loader.dataset.class_names[1],
        "class_3_name": train_loader.dataset.class_names[2],
        "num_acc": num_correct_count / len(train_loader.dataset),
        "col_acc": col_correct_count / len(train_loader.dataset),
        "loc_acc": loc_correct_count / len(train_loader.dataset),
        "acc": correct_count / len(train_loader.dataset),
        "num_loss": num_loss,
        "col_loss": col_loss,
        "loc_loss": loc_loss,
        "loss": total_loss
    }


def test(args, model, device, test_loader, held_out, control):
    label_1_name = test_loader.dataset.class_names[0].capitalize()
    label_2_name = test_loader.dataset.class_names[1].capitalize()
    label_3_name = test_loader.dataset.class_names[2].capitalize()

    model.eval()
    num_loss = 0
    col_loss = 0
    loc_loss = 0
    num_correct_count = 0
    col_correct_count = 0
    loc_correct_count = 0
    correct_count = 0

    left_out_num_correct_count = 0
    left_out_col_correct_count = 0
    left_out_loc_correct_count = 0
    left_out_correct_count = 0
    left_out_count = 0

    non_left_out_num_correct_count = 0
    non_left_out_col_correct_count = 0
    non_left_out_loc_correct_count = 0
    non_left_out_correct_count = 0
    non_left_out_count = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            num_target, col_target, loc_target = target[:, 0], target[:, 1], target[:, 2]

            num_output, col_output, loc_output = model(data)
            num_loss += F.nll_loss(num_output, num_target, reduction='sum').item()
            col_loss += F.nll_loss(col_output, col_target, reduction='sum').item()
            loc_loss += F.nll_loss(loc_output, loc_target, reduction='sum').item()

            # Calculate accuracy
            # get the index of the max log-probability
            pred = torch.cat((num_output.argmax(dim=1, keepdim=True),
                              col_output.argmax(dim=1, keepdim=True),
                              loc_output.argmax(dim=1, keepdim=True)), 1)
            num_correct = pred.eq(target.view_as(pred))[:, 0]
            col_correct = pred.eq(target.view_as(pred))[:, 1]
            loc_correct = pred.eq(target.view_as(pred))[:, 2]
            correct = num_correct * col_correct * loc_correct  # all must be correct

            num_correct_count += num_correct.sum().item()
            col_correct_count += col_correct.sum().item()
            loc_correct_count += loc_correct.sum().item()
            correct_count += correct.sum().item()

            # Calculate left-out accuracy
            mask = np.zeros(num_target.size())
            for pair in held_out:
                diff_array = np.absolute(target.cpu().numpy() - np.array(pair))
                mask = np.logical_or(mask, diff_array.sum(axis=1) == 0)

            mask = torch.Tensor(mask.astype("uint8")).byte().to(device)

            left_out_num_correct = num_correct * mask
            left_out_col_correct = col_correct * mask
            left_out_loc_correct = loc_correct * mask
            left_out_correct = left_out_num_correct * left_out_col_correct * left_out_loc_correct

            left_out_num_correct_count += left_out_num_correct.sum().item()
            left_out_col_correct_count += left_out_col_correct.sum().item()
            left_out_loc_correct_count += left_out_loc_correct.sum().item()
            left_out_correct_count += left_out_correct.sum().item()
            left_out_count += mask.sum().item()

            # Calculate non_left-out accuracy
            mask = np.zeros(num_target.size())
            for pair in control:
                diff_array = np.absolute(target.cpu().numpy() - np.array(pair))
                mask = np.logical_or(mask, diff_array.sum(axis=1) == 0)

            mask = torch.Tensor(mask.astype("uint8")).byte().to(device)

            non_left_out_num_correct = num_correct * mask
            non_left_out_col_correct = col_correct * mask
            non_left_out_loc_correct = loc_correct * mask
            non_left_out_correct = non_left_out_num_correct * non_left_out_col_correct * non_left_out_loc_correct

            non_left_out_num_correct_count += non_left_out_num_correct.sum().item()
            non_left_out_col_correct_count += non_left_out_col_correct.sum().item()
            non_left_out_loc_correct_count += non_left_out_loc_correct.sum().item()
            non_left_out_correct_count += non_left_out_correct.sum().item()
            non_left_out_count += mask.sum().item()

    total_loss = num_loss + col_loss + loc_loss

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          '({} Accuracy: {}/{} ({:.0f}%), {} Accuracy: {}/{} ({:.0f}%), {} Accuracy: {}/{} ({:.0f}%))\n'.format(
        total_loss,
        correct_count, len(test_loader.dataset), 100. * correct_count / len(test_loader.dataset),
        label_1_name, num_correct_count, len(test_loader.dataset), 100. * num_correct_count / len(test_loader.dataset),
        label_2_name, col_correct_count, len(test_loader.dataset), 100. * col_correct_count / len(test_loader.dataset),
        label_3_name, loc_correct_count, len(test_loader.dataset), 100. * loc_correct_count / len(test_loader.dataset)
    ))

    left_out_acc = None
    if left_out_count > 0:
        print('Left-Out Accuracy: {}/{} ({:.0f}%)\n'
              '(Left-Out {} Accuracy: {}/{} ({:.0f}%), Left-Out {} Accuracy: {}/{} ({:.0f}%),'
              ' Left-Out {} Accuracy: {}/{} ({:.0f}%))\n'.format(
            left_out_correct_count, left_out_count, 100. * left_out_correct_count / left_out_count,
            label_1_name, left_out_num_correct_count, left_out_count, 100. * left_out_num_correct_count / left_out_count,
            label_2_name, left_out_col_correct_count, left_out_count, 100. * left_out_col_correct_count / left_out_count,
            label_3_name, left_out_loc_correct_count, left_out_count, 100. * left_out_loc_correct_count / left_out_count
        ))
        left_out_acc = left_out_correct_count / left_out_count

    non_left_out_acc = None
    if non_left_out_count > 0:
        print('non_left-Out Accuracy: {}/{} ({:.0f}%)\n'
              '(non_left-Out {} Accuracy: {}/{} ({:.0f}%), non_left-Out {} Accuracy: {}/{} ({:.0f}%))\n'.format(
            non_left_out_correct_count, non_left_out_count, 100. * non_left_out_correct_count / non_left_out_count,
            label_1_name, non_left_out_num_correct_count, non_left_out_count,
                                                            100. * non_left_out_num_correct_count / non_left_out_count,
            label_2_name, non_left_out_col_correct_count, non_left_out_count,
                                                            100. * non_left_out_col_correct_count / non_left_out_count,
            label_3_name, non_left_out_loc_correct_count, non_left_out_count,
                                                            100. * non_left_out_loc_correct_count / non_left_out_count,
        ))
        non_left_out_acc = non_left_out_correct_count / non_left_out_count

    return {
        "class_1_name": test_loader.dataset.class_names[0],
        "class_2_name": test_loader.dataset.class_names[1],
        "class_3_name": test_loader.dataset.class_names[2],
        "num_acc": num_correct_count / len(test_loader.dataset),
        "col_acc": col_correct_count / len(test_loader.dataset),
        "loc_acc": loc_correct_count / len(test_loader.dataset),
        "acc": correct_count / len(test_loader.dataset),
        "left_out_num_acc": left_out_num_correct_count / left_out_count if left_out_count != 0 else None,
        "left_out_col_acc": left_out_col_correct_count / left_out_count if left_out_count != 0 else None,
        "left_out_loc_acc": left_out_loc_correct_count / left_out_count if left_out_count != 0 else None,
        "left_out_acc": left_out_acc if left_out_count != 0 else None,
        "non_left_out_num_acc": non_left_out_num_correct_count / non_left_out_count if non_left_out_count != 0 else None,
        "non_left_out_col_acc": non_left_out_col_correct_count / non_left_out_count if non_left_out_count != 0 else None,
        "non_left_out_loc_acc": non_left_out_col_correct_count / non_left_out_count if non_left_out_count != 0 else None,
        "non_left_out_acc": non_left_out_acc if non_left_out_count != 0 else None,
        "num_loss": num_loss,
        "col_loss": col_loss,
        "loss": total_loss
    }


def run(train_loader_fn, test_loader_fn, model_fn, args):
    num_classes = args['num_classes']
    args['use_cuda'] = not args['no_cuda'] and torch.cuda.is_available()
    print("use_cuda? ", args['use_cuda'])

    train_results, test_results = {}, {}

    keep_pcts = args['keep_pcts']
    print("keep_pcts: ", keep_pcts)
    print("weighted loss: ", args["weighted"])

    state_dicts = {}
    for keep_pct in keep_pcts:
        args['keep_pct'] = keep_pct
        print("Keep pct: ", keep_pct)

        device = torch.device("cuda" if args['use_cuda'] else "cpu")
        model = model_fn(num_classes).to(device)

        random_indices = np.arange(10)
        if not args["save_model"]:
            # If saving the model we expect the labels to be in the default order
            np.random.shuffle(random_indices)
        args['color_indices'] = random_indices

        optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])

        keep_pct_train_results, keep_pct_test_results = [], []
        train_loader = train_loader_fn(args)
        test_loader = test_loader_fn(args)

        if args['weighted']:
            pass # TODO
            # args['alpha'] = experiments.get_weight.get_alpha(model.name, train_loader.dataset.name)
            # print(args['alpha'])

        for epoch in range(1, args['epochs'] + 1):
            keep_pct_train_results.append(train(args, model, device, train_loader, optimizer, epoch))
            keep_pct_test_results.append(
                test(args, model, device, test_loader, train_loader.dataset.held_out, train_loader.dataset.control))
        train_results[keep_pct] = keep_pct_train_results
        test_results[keep_pct] = keep_pct_test_results

        if args["save_model"]:
            state_dicts[keep_pct] = model.state_dict()

    if args["save_model"]:
        return {"train_results": train_results, "test_results": test_results, "state_dict": state_dicts}

    return {"train_results": train_results, "test_results": test_results}

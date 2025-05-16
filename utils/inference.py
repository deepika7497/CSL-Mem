import torch

def inference(net, data_loader, device='cpu', loss=None):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (data, labels) in data_loader:
            data = data.to(device)
            labels = labels.to(device)
            out = net(data)

            if(loss != None):
                loss_val = loss(out, labels)

            _, pred = torch.max(out, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size()[0]
        accuracy = float(correct) * 100.0/ float(total)
    
    if(loss != None):
        return correct, total, accuracy, loss_val
    return correct, total, accuracy


def inference_indexed_imagenet(net, dataset, device, batch_size=128):
    net.eval()
    correct = 0
    total = 0
    split='test'
    with torch.no_grad():
        for data in dataset.iterate(split, batch_size, shuffle=False, augmentation=False):
            images = data['image'].numpy().transpose(0, 3, 1, 2)
            inputs = torch.from_numpy(images)
            targets = torch.from_numpy(data['label'].numpy())
            idxs = data['index'].numpy()
            inputs, targets = inputs.cuda(), targets.cuda()
            out = net(inputs)
            _, pred = torch.max(out, dim=1)
            correct += (pred == targets).sum().item()
            total += targets.size()[0]

        accuracy = float(correct) * 100.0/ float(total)
    return correct, total, accuracy
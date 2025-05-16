import torch

def get_loss_and_grad_for_batch(net, criterion, batch_data, batch_labels, temp=1):
    net.eval()
    batch_data.requires_grad_()
    outputs = net(batch_data)
    loss = criterion(outputs / temp, batch_labels)
    grad = torch.autograd.grad(loss.sum(), batch_data)[0]
    loss_grad = grad.reshape(grad.size(0), -1).norm(dim=1).cpu().detach()

    net.zero_grad()
    if batch_data.grad is not None:
        batch_data.grad.zero_()
    return loss, loss_grad, torch.nn.functional.softmax(outputs, 1).max(1)[0]

def get_loss_for_batch(net, criterion, batch_data, batch_labels, temp=1):
    net.eval()
    outputs = net(batch_data)
    loss = criterion(outputs / temp, batch_labels)
    return loss

def get_regularized_curvature_for_batch(net, criterion, batch_data, batch_labels, h=1e-3, niter=10, temp=1):
        num_samples = batch_data.shape[0]
        net.eval()
        regr = torch.zeros(num_samples)
        eigs = torch.zeros(num_samples)
        for _ in range(niter):
            v = torch.randint_like(batch_data, high=2).cuda()
            # Generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1

            v = h * (v + 1e-7)

            batch_data.requires_grad_()
            outputs_pos = net(batch_data + v)
            outputs_orig = net(batch_data)
            loss_pos = criterion(outputs_pos / temp, batch_labels).sum()
            loss_orig = criterion(outputs_orig / temp, batch_labels).sum()
            grad_diff = torch.autograd.grad((loss_pos - loss_orig), batch_data)[0]

            regr += grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1).cpu().detach()
            eigs += torch.diag(torch.matmul(v.reshape(num_samples, -1), grad_diff.reshape(num_samples, -1).T)).cpu().detach()
            net.zero_grad()
            if batch_data.grad is not None:
                batch_data.grad.zero_()

        curv_estimate = regr / niter
        return curv_estimate
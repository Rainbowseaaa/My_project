import torch
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

# def depadd2show(img, sps):
#     n_depad = (sps['neural_pixels'] - sps['target_pixels']) // 2
#     if n_depad == 0:
#         return img
#     return img[n_depad:-n_depad, n_depad:-n_depad]

def several_sample_visual_for_classification(diff_nn, dataset, n, device, sps):
    # 这个data_loader的batch size必须等于1
    diff_nn.model.layer_outputs = []
    diff_nn.model.hook_register()
    diff_nn.model.eval()
    ind = torch.randperm(len(dataset))[:n]
    data = [dataset[i] for i in ind]
    fig, ax = plt.subplots(figsize=[18, 15], nrows=len(data), ncols=len(
        diff_nn.model.names) + 2)
    try:
        with torch.no_grad():
            for i, (x, yhat) in enumerate(data):
                x = x.to(device)
                # yhat = yhat.to(device)
                y, CCD = diff_nn.model(x)
                pred = y.argmax()
                ax[i, 0].imshow(x.squeeze().cpu())
                ax[i, 0].set_title(f"{yhat}")
                ax[i, 0].set_title(f"Input")
                for j in range(len(diff_nn.model.layer_outputs)):
                    ax[i, j + 1].imshow(diff_nn.model.layer_outputs[j].squeeze().cpu().abs())
                ax[i, -1].imshow(CCD.cpu().squeeze())
                diff_nn.model.detec.show_boundaries(ax[i, -1])
                ax[i, -1].set_title(f"Output, pred={pred}")
                diff_nn.model.layer_outputs = []
    finally:
        diff_nn.model.layer_outputs = []
        diff_nn.model.remove_hooks()
    return fig, ax
def several_sample_visual_for_image(model, dataset, n, device, sps):
    # 这个data_loader的batch size必须等于1
    model.hook_register()
    model.eval()
    ind = torch.randperm(len(dataset))[:n]
    data = [dataset[i] for i in ind]
    fig, ax = plt.subplots(figsize=[18, 15], nrows=len(data),
                           ncols=len(model.names) + 2)
    try:
        for i, (x, _) in enumerate(data):
            x = x.to(device)
            yhat = x.to(device)
            y = model(x)
            y = y.detach()
            yhat = yhat / torch.mean(yhat, dim=(-1, -2), keepdim=True)
            y = y / torch.mean(y, dim=(-1, -2), keepdim=True)
            ax[i, 0].imshow(x.squeeze().cpu())
            for j in range(len(model.layer_outputs)):
                ax[i, j + 1].imshow(model.layer_outputs[j].squeeze().cpu().abs()**2)
            ax[i, -1].imshow(y.cpu().squeeze())
            # ax[i, -1].set_title(f"MSE={((y.abs()-yhat).squeeze().cpu().abs()**2).mean()}")
            model.layer_outputs = []
    finally:
        model.layer_outputs = []
        model.remove_hooks()
    return fig, ax
def several_sample_visual_for_regression(model, dataset, n, device, sps):
    # 这个data_loader的batch size必须等于1
    model.hook_register()
    model.eval()
    ind = torch.randperm(len(dataset))[:n]
    data = [dataset[i] for i in ind]
    fig, ax = plt.subplots(figsize=[18, 15], nrows=len(data),
                           ncols=len(model.names) +  3)
    try:
        for i, (x, yhat) in enumerate(data):
            x = x.to(device)
            yhat = yhat.to(device)
            y = model(x)
            y = y.detach()
            yhat = yhat / torch.mean(yhat, dim=(-1, -2), keepdim=True)
            y = y / torch.mean(y, dim=(-1, -2), keepdim=True)
            ax[i, 0].imshow(x.squeeze().cpu())
            for j in range(len(model.layer_outputs)):
                ax[i, j + 1].imshow(model.layer_outputs[j].squeeze().cpu().abs()**2)
            ax[i, -2].imshow(yhat.cpu())
            ax[i, -2].set_title(f"Target")
            ax[i, -1].imshow(y.cpu().squeeze())
            ax[i, -1].set_title(f"MSE={((y.abs()-yhat).squeeze().cpu().abs()**2).mean()}")
            model.layer_outputs = []
    finally:
        model.layer_outputs = []
        model.remove_hooks()
    return fig, ax

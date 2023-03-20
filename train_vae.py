import numpy as np
import torch
import dataset
from dataset import VAEDataset
from models.vae import VQVAE
from torch.utils import data
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


batch_size = 33
epoch = 200
lr_update_epoch = 20
learning_rate = 0.0001
min_learning_rate = 0.000005
params_save_path = "./params/vae.pth"
loader = data.DataLoader(dataset=VAEDataset(), batch_size=batch_size, shuffle=True, num_workers=0)
print(len(loader.dataset))

model = VQVAE(in_dim=3, num_embeddings=512, embedding_dim=256)
model.to(device)
optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters())


def update_lr(optimizer, gamma=0.5):
    global learning_rate
    learning_rate = max(learning_rate*gamma, min_learning_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    print("lr update finished  cur lr: %.5f" % learning_rate)


def processbar(current, totle):
    process_str = ""
    for i in range(int(20*current/totle)):
        process_str += "█"
    while len(process_str) < 20:
        process_str += " "
    return "%s|  %d / %d" % (process_str, current, totle)


def sample(net, x, epoch_count):
    def tensor_to_img(x):
        b, h, w, c = x.shape[0], x.shape[2], x.shape[3], x.shape[1]
        x = (x.permute([0, 2, 3, 1]).contiguous().view(-1, 3) * torch.Tensor(dataset.std).to(device).view(1, 3) + torch.Tensor(dataset.mean).to(device).view(1, 3)) * 255
        x = x.contiguous().view(b, h, w, c)
        x = x.cpu().numpy().astype(np.uint8)
        x = np.concatenate([x[i] for i in range(x.shape[0])], axis=1)
        return x
    net.eval()
    with torch.no_grad():
        _, recon = net(x)
        x_img = tensor_to_img(x)
        recon_img = tensor_to_img(recon)
        sampled_img = np.concatenate([x_img, recon_img], axis=0)
        sampled_img = cv2.cvtColor(sampled_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("./sample_imgs/epoch-%d.jpg" % epoch_count, sampled_img)
    net.train()


if __name__ == '__main__':
    min_loss = 1e8
    for epoch_count in range(1, 1 + epoch):
        loss_val, processed = 0, 0
        model.train()
        for x in loader:
            x = x.to(device).squeeze(0)
            loss = model(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss_val == 0:
                sample(model, x.detach(), epoch_count)

            loss_val += loss.item()
            processed += x.shape[0]
            print("\r进度：%s  本批loss:%.5f" % (processbar(processed, len(loader.dataset)), loss.item()), end="")
        print("\nepoch: %d  loss: %.5f" % (epoch_count, loss_val))
        if min_loss > loss_val:
            min_loss = loss_val
            print("save...")
            torch.save(model.state_dict(), params_save_path)
            print("save finished !!!")

        if epoch_count % lr_update_epoch == 0:
            update_lr(optimizer, 0.5)

import argparse
import torch
import torch.optim as optim
import numpy as np
from torch import nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import os, time
from matplotlib import pyplot as plt 
from tqdm import tqdm
from PIL import Image
from tensorboardX import SummaryWriter
import time


class CVAE(nn.Module):
    def __init__(self, img_size, label_size, latent_size, hidden_size=256):
        super(CVAE, self).__init__()
        self.img_size = img_size  # (C, H, W)
        self.label_size = label_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        # Encoder.
        '''
        img   -> fc  ->                   -> fc -> mean    
                        concat -> encoder                  -> z
        label -> fc  ->                   -> fc -> logstd 
        '''
        self.enc_img_fc = nn.Linear(int(np.prod(self.img_size)), self.hidden_size)
        self.enc_label_fc = nn.Linear(self.label_size, self.hidden_size)
        self.encoder = nn.Sequential(
            nn.Linear(2 * self.hidden_size, 2 * self.hidden_size), nn.ReLU(),
            nn.Linear(2 * self.hidden_size, 2 * self.hidden_size), nn.ReLU(),
        )
        self.z_mean = nn.Linear(2 * self.hidden_size, self.latent_size)
        self.z_logstd = nn.Linear(2 * self.hidden_size, self.latent_size)
        # Decoder.
        '''
        latent -> fc ->
                         concat -> decoder -> reconstruction
        label  -> fc ->
        '''
        self.dec_latent_fc = nn.Linear(self.latent_size, self.hidden_size)
        self.dec_label_fc = nn.Linear(self.label_size, self.hidden_size)
        self.decoder = nn.Sequential(
            nn.Linear(2 * self.hidden_size, 2 * self.hidden_size), nn.ReLU(),
            nn.Linear(2 * self.hidden_size, 2 * self.hidden_size), nn.ReLU(),
            nn.Linear(2 * self.hidden_size, int(np.prod(self.img_size))), nn.Sigmoid(),
        )
        # TODO: assume the distribution of reconstructed images is a Gaussian distibution. Write the log_std here.
        self.recon_logstd = None

    def encode(self, batch_img, batch_label):
        '''
        :param batch_img: a tensor of shape (batch_size, C, H, W)
        :param batch_label: a tensor of shape (batch_size, self.label_size)
        :return: a batch of latent code of shape (batch_size, self.latent_size)
        '''
        # TODO: compute latent z from images and labels
        # print(batch_label.size())
        out_img = self.enc_img_fc(torch.flatten(batch_img, start_dim=1))
        out_label = self.enc_label_fc(batch_label)
        out = torch.cat([out_img, out_label], -1)
        out = self.encoder(out)
        z_mean = self.z_mean(out)
        # print("z_mean.size() = ", z_mean.size())
        z_logstd = self.z_logstd(out)
        
        # sample
        z_std = torch.exp(z_logstd)
        noise = torch.randn_like(z_std)

        # memory
        self.z_mean_val = z_mean
        self.z_std_val = z_std

        return z_mean + z_std * noise

    def decode(self, batch_latent, batch_label):
        '''
        :param batch_latent: a tensor of shape (batch_size, self.latent_size)
        :param batch_label: a tensor of shape (batch_size, self.label_size)
        :return: reconstructed results
        '''

        out_latent = self.dec_latent_fc(batch_latent)
        out_label = self.dec_label_fc(batch_label)
        out = torch.cat([out_latent, out_label], -1)
        out = self.decoder(out)

        out = out.reshape((-1,) + self.img_size)

        return out  # Placeholder.

    def sample(self, batch_latent, batch_label):
        '''
        :param batch_latent: a tensor of size (batch_size, self.latent_size)
        :param batch_label: a tensor of size (batch_size, self.label_dim)
        :return: a tensor of size (batch_size, C, H, W), each value is in range [0, 1]
        '''
        with torch.no_grad():
            # TODO: get samples from the decoder.
            return self.decode(batch_latent, batch_label)
            pass
        return None  # Placeholder.


#########################
####  DO NOT MODIFY  ####
def generate_samples(cvae, n_samples_per_class, device):
    cvae.eval()
    latent = torch.randn((n_samples_per_class * 10, cvae.latent_size), device=device)
    label = torch.eye(cvae.label_size, dtype=torch.float, device=device).repeat(n_samples_per_class, 1)
    imgs = cvae.sample(latent, label).cpu()
    label = torch.argmax(label, dim=-1).cpu()
    samples = dict(imgs=imgs, labels=label)
    return samples
#########################

def save_samples_image(samples, path="output.png"):
    sample_images = samples["imgs"]
    sample_labels = samples["labels"]
    toPIL = transforms.ToPILImage()

    w_num = 10
    h_num = int(len(sample_images)/w_num) + 1
    UNIT_SIZE = 28 # 一张图的大小是200*200
    GAP = 3
    target_shape = (w_num * (UNIT_SIZE + GAP), h_num * (UNIT_SIZE + GAP)) # shape[0]表示横坐标，shape[1]表示纵坐标
    target = Image.new('RGB', target_shape)
    width = 0
    for img in sample_images:
        x, y = int(width%target_shape[0]), int(width/target_shape[0])*(UNIT_SIZE+10) # 左上角坐标，从左到右递增
        target.paste(toPIL(img), (x, y, x+UNIT_SIZE, y+UNIT_SIZE))
        width += (UNIT_SIZE+GAP)

    target.save(path)

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    # Load dataset
    if args.dataset == "mnist":
        dataset = MNIST(root="../data",
                        transform=transforms.ToTensor(),  # TODO: you may want to tweak this
                        train=not args.eval)
        dataloader = DataLoader(dataset, args.batch_size, shuffle=True, drop_last=True)
    else:
        raise NotImplementedError

    # Configure
    if not args.eval:
        time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
        logdir = args.logdir if args.logdir is not None else "logs/cvae_" + time_str
        os.makedirs(logdir, exist_ok=True)
        writer = SummaryWriter(logdir)

        imgdir = args.imgdir if args.imgdir is not None else "imgs/cvae_" + time_str
        os.makedirs(imgdir, exist_ok=True)

        checkpointdir = args.checkpointdir if args.checkpointdir is not None else "checkpoints/cvae_" + time_str
        os.makedirs(checkpointdir, exist_ok=True)

    label_dim = 10
    img_dim = (1, 28, 28)
    latent_dim = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cvae = CVAE(img_dim, label_dim, latent_dim)
    cvae.to(device)
    # optimizer = optim.Adam(cvae.parameters(), lr=args.lr)
    optimizer = optim.SGD(cvae.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 50, 80], gamma=0.1)

    # plt.figure()
    # toPIL = transforms.ToPILImage()
    # plt.imshow(toPIL(dataset[0][0]), cmap='gray')
    # plt.savefig("output.png")

    # for images, labels in dataloader:
    #     plt.figure()
    #     toPIL = transforms.ToPILImage()
    #     plt.imshow(toPIL(images[0]), cmap='gray')
    #     plt.savefig("output.png")
    #     print('lab =', labels[0])
    #     break

    def KLLoss(z_mean, z_std):
        return 0.5 * torch.mean(z_mean ** 2 + z_std ** 2 - 1 - torch.log(z_std))

    if args.grade:
        from grader import Grader
        grader = Grader()

    if not args.eval:
        for name, param in cvae.named_parameters():
            print(name, param.shape)
        prior = torch.distributions.Normal(0, 1)

        # criterion = nn.MSELoss()
        criterion = nn.BCELoss()

        best_acc = 0.
        best_epoch = -1

        for epoch in range(args.num_epochs):
            # TODO: Training, logging, saving, visualization, etc.
            loss_sum, mse_sum, kl_sum = 0, 0, 0
            for it, (images, labels) in tqdm(enumerate(dataloader)):
                images = images.to(device)
                labels = labels.to(device)

                labels_onehot = torch.zeros((labels.size(0), label_dim)).to(device)
                labels_onehot.scatter_(dim=1, index=labels.view(-1, 1), value=1)
                # print(labels_onehot.size(), labels_onehot)

                optimizer.zero_grad()
                latent = cvae.encode(images, labels_onehot)
                recon = cvae.decode(latent, labels_onehot)
                kl_loss = KLLoss(cvae.z_mean_val, cvae.z_std_val)
                # TODO: finish loss = MSE + KL
                mse_loss = criterion(recon, images)
                loss = mse_loss + kl_loss # changed
                loss.backward()
                optimizer.step()

                loss_sum += loss.data
                mse_sum += mse_loss.data
                kl_sum += kl_loss.data
            
            loss_sum /= len(dataloader)
            mse_sum /= len(dataloader)
            kl_sum /= len(dataloader)
            writer.add_scalar('loss/TotalLoss', loss_sum, epoch)
            writer.add_scalar('loss/MSE', mse_sum, epoch)
            writer.add_scalar('loss/KL', kl_sum, epoch)
            print("Epoch %d, iteration %d: loss=%.6f, mse=%.6f, kl=%.6f" % (epoch, it, loss_sum, mse_sum, kl_sum))

            samples = generate_samples(cvae, 10, device)
            save_samples_image(samples, os.path.join(imgdir, f"epoch{epoch}.png"))

            checkpoint = {'model': cvae.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, os.path.join(checkpointdir, f"epoch{epoch}.pt"))

            if args.grade:
                cvae.eval()
                samples = generate_samples(cvae, 1000, device)
                acc = grader.grade(samples)
                writer.add_scalar('grade/accuracy', acc, epoch)
                print("Epoch %d: accuracy=%.6f" % (epoch, acc))

                if acc > best_acc:
                    torch.save(checkpoint, os.path.join(checkpointdir, "best.pt"))
                    best_acc = acc
                    best_epoch = epoch
        print(f"Best accuracy: {best_acc}, at epoch {best_epoch}")

    else:
        assert args.load_path is not None
        checkpoint = torch.load(args.load_path, map_location=device)
        cvae.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        cvae.eval()
        samples = generate_samples(cvae, 1000, device)
        torch.save(samples, "vae_generated_samples.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", choices=["mnist"], default="mnist")
    parser.add_argument("--lr", type=float, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--imgdir", type=str, default=None)
    parser.add_argument("--checkpointdir", type=str, default=None)
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--grade", action="store_true", default=False)
    parser.add_argument("--load_path", type=str, default=None)
    args = parser.parse_args()
    main(args)

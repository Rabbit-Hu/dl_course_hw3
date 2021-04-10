from classifier import Net
import torch
from torchvision import transforms
from PIL import Image
import os

class Grader:
    def __init__(self):
        self.device = torch.device("cuda")
        self.model = Net()
        self.model.load_state_dict(torch.load("mnist_cnn.pt"))
        self.model = self.model.to(self.device)
        self.model.eval()

        # toPIL = transforms.ToPILImage()
        # os.makedirs('debug', exist_ok=True)
        # for i, img in enumerate(imgs[:20]):
        #     img = toPIL(img)
        #     img.save(f'debug/fake_img{i}.png')

        self.transform=transforms.Compose([
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def grade(self, samples):
        imgs, labels = samples['imgs'].to(self.device), samples['labels'].to(self.device)
        pred = self.model(self.transform(imgs)).argmax(dim=1)
        # print(pred[:20])
        acc = torch.sum(pred == labels) / len(pred)
        # print("Accuracy:", acc)
        return acc

if __name__ == '__main__':
    grader = Grader()
    samples = torch.load("vae_generated_samples.pt")
    acc = grader.grade(samples)
    print("Accuracy:", acc)
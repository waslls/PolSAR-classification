import  torch
from    torch import  nn
from    torch.nn import functional as F
from    torch.utils.data import DataLoader
from    torchvision import datasets
from    torchvision import transforms
from    torch import nn, optim

# from    torchvision.models import resnet18


class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out):

        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        # self.ca1 = ChannelAttention(ch_out)
        # self.sa1 = SpatialAttention()

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
                nn.BatchNorm2d(ch_out)
            )


    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # short cut.
        out = self.extra(x) + out

        return out

class ResNet(nn.Module):

    def __init__(self, f_num=23, num_class=16):
        super(ResNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(f_num, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        # followed 4 blocks
        # [b, 64, h, w] => [b, 128, h ,w]
        self.blk1 = ResBlk(64, 64)
        # [b, 128, h, w] => [b, 256, h, w]
        self.blk2 = ResBlk(64, 128)
        # # [b, 256, h, w] => [b, 512, h, w]
        # self.blk3 = ResBlk(128, 256)
        # # [b, 512, h, w] => [b, 1024, h, w]
        # self.blk4 = ResBlk(256, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        # [b, 64, h, w] => [b, 1024, h, w]
        x = self.blk1(x)
        x = self.blk2(x)
        # x = self.blk3(x)
        # x = self.blk4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# x = torch.rand([100, 23, 9, 9])
# net = ResNet()
# out = net(x)
# print(out.shape)



# def main():
#     batchsz = 32
#
#     cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
#         transforms.Resize((32, 32)),
#         transforms.ToTensor()
#     ]), download=True)
#     cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)
#
#     cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
#         transforms.Resize((32, 32)),
#         transforms.ToTensor()
#     ]), download=True)
#     cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)
#
#
#     x, label = iter(cifar_train).next()
#     print('x:', x.shape, 'label:', label.shape)
#
#     device = torch.device('cuda')
#     # model = Lenet5().to(device)
#     model = ResNet18().to(device)
#
#     criteon = nn.CrossEntropyLoss().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#     print(model)
#
#     for epoch in range(1000):
#
#         model.train()
#         for batchidx, (x, label) in enumerate(cifar_train):
#             # [b, 3, 32, 32]
#             # [b]
#             x, label = x.to(device), label.to(device)
#
#             logits = model(x)
#             # logits: [b, 10]
#             # label:  [b]
#             # loss: tensor scalar
#             loss = criteon(logits, label)
#
#             # backprop
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#
#         #
#         print(epoch, 'loss:', loss.item())
#
#
#         model.eval()
#         with torch.no_grad():
#             # test
#             total_correct = 0
#             total_num = 0
#             for x, label in cifar_test:
#                 # [b, 3, 32, 32]
#                 # [b]
#                 x, label = x.to(device), label.to(device)
#
#                 # [b, 10]
#                 logits = model(x)
#                 # [b]
#                 pred = logits.argmax(dim=1)
#                 # [b] vs [b] => scalar tensor
#                 correct = torch.eq(pred, label).float().sum().item()
#                 total_correct += correct
#                 total_num += x.size(0)
#                 # print(correct)
#
#             acc = total_correct / total_num
#             print(epoch, 'acc:', acc)
#
#
# if __name__ == '__main__':
#     main()
import  torch
from    torch import  nn
from    torch.nn import functional as F
from    torch.utils.data import DataLoader
from    torchvision import datasets
from    torchvision import transforms
from    torch import nn, optim

# from    torchvision.models import resnet18
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out):

        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)

        self.ca0 = ChannelAttention(ch_out)
        self.sa0 = SpatialAttention()

        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.ca1 = ChannelAttention(ch_out)
        self.sa1 = SpatialAttention()

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
                nn.BatchNorm2d(ch_out)
            )


    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.ca0(out) * out
        out = self.sa0(out) * out
        out = self.bn2(self.conv2(out))
        out = self.ca1(out) * out
        out = self.sa1(out) * out
        # short cut.
        out = self.extra(x) + out

        return out

class ResNet(nn.Module):

    def __init__(self, f_num=124, num_class=16):
        super(ResNet, self).__init__()

        self.ca4 = ChannelAttention(124)
        self.sa4 = SpatialAttention()
        self.conv1 = nn.Sequential(
            nn.Conv2d(f_num, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256)
        )
        self.ca3 = ChannelAttention(256)
        self.sa3 = SpatialAttention()
        # followed 4 blocks
        # [b, 64, h, w] => [b, 128, h ,w]
        self.blk1 = ResBlk(256, 256)
        # [b, 128, h, w] => [b, 256, h, w]
        self.blk2 = ResBlk(256, 512)
        # # [b, 256, h, w] => [b, 512, h, w]
        # self.blk3 = ResBlk(128, 256)
        # # [b, 512, h, w] => [b, 1024, h, w]
        # self.blk4 = ResBlk(256, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_class)


    def forward(self, x):
        # ca = self.ca4(x)
        x = self.ca4(x) * x
        # x = self.sa4(x) * x
        x = F.relu(self.conv1(x))
        x = self.ca3(x) * x
        x = self.sa3(x) * x
        # [b, 64, h, w] => [b, 1024, h, w]
        x = self.blk1(x)
        x = self.blk2(x)
        # x = self.blk3(x)
        # x = self.blk4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)


        return x

# x = torch.rand([100, 124, 9, 9])
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
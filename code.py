import argparse, os, numpy as np, math, torchvision.transforms as transforms
from torchvision.utils import save_image; from torch.utils.data import DataLoader, Dataset ;from torchvision import datasets; from torch.autograd import Variable
import torch.nn as nn, torch.nn.functional as F, torch
import matplotlib.pyplot as plt

os.chdir('D:\\Academic\Project')

def show(img):
  fakeimg = np.transpose(img.detach().numpy(), (1, 2, 0)).squeeze()
  plt.imshow(fakeimg, cmap='gray')
  plt.show()


test = torch.load('images/mnist/MNIST/processed/test.pt')
train = torch.load('images/mnist/MNIST/processed/training.pt')
plt.imshow(train[0][0].numpy(), cmap='gray')
plt.show()


""" DISPLAY image
"""
import gzip
f = gzip.open('data/mnist/MNIST/raw/train-images-idx3-ubyte.gz','r')
image_size = 28
num_images = 5
f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)


image = np.asarray(data[1]).squeeze()
plt.imshow(image)
plt.show()


""" COMPARE loaded and downloaded dataset
"""
mnist len 60000
mnist[0] len 2
mnist[0][0]  shape (torch.Size([1, 32, 32]))
mnist[0][1]  int e.g. 5

dataloader len 938
dataloader[0] shape (torch.Size([64, 1, 32, 32]))

training len 2
training[0] shape (torch.Size([60000, 28, 28]))
training[1] shape (torch.Size([60000]))

dataloader len 938
dataloader[0] shape (torch.Size([64, 1, 28, 28]))


training = torch.load('data/mnist/MNIST/processed/training.pt')
test = torch.load('data/mnist/MNIST/processed/test.pt')
n, w, h = training[0].shape # torch.Size([60000, 28, 28])
train_img = training[0].reshape((-1, 1, h, w)).float()   # shape torch.Size([60000, 1, 28, 28])
train_label = training[1]
train_dataset = torch.utils.data.TensorDataset(train_img, train_label)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batch_size,
    shuffle=True,
)

# Images
X_train = training[0].reshape((-1, 1, 28, 28))
X_train = torch.as_tensor(X_train)

# Labels
y_train = training[1].reshape(training[1].shape[0], 1)
y_train = torch.as_tensor(y_train)

# custom transform
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[index]

        if self.transform:
            x = self.transform(x)

        return x, 1
    
    def __len__(self):
        return self.tensors.size(0)

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, imgs, transform=None):
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        x = self.imgs[index]

        if self.transform:
            x = self.transform(x)

        return x, 1
    
    def __len__(self):
        return len(self.imgs)

transform = transforms.Compose([
  transforms.Resize(opt.img_size),
  transforms.ToTensor(),
  transforms.Normalize([0.5], [0.5])
])


Current key: b'000001ec5684cb40f432f996f8a38e5d076114d8'
Current key: b'0000036b25b1ae054cdf2e3ee954fe2d21db6ae0'
Current key: b'00000b94a63ae2e1c08b4e7452d088156a9a8273'
Current key: b'0000128037967f0d4b7ba748a80d5b248d1203f8'
Current key: b'000015122516efa29c25870b6b60fafe2fae1513'
Current key: b'00001c1755ac170e5382876232aec651f6bda841'
Current key: b'000023924aa8e512db983cba65e30cc106123ce3'
Current key: b'0000356acb787613fc8d8715cc6c182c05173535'
Current key: b'00003f8ec7ff5d59865ab2b6fb58bc663ace3b23'
Current key: b'000042c508f11f4120f86daf01ba72f9444012c6'
Current key: b'00004a106344f353c7dcc4beef71a2b3fe86e342'
Current key: b'00004da4d7f164a6c66ba29ab32a566541a7646e'
Current key: b'000051f9e285ba945cc32d8e3f0478b2bb063c49'
Current key: b'00005914bd150a1e0c26d2284777319fa695bb28'
               0001c44e5f5175a7e6358d207660f971d90abaf4 0
000319b73404935eec40ac49d1865ce197b3a553 1
00038e8b13a97577ada8a884702d607220ce6d15 2
00039ba1bf659c30e50b757280efd5eba6fc2fe1 3



test_db_path = 'D://Academic/Project/lsun/test_lmdb'
test_img = load(test_db_path, -1)   # len 10000

val_db_path = 'D://Academic/Project/lsun/bedroom_val_lmdb'
val_img = load(val_db_path, 300)


db_path = 'D://Academic/Project/lsun/bedroom_train_lmdb'
train_bedroom = load(db_path, 20000)

cv2.imshow('tmp', test_img[3290])



""" Prediction
"""
mnist_test = datasets.MNIST(
    "data/mnist",
    train=False,
    download=True,
    transform=transforms.Compose(
        [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )
)

for i, (img, label) in enumerate(mnist_test):
  print(img.shape)
  show(img)
  print(label)
  if i == 5: break




""" Investigating Generator
"""
def get():
  for i, (imgs, _) in enumerate(dataloader):
    print(len(imgs))
    z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
    gen_imgs = generator(z)
    return z, gen_imgs

z, gen_imgs = get()   # z.shape torch.Size([64, 100])

z = Variable(Tensor(np.random.normal(0, 1, (64, opt.latent_dim))))
gen_imgs = generator(z)

"""
z:
tensor([[-0.0690,  0.7617, -0.3180,  ...,  1.1912, -0.9976,  0.0913],
        [ 0.6834,  1.1011,  1.1965,  ..., -0.2242, -1.3529,  1.1333],
        [ 0.1883,  1.1829,  0.0165,  ..., -0.4388,  0.1379, -1.8526],
        ...,
        [-0.4834,  0.2075, -0.3047,  ...,  0.8072,  0.5202, -0.4946],
        [ 2.3189, -0.6971, -0.4549,  ...,  1.2232, -0.0290, -0.3388],
        [-0.9616, -0.1829, -0.5975,  ..., -0.9919, -2.1247, -0.2331]],
       device='cuda:0')

torch.mean(z)
tensor(0.0068, device='cuda:0')

torch.std(z)
tensor(1.0065, device='cuda:0')
"""

init_size = opt.img_size // 4
l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * init_size ** 2))
l1.cuda()
out = l1(z)   # out.shape torch.Size([64, 131072])
out_view = out.view(out.shape[0], 128, init_size, init_size)   # out_view.shape torch.Size([64, 128, 32, 32])
conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )
conv_blocks.cuda()
img = conv_blocks(out_view)
show(img[0].cpu())



""" Investigating Discriminator
"""
opt.img_size = 32

generator = Generator()
generator.cuda()
generator.apply(weights_init_normal)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

z = Variable(Tensor(np.random.normal(0, 1, (64, opt.latent_dim))))
gen_imgs = generator(z)
gen_imgs.shape

training = torch.load('data/mnist/MNIST/processed/training.pt')
test = torch.load('data/mnist/MNIST/processed/test.pt')
n, w, h = training[0].shape # torch.Size([60000, 28, 28])
train_img = training[0].reshape((-1, 1, h, w)).float()   # shape torch.Size([60000, 1, 28, 28])
train_label = training[1]
train_dataset = torch.utils.data.TensorDataset(train_img, train_label)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batch_size,
    shuffle=True,
)

discriminator = Discriminator()
discriminator.cuda()
discriminator.apply(weights_init_normal)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

pred = discriminator(gen_imgs)  # shape torch.Size([64, 1])


def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

model.cuda()
out = model(gen_imgs)   # shape torch.Size([64, 128, 2, 2])
out = out.view(out.shape[0], -1)  # torch.Size([64, 512])

ds_size = opt.img_size // 2 ** 4

adv_layer = nn.Linear(128 * ds_size ** 2, 1)
adv_layer.cuda()    # Linear(in_features=512, out_features=1, bias=True)

validity = adv_layer(out)   # shape torch.Size([64, 1])




""" normalize
"""

training = torch.load('data/mnist/MNIST/processed/training.pt')
n, w, h = training[0].shape # torch.Size([60000, 28, 28])
train_img = training[0].reshape((-1, 1, h, w)).float()   # shape torch.Size([60000, 1, 28, 28])
train_label = training[1]
img = np.transpose(train_img[0].detach().numpy(), (1, 2, 0)).squeeze()

def normalize(img):
  min, max = np.amin(img), np.amax(img)
  return 2 * ((img - min) / (max - min)) - 1

def normalize(img):
  min, max = torch.min(img), torch.max(img)
  return 2 * ((img - min) / (max - min)) - 1

normalized_train = []
for img in train_img:
  normalized_train.append(normalize(img))

normalized_train = torch.cat(res, 0).reshape(60000, 1, 28, 28)

train_dataset = torch.utils.data.TensorDataset(normalized_train, train_label)

self_norm_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batch_size,
    shuffle=True,
)


# architecture
w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')



def convt(h, k, stride, padding, op, d):
    return (h - 1)*stride - 2*padding + d*(k-1) + op + 1

def conv(h, k, s, p, d):
    return math.floor((h + 2*p - d*(k - 1) - 1) / s + 1)

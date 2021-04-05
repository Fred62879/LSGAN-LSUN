import cv2, lmdb, argparse, os, numpy as np, math, torchvision.transforms as transforms
from torchvision.utils import save_image; from torch.utils.data import DataLoader ;from torchvision import datasets; from torch.autograd import Variable
import torch.nn as nn, torch.nn.functional as F, torch
import matplotlib.pyplot as plt
from PIL import Image



# [batch, channel, height, width]

def load(db_path, limit):
    count, imgs = 0, []
    env = lmdb.open(db_path, map_size=1099511627776,
                    max_readers=100, readonly=True)
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            img = cv2.imdecode(
                # np.fromstring(val, dtype=np.uint8), 1)
                np.frombuffer(val, dtype=np.uint8), 1)
            # cv2.imshow(window_name, img)
            imgs.append(img)
            count += 1

            if count == limit: break
    return imgs

# down sample image so that the larger dimension is limit
def downSample(img, limit):
  h, w, c = img.shape
  
  if w > h:
    scale = w / limit
    d_h = int(h / scale)
    return cv2.resize(img, (limit, d_h))
  
  scale = h / limit
  d_w = int(w / scale)
  return cv2.resize(img, (d_w, limit))

# padding img to be d_sz by d_sz
def padding(img, d_sz):
  h, w, c = img.shape

  horiz = d_sz - w
  left = horiz // 2
  right = horiz - left

  vertic = d_sz - h   # sum of top and bottom pad size
  top = vertic // 2
  bottom = vertic - top

  return np.pad(img, ((top, bottom), (left, right), (0, 0)), mode = 'constant', constant_values = (0, 0))

def normalize(img, mean, std):
  img = img / 255
  n_c = img.shape[2]
  for c in range(n_c):
    img[:,:,c] = (img[:,:,c] - mean) / std
  return img

def process(imgs, d_w, d_sz):
  res = []
  for img in imgs:
    down_img = downSample(img, d_w)
    padded_img = padding(down_img, d_sz)
    # normalized_img = normalize(padded_img, .5, .5)
    # res.append(normalized_img)
    res.append(padded_img)
  return np.array(res)

def crop_center(img, cropx, cropy):
    y, x, _ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

# class CustomTensorDataset(Dataset):
#     """TensorDataset with support of transforms.
#     """
#     def __init__(self, imgs, transform=None):
#         self.imgs = imgs
#         self.transform = transform

#     def __getitem__(self, index):
#         x = self.imgs[index]
#         #x_np = np.array(x)
#         #plt.hist(x_np)

#         if self.transform:
#             x = self.transform(x)

#         return x, 1
    
#     def __len__(self):
#         return len(self.imgs)

# transform = transforms.Compose([
#   transforms.ToTensor(),
#   transforms.Normalize([0.5], [0.5])
# ])

db_path = 'D://Academic/Project/lsun/bedroom_train_lmdb'
train_bedroom = load(db_path, 50000)
processed_imgs = process(train_bedroom, 64, 64)
np.save('D://Academic/Project/data/orig5.npy')

# imgs = load('bedroom', 128)
# processed_imgs = process(imgs, 56, 64)
# ds = np.transpose(processed_imgs, (0, 3, 1, 2))  # shape (128, 3, 64, 64)
# dss = torch.from_numpy(ds)  # shape torch.Size([128, 3, 64, 64])
# # show(dss[0])
# dum_label = torch.from_numpy(np.zeros((128)))

# dataset = torch.utils.data.TensorDataset(dss, dumb_label)
# dataloader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=opt.batch_size,
#     shuffle=True,
# )

# orig_imgs = np.load('pbedrooms.npy')
# imgs = []
# for img in orig_imgs:
#     imgs.append(Image.fromarray(img / 255.0))

# self_data = CustomTensorDataset(imgs, transform)

# # Mar 29
# processed_imgs = np.load('processed.npy')
# t = torch.tensor(np.transpose(processed_imgs, (0, 3, 1, 2)))
# n = t.size(0)
# label = torch.tensor(np.zeros(n))
# train_dataset = torch.utils.data.TensorDataset(t, label)
# train_dataloader = torch.utils.data.DataLoader(
#     train_dataset,
#     batch_size=opt.batch_size,
#     shuffle=True,
# )


# torch.save(generator.state_dict(), 'generator0')
# torch.save(discriminator.state_dict(), 'discriminator0')

# model = Generator()
# model.load_state_dict(torch.load('generator0'))
# model.eval()
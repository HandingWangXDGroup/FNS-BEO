#    Authors:    Chao Li, Wen Yao, Handing Wang, Tingsong Jiang, Xiaoya Zhang
#    Xidian University, China
#    Defense Innovation Institute, Chinese Academy of Military Science, China
#    EMAIL:      lichaoedu@126.com
#    DATE:       May 2023
# ------------------------------------------------------------------------
# This code is part of the program that produces the results in the following paper:
#
# Chao Li, Wen Yao, Handing Wang, Tingsong Jiang, Xiaoya Zhang, Bayesian Evolutionary Optimization for Crafting High-quality Adversarial Examples with Limited Query
# Budget, Applied Soft Computing, 2023.
#
# You are free to use it for non-commercial purposes. However, we do not offer any forms of guanrantee or warranty associated with the code. We would appreciate your acknowledgement.
# ------------------------------------------------------------------------

import numpy as np
import torch
import torch.utils.data as Data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from VGG16_Model import vgg
import torch.nn.functional as F
from FNS_BEO import FDE
import math
import warnings



warnings.filterwarnings("ignore")


batch_size = 1
test_dataset = dsets.CIFAR10(root='/mnt/jfs/CIFAR_data',
                                download =False,
                                transform =transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
                                            ]),
                                train =False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = vgg().to(device)
model.load_state_dict(torch.load(r'vgg16_params.pkl', map_location=torch.device('cuda')))

count = 0
total_count = 0
net_correct = 0
model.eval()

def select_other_images(first_label):
  sum_images = []
  meet = 0
  first_label = first_label.cpu().detach().numpy()
  for images, labels in test_loader:
      if labels != first_label:
          images = images.to(device)
          outputs = model(images)
          outputs = outputs.cpu().detach().numpy()
          if np.argmax(outputs) == labels:
                target_images = torch.tensor(images)
                target_images = np.array(target_images.cpu())
                sum_images.append(target_images)
                meet += 1
                if meet == 20:
                   break
  sum_images=np.array(sum_images)
  num_images = np.size(sum_images, 0)
  if num_images > 1:
     sum_images=sum_images.squeeze()
  else:
     sum_images = sum_images.squeeze()
     sum_images = torch.tensor(sum_images)
     sum_images = sum_images.unsqueeze(0)
     sum_images = sum_images.detach().numpy()
  return sum_images


for images, labels in test_loader:
      images = images.to(device)
      labels = labels.to(device)
      output = model(images)
      _, pre = torch.max(output.data, 1)
      total_count += 1
      if pre == labels:
           net_correct += 1
           clean_soft = F.softmax(output, dim=1)[0]
           cleaninfo_entropy = 0
           for i in range(10):
               cleaninfo_entropy += clean_soft[i] * math.log(clean_soft[i])
           cleaninfo_entropy = -cleaninfo_entropy
           if net_correct <= 100:
               output = output.cpu().detach().numpy()
               min_value = np.min(output)
               output.itemset(labels, min_value)
               second_label = np.argmax(output)
               sum_images = select_other_images(labels)
               images, eva_num = FDE(images, sum_images, labels, cleaninfo_entropy)
               images = images.to(device)
               labels = labels.to(device)
               outputs = model(images)
               _, pre = torch.max(outputs.data, 1)
               if pre == labels:
                    count += 1

      acctak_count = net_correct - count
      print('total count:', total_count, 'net correct:', net_correct, 'atatck fail:', count, 'attack success:', acctak_count )
      if net_correct > 0:
          print('Success ratio of attack: %f %%' % (100 * float(acctak_count) / net_correct))

      if net_correct == 100:
         break
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


import random
import torch
import numpy as np
from VGG16_Model import vgg
import scipy.stats
import torch.nn.functional as FS
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from non_dominated_sorting import fast_non_dominated_sort
from latin import latin
import math
import warnings



warnings.filterwarnings("ignore")
population_size = 50
generations = 100
F = 0.5
CR = 0.6
xmin = -1
xmax = 1
eps = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = vgg().to(device)
model.load_state_dict(torch.load(r'vgg16_params.pkl', map_location=torch.device('cuda')))

def Kcalculate_fitness(taget_image, sample_adv_images, population, first_labels, dim):
    taget_image = taget_image.cpu().detach().numpy()
    Kfitness = []
    function_value=np.zeros(100)
    attack_direction=np.zeros((100, 3, 32, 32))
    for i in range(100):
        for j in range(0,dim):
             attack_direction[i, :, :, :]= attack_direction[i, :, :, :] + population[i, j] * (sample_adv_images[j, :, :, :] - taget_image[0, :, :, :])
        attack_direction[i, :, :, :]= np.sign(attack_direction[i, :, :, :])

    model.eval()
    for b in range(100):
       attack_image = taget_image + eps * attack_direction[b, :, :, :]
       attack_image = torch.from_numpy(attack_image)
       attack_image = attack_image.to(device)
       outputs = model(attack_image.float())
       outputs = outputs.cpu().detach().numpy()
       d = outputs[0, first_labels]
       c = np.min(outputs)
       outputs.itemset(first_labels, c)
       g = np.max(outputs)
       function_value[b] = d-g
       Kfitness.append(function_value[b])

    return Kfitness

def calculate_fitness(taget_image, sample_adv_images, population, first_labels, dim, size):
    taget_image = taget_image.cpu().detach().numpy()
    adv_entropy = []
    fitness = []
    function_value = np.zeros(size)
    attack_direction = np.zeros((size, 3, 32, 32))
    for i in range(size):
        for j in range(0, dim):
            attack_direction[i, :, :, :] = attack_direction[i, :, :, :] + population[i, j] * (sample_adv_images[j, :, :, :] - taget_image[0, :, :, :])

        attack_direction[i, :, :, :] = np.sign(attack_direction[i, :, :, :])

    model.eval()
    for b in range(size):
       attack_image = taget_image + eps * attack_direction[b, :, :, :]
       attack_image = torch.from_numpy(attack_image)
       attack_image = attack_image.to(device)
       outputs = model(attack_image.float())
       adv_soft = FS.softmax(outputs, dim=1)[0]
       info_entropy = 0
       for i in range(10):
           info_entropy += adv_soft[i] * math.log(adv_soft[i])
       info_entropy = -info_entropy
       info_entropy = info_entropy.cpu().detach().numpy()
       adv_entropy.append(info_entropy)
       outputs = outputs.cpu().detach().numpy()
       d = outputs[0, first_labels]
       c = np.min(outputs)
       outputs.itemset(first_labels, c)
       g = np.max(outputs)
       function_value[b] = d-g
       fitness.append(function_value[b])

    return fitness, adv_entropy

def mutation(population,dim):

    Mpopulation=np.zeros((population_size,dim))
    for i in range(population_size):
        r1 = r2 = r3 = 0
        while r1 == i or r2 == i or r3 == i or r2 == r1 or r3 == r1 or r3 == r2:
            r1 = random.randint(0, population_size - 1)
            r2 = random.randint(0, population_size - 1)
            r3 = random.randint(0, population_size - 1)
        Mpopulation[i] = population[r1] + F * (population[r2] - population[r3])

        for j in range(dim):
            if xmin <= Mpopulation[i, j] <= xmax:
                Mpopulation[i, j] = Mpopulation[i, j]
            else:
                Mpopulation[i, j] = xmin + random.random() * (xmax - xmin)
    return Mpopulation

def crossover(Mpopulation, population, dim):
  Cpopulation = np.zeros((population_size, dim))
  for i in range(population_size):
     for j in range(dim):
        rand_j = random.randint(0, dim - 1)
        rand_float = random.random()
        if rand_float <= CR or rand_j == j:
             Cpopulation[i, j] = Mpopulation[i, j]
        else:
             Cpopulation[i, j] = population[i, j]
  return Cpopulation

def selection(Cpopulation, population, gp, Best_solu):

    _, _, Cfitness, _, _ = surrogate_evalu(Cpopulation, gp, Best_solu)
    _, _, pfitness, _, _ = surrogate_evalu(population, gp, Best_solu)
    for i in range(population_size):
        if Cfitness[i] > pfitness[i]:
            population[i] = Cpopulation[i]
        else:
            population[i] = population[i]

    return population

def surrogate_evalu(x_set, gp, Best_solu):
    means, sigmas = gp.predict(x_set, return_std=True)
    PI = np.zeros(50)
    EI = np.zeros(50)
    LCB = np.zeros(50)
    for y in range(50):
        LCB[y] = means[y] - 2 * sigmas[y]
        z = (Best_solu - means[y]) / sigmas[y]
        PI[y] = scipy.stats.norm.cdf(z)
        EI[y] = (Best_solu - means[y]) * scipy.stats.norm.cdf(z) + sigmas[y] * scipy.stats.norm.pdf(z)
    return means, sigmas, PI, EI, LCB

def FDE(taget_image, adversarial_images,  first_labels, clean_entropy):
    clean_entropy = clean_entropy.cpu().detach().numpy()
    num = np.size(adversarial_images, 0)
    if num >= 10:
        dim = 10
        index = random.sample(range(0, num), dim)
        sample_adv_images = adversarial_images[index]
    else:
        dim = num
        sample_adv_images = adversarial_images

    # init the train data of the surrogate model
    population = latin(100, dim, -1, 1)
    # computing the object value of the train data
    Kfit = Kcalculate_fitness(taget_image, sample_adv_images, population, first_labels, dim)
    eval_num = 100
    Best_solu = min(Kfit)
    Best_indi_index = np.argmin(Kfit)
    Best_indi = population[Best_indi_index, :]
    # surrogate model
    kernel = RBF(1.0, (1e-5, 1e5))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    xobs = population
    yobs = np.array(Kfit)
    gp.fit(xobs, yobs)

    # evolution population
    population = population[0:50, :]
    pro = [1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 1]
    r = [0, 0, 0, 0, 0, 0, 0]
    init_entroy = [clean_entropy, clean_entropy, clean_entropy, clean_entropy, clean_entropy, clean_entropy, clean_entropy]
    for step in range(generations):
        if Best_solu < 0:
            break
        Mpopulation = mutation(population, dim)
        Cpopulation = crossover(Mpopulation, population, dim)
        population = selection(Cpopulation, population, gp, Best_solu)
        sfit, std, Pi, Ei, lcb = surrogate_evalu(population, gp, Best_solu)
        index = []
        rand_value = random.random()
        if rand_value < pro[0]:
            oper_id = 0
            sorted_id = sorted(range(len(Ei)), key=lambda k: Ei[k], reverse=True)
            index.append(sorted_id[0])
        if pro[0] <= rand_value < pro[1]:
            oper_id = 1
            fronts = fast_non_dominated_sort(sfit, std)
            fist_front = fronts[0]
            rand_index1 = random.randint(0, len(fist_front)-1)
            index.append(fist_front[rand_index1])
        if pro[1] <= rand_value < pro[2]:
            oper_id = 2
            fronts = fast_non_dominated_sort(sfit, std)
            fist_front = fronts[0]
            front_fit = sfit[fist_front]
            fit_index = np.argmin(front_fit)
            index.append(fist_front[fit_index])
        if pro[2] <= rand_value < pro[3]:
            oper_id = 3
            fronts = fast_non_dominated_sort(sfit, std)
            fist_front = fronts[0]
            front_std = std[fist_front]
            std_index = np.argmin(front_std)
            index.append(fist_front[std_index])
        if pro[3] <= rand_value < pro[4]:
            oper_id = 4
            sorted_id = sorted(range(len(std)), key=lambda k: std[k], reverse=True)
            index.append(sorted_id[0])
        if pro[4] <= rand_value < pro[5]:
            oper_id = 5
            sorted_id = sorted(range(len(Pi)), key=lambda k: Pi[k], reverse=True)
            index.append(sorted_id[0])
        if pro[5] <= rand_value < pro[6]:
            oper_id = 6
            sorted_id = sorted(range(len(lcb)), key=lambda k: lcb[k], reverse=True)
            index.append(sorted_id[-1])
        add_xdata = population[index, :]
        size = len(index)
        Tfitness, adv_entro = calculate_fitness(taget_image, sample_adv_images, add_xdata, first_labels, dim, size)
        init_entroy[oper_id] = max(adv_entro)
        max_entroy = max(init_entroy)
        min_entroy = min(init_entroy)
        r[0] = (init_entroy[0] - max_entroy) / (max_entroy - min_entroy)
        r[1] = (init_entroy[1] - max_entroy) / (max_entroy - min_entroy)
        r[2] = (init_entroy[2] - max_entroy) / (max_entroy - min_entroy)
        r[3] = (init_entroy[3] - max_entroy) / (max_entroy - min_entroy)
        r[4] = (init_entroy[4] - max_entroy) / (max_entroy - min_entroy)
        r[5] = (init_entroy[5] - max_entroy) / (max_entroy - min_entroy)
        r[6] = (init_entroy[6] - max_entroy) / (max_entroy - min_entroy)
        sumr = math.exp(r[0]) + math.exp(r[1]) + math.exp(r[2]) + math.exp(r[3]) + math.exp(r[4]) + math.exp(r[5]) + math.exp(r[6])
        pro[0] = math.exp(r[0]) / sumr
        pro[1] = math.exp(r[1]) / sumr + pro[0]
        pro[2] = math.exp(r[2]) / sumr + pro[1]
        pro[3] = math.exp(r[3]) / sumr + pro[2]
        pro[4] = math.exp(r[4]) / sumr + pro[3]
        pro[5] = math.exp(r[5]) / sumr + pro[4]
        eval_num +=size
        add_ydata = Tfitness
        add_xdata = add_xdata.tolist()
        xobs = xobs.tolist()
        yobs = yobs.tolist()
        for c in range(size):
            xobs.append(add_xdata[c])
            yobs.append(add_ydata[c])
        xobs = np.array(xobs)
        yobs = np.array(yobs)
        gp.fit(xobs, yobs)
        sBset_solu = min(Tfitness)
        if Best_solu > sBset_solu:
            Best_solu = sBset_solu
            if size == 1:
                Best_indi = population[index[0], :]
            else:
                sBset_solu_index = np.argmin(Tfitness)
                add_xdata = np.array(add_xdata)
                Best_indi = add_xdata[sBset_solu_index, :]
        if eval_num >= 200:
            break

    Finalattack_sign = np.zeros((1, 3, 32, 32))
    taget_image = taget_image.cpu().detach().numpy()
    for j in range(0, dim):
       Finalattack_sign[0, :, :, :] = Finalattack_sign[0, :, :, :] + Best_indi[j] * (sample_adv_images[j, :, :, :] - taget_image[0, :, :, :])
    Final_direction = np.sign(Finalattack_sign)
    final_image = taget_image + eps * Final_direction
    final_image = torch.from_numpy(final_image)
    final_image = final_image.float()
    final_image[0, :, :, :] = torch.clamp(final_image[0, :, :, :], -1, 1)

    return final_image, eval_num



import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from config import _C as cfg

class BetaMixtureModel:
    def __init__(self, alphas_init=[1.0, 2.0], betas_init=[2.0, 1.0], weights_init=[0.5, 0.5], epsilon=1e-6):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weights = np.array(weights_init, dtype=np.float64)
        self.epsilon = epsilon
        self.maxx = None
        self.minx = None

    def normalize(self, sequence):
        max_val = np.max(sequence)
        min_val = np.min(sequence)
        self.maxx = max_val
        self.minx = min_val
        if max_val == 0:
            normalized = np.full_like(sequence, 0.5, dtype=np.float64)
        else:
            normalized = (sequence - min_val) / (max_val - min_val)
        normalized = normalized * (1 - 2 * self.epsilon) + self.epsilon
        return normalized

    def weighted_mean(self, x, w):
        return np.sum(w * x) / (np.sum(w) + self.epsilon)

    def fit_beta_weighted(self, x, w):
        x_bar = self.weighted_mean(x, w)
        s2 = self.weighted_mean((x - x_bar)**2, w)
        if s2 < self.epsilon:
            s2 = self.epsilon
        alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
        beta = alpha * (1 - x_bar) / x_bar
        alpha = max(alpha, self.epsilon)
        beta = max(beta, self.epsilon)
        return alpha, beta

    def responsibilities(self, x):
        w_likelihood = np.array([self.weights[k] * stats.beta.pdf(x, self.alphas[k], self.betas[k]) for k in range(2)])
        w_likelihood[w_likelihood <= self.epsilon] = self.epsilon
        total_likelihood = w_likelihood.sum(axis=0)
        r = w_likelihood / (total_likelihood + self.epsilon)
        return r

    def fit(self, sequence, max_iter):
        x = self.normalize(sequence)
        for _ in range(max_iter):
            resp = self.responsibilities(x)
            self.alphas[0], self.betas[0] = self.fit_beta_weighted(x, resp[0])
            if self.betas[0] < 1.02:
                self.betas[0] = 1.1
            self.alphas[1], self.betas[1] = self.fit_beta_weighted(x, resp[1])
            if self.alphas[1] < 1.02:
                self.alphas[1] = 1.1
            self.weights = resp.sum(axis=1)
            self.weights /= self.weights.sum()
        return self

    def predict(self, x):
        x = (x - self.minx) / (self.maxx - self.minx)
        x = x * (1 - 2 * self.epsilon) + self.epsilon
        post = np.array([self.weights[k] * stats.beta.pdf(x, self.alphas[k], self.betas[k]) for k in range(2)])
        post /= post.sum(axis=0) + self.epsilon
        
        # Calculate CDF for each distribution
        cdf = np.array([stats.beta.cdf(x, self.alphas[k], self.betas[k]) for k in range(2)])
        return post[0], post[1], cdf[0], cdf[1]

    def log_likelihood(self, x):
        log_w_likelihood = np.log(np.sum([self.weights[k] * stats.beta.pdf(x, self.alphas[k], self.betas[k]) for k in range(2)], axis=0) + self.epsilon)
        return np.sum(log_w_likelihood)

    def print_beta(self):
        print("Mixture 1, prob: {:.2f}, alpha1: {:.2f}, beta1: {:.2f}".format(self.weights[0], self.alphas[0], self.betas[0]))
        print("Mixture 2, prob: {:.2f}, alpha2: {:.2f}, beta2: {:.2f}".format(self.weights[1], self.alphas[1], self.betas[1]))

def load_wrong_event(cfg):
    if cfg.pretrained == True:
        wrong_event = torch.load("./stage1/{}_pretrained_{}_{}_{}_wrongevent.pt".format(cfg.backbone, cfg.dataset, cfg.noise_mode, cfg.noise_ratio))
    else:
        wrong_event = torch.load("./stage1/{}_scratch_{}_{}_{}_wrongevent.pt".format(cfg.backbone, cfg.dataset, cfg.noise_mode, cfg.noise_ratio))
    return wrong_event

def draw_wrong_event(cfg, wrong_event):
    plt.hist(wrong_event, bins=max(wrong_event), color=['salmon'], edgecolor='black', histtype='barstacked',
             label=["Data's wrong event"])
    plt.xlabel('Wrong Event Distribution')
    plt.ylabel('Fraction of Corresponding Examples')
    plt.title(f'{cfg.dataset} | {cfg.noise_ratio} | {cfg.noise_mode} | Epoch {cfg.epochs}')
    plt.legend()
    if cfg.pretrained == True:
        plt.savefig("./stage1/{}_pretrained_{}_{}_{}_wrongevent.png".format(cfg.backbone, cfg.dataset, cfg.noise_mode, cfg.noise_ratio, cfg.epochs))
    else:
        plt.savefig("./stage1/{}_scratch_{}_{}_{}_wrongevent.png".format(cfg.backbone, cfg.dataset, cfg.noise_mode, cfg.noise_ratio, cfg.epochs))
    plt.show()
    
if __name__ == '__main__':
    wrong_event = torch.load('/mnt/lustre/zhonghuaping.p/zhangkuan/KDD2025/OtherPaperCode/IDO/stage1/resnet50_pretrained_cifar100_sym_0.2_wrongevent.pt')
    bmm_model = BetaMixtureModel()
    bmm_model.fit(wrong_event, max_iter=10)
    bmm_model.print_beta()
    bmm_model.predict([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
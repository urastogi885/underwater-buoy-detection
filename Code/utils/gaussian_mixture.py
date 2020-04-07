import numpy as np
from math import sqrt
from scipy.stats import norm
import matplotlib.pyplot as plt


def gaussian(x_data, means, cov):
    """

    :param x_data:
    :param means:
    :param cov:
    :return:
    """
    data_size = means.shape[0]
    x_hat = (x_data - means).reshape((-1, 1, data_size))
    cov_inv = None
    if np.linalg.det(cov) == 0:
        cov_inv = np.linalg.pinv(cov)
    else:
        cov_inv = np.linalg.inv(cov)
    fac = np.einsum('...k,kl,...l->...', x_hat, cov_inv, x_hat)
    # (1 / (cov * sqrt(2 * np.pi))) * np.exp(-np.power(x_hat, 2.) / (2 * np.power(cov, 2.)))
    # np.exp(-fac / 2) / (np.sqrt((2 * np.pi) ** data_size * np.linalg.det(cov)))
    return np.exp(-fac / 2) / (np.sqrt((2 * np.pi) ** data_size * np.linalg.det(cov)))


# Define class to build a custom gaussian mixture model
class GaussianMixture:
    def __init__(self, dataset, clusters):
        """
        Initilizze gaussian mixture class
        :param dataset:
        :param clusters:
        """
        # Store dataset and clusters as global class variables
        self.data = dataset
        self.clusters = clusters
        # Get dimension of the gaussian model
        self.dim = self.data.shape[1]
        # Initialize the mean
        num = self.data.shape[0]
        # Define empty lists to store means, covariances, and probabilities
        self.mean = []
        self.cov = []
        self.prob = []
        # Retrieve mean and covariance from the dataset
        for k in range(self.clusters):
            self.mean.append(np.sum(self.data[k * num:(k + 1) * num, ...], axis=0) / num)
            self.cov.append(np.identity(self.dim) * 200)
        # Set equal weight for all clusters
        self.weight = np.array([1.0 for _ in range(self.clusters)])
        # Set initial likelihood to 0
        self.likelihood = 0.0

    def expectation(self):
        """

        :return:
        """
        prob_pri = [gaussian(self.data, self.mean[k], self.cov[k]) * self.weight[k] for k in range(self.clusters)]
        # sum up over clusters
        total = sum(prob_pri)
        no_prob = np.ones(total.shape) / self.clusters
        self.prob = [np.where(total != 0, prob_pri[k] / total, no_prob) for k in range(self.clusters)]

    def maximization(self):
        """

        :return:
        """
        for k in range(self.clusters):
            p_total = np.sum(self.prob[k])
            p_weighted = self.prob[k].reshape((-1, 1)) * self.data
            # Update the mean list
            p_weighted_sum = np.sum(p_weighted, axis=0)
            self.mean[k] = p_weighted_sum / p_total
            # Update the covariance list
            p_hat = self.data - self.mean[k]
            p_cov = self.prob[k].reshape((-1, 1, 1)) * p_hat[:, :, None] * p_hat[:, None, :]
            p_cov = np.sum(p_cov, axis=0)
            self.cov[k] = p_cov / p_total
            self.weight[k] = p_total / self.data.shape[0]

    def train(self):
        """

        :return:
        """
        self.expectation()
        self.maximization()

    def get_likelihood(self):
        """

        :return:
        """
        likelihood = [gaussian(self.data, self.mean[k], self.cov[k]) * self.weight[k] for k in range(self.clusters)]
        likelihood = np.sum(np.log(sum(likelihood)))
        return likelihood

    def get_model(self):
        """

        :return:
        """
        return self.mean, self.cov, self.weight

    def get_pdf(self, x_data):
        """

        :param x_data:
        :return:
        """
        pdf = np.zeros(x_data.shape[0])
        for k in range(self.clusters):
            pdf += self.weight[k] * gaussian(x_data, self.mean[k], self.cov[k]).reshape((-1))
        return pdf

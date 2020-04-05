import numpy as np
import matplotlib.pyplot as plt


def gaussian(x_data, means, covariances):
    """

    :param x_data:
    :param means:
    :param covariances:
    :return:
    """
    data_size = means.shape[0]
    x_hat = (x_data - means).reshape((-1, 1, data_size))
    cov_inv = None
    if np.linalg.det(covariances) == 0:
        cov_inv = np.linalg.pinv(covariances)
    else:
        cov_inv = np.linalg.inv(covariances)
    fac = np.einsum('...k,kl,...l->...', x_hat, cov_inv, x_hat)
    return np.exp(-fac / 2) / (np.sqrt((2 * np.pi) ** data_size * np.linalg.det(covariances)))


def generate_data(means, covariances, data_size):
    """

    :param means:
    :param covariances:
    :param data_size:
    :return:
    """
    return np.random.multivariate_normal(means, covariances, data_size)


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

"""
if __name__ == "__main__":
    #  Gaussian model: [mean, cov]
    G1 = [np.array([0, 1]), np.identity(2) * 0.2]
    G2 = [np.array([4, 5]), np.identity(2) * 0.1]
    G_list = [G1, G2]
    # Define the number of clusters
    K = len(G_list)
    # Generate training data
    data_list = [generate_data(G_list[k][0], G_list[k][1], 100) for k in range(K)]  # K*dataNum*dim
    data = np.concatenate(data_list)  # dataNum*dim
    #  Mixture Gaussian model
    n_iterations = 50
    mix = GaussianMixture(data, len(G_list))
    # [mix.train() for i in range(n_iterations)]
    for i in range(n_iterations):
        print("number of iterations: ", i)
        mix.train()
        mean, var, _ = mix.get_model()

        #  Plot results dynamically
        plt.figure(1)
        plt.plot(data[:, 0], data[:, 1], '.')
        circle = []
        for k in range(K):
            circle.append(plt.Circle(mean[k], np.sqrt(var[k][0, 0]), edgecolor='r', facecolor='none'))
            plt.gcf().gca().add_artist(circle[k])
        plt.pause(0.1)
        plt.clf()

    #  Plot final results
    plt.close()
    plt.figure(1)
    plt.plot(data[:, 0], data[:, 1], '.')
    circle = []
    for k in range(K):
            circle.append(plt.Circle(mean[k], np.sqrt(var[k][0, 0]), edgecolor='r', facecolor='none'))
        plt.gcf().gca().add_artist(circle[k])

    #  Generate ground truth
    # x_gt = np.linspace(-10, 15, 300)
    # y_gt_list = [gaussian(x_gt, G_list[k][0], G_list[k][1]) for k in range(K)]
    # y_gt = sum(y_gt_list)/K

    # plt.figure()
    # plt.plot(x_gt, y_gt_list)

    #  Plot results
    plt.figure(2)
    data = np.sort(data, axis=0)
    prob = mix.get_pdf(data)
    print(data.shape)
    plt.plot(data, prob, '.')
    plt.show()
"""

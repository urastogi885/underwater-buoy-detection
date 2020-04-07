import numpy as np
from matplotlib import pyplot as plt


def get_pdf(points, mean_value, cov):
    cov = [cov]
    cov_inv = 1/cov[0]
    cov_inv = [cov_inv]
    diff = points - mean_value
    diff_trans = np.transpose(diff)
    # return pdf value of these points
    return (2.0 * np.pi) ** (-1 / 2.0) * (1.0 / cov[0] ** 0.5) * \
        np.exp(-0.5 * np.multiply(np.multiply(diff_trans, cov_inv), diff))


def gaussian():
    mu1 = 3
    mu2 = 0
    mu3 = 2

    sigma1 = 0.5
    sigma2 = 1
    sigma3 = 0.3

    s1 = np.random.normal(mu1, sigma1, 50)
    s2 = np.random.normal(mu2, sigma2, 50)
    s3 = np.random.normal(mu3, sigma3, 50)
    x_data = np.array(list(s1) + list(s2) + list(s3))

    np.random.shuffle(x_data)

    bins = np.linspace(np.min(x_data), np.max(x_data), 100)

    plt.figure(figsize=(10, 7))
    plt.xlabel("$x$")
    plt.ylabel("pdf")
    plt.scatter(x_data, [0.005] * len(x_data), color='navy', s=30, marker=2, label="Train data")
    plt.title('True pdf ')
    # combined pdf func
    plt.plot(bins, get_pdf(bins, mu1, sigma1) + get_pdf(bins, mu2, sigma2) + get_pdf(bins, mu3, sigma3), color='red',
             lw=2, label="True pdf")
    plt.legend(loc='upper right')
    plt.savefig('True_pdf_with.png')

    return bins, mu1, mu2, mu3, sigma1, sigma2, sigma3, x_data


def expectation_maximization():
    bins, mu1, mu2, mu3, sigma1, sigma2, sigma3, x_data = gaussian()
    k = 3
    weights = np.ones(k) / k
    means = np.random.choice(x_data, k)
    variances = np.random.random_sample(size=k)
    # visualize the training data
    k = 2
    z = 0
    eps = 0.00001
    iterations = 50
    iterations_completed = 0
    log_likelihood = []
    while iterations_completed < iterations:
        z += 1
    # calculate the maximum likelihood of each observation xi
        likelihood_list = []
        log_likelihood_list = []

        # Expectation step
        for j in range(k):
            likelihood_list.append(get_pdf(x_data, means[j], np.sqrt(variances[j])))
            log_likelihood_list.append(get_pdf(x_data, means[j], np.sqrt(variances[j]))*weights[j])

        likelihood_list = np.array(likelihood_list)
        log_likelihood_list = np.array(log_likelihood_list).T
        likelihood_sum = np.sum(log_likelihood_list, axis=1)
        log_value = np.sum(np.log(likelihood_sum))
        log_likelihood.append(log_value)

        b = []

        # Maximization step
        for j in range(k):
            # use the current values for the parameters to evaluate the posterior
            # probabilities of the data to have been generanted by each gaussian
            b.append((likelihood_list[j] * weights[j]) / (np.sum([likelihood_list[i] * weights[i] for i in range(k)],
                                                                 axis=0) + eps))
            # update mean and variance
            means[j] = np.sum(b[j] * x_data) / (np.sum(b[j]+eps))
            variances[j] = np.sum(b[j] * np.square(x_data - means[j])) / (np.sum(b[j] + eps))
            # update the weights
            weights[j] = np.mean(b[j])

        if len(log_likelihood) < 2:
            continue
        if np.abs(log_value - log_likelihood[-2]) < eps:
            plt.figure(figsize=(10, 7))
            plt.xlabel("$x$")
            plt.ylabel("pdf")
            plt.title(" Best result reached at Iteration {}".format(iterations_completed))
            plt.scatter(x_data, [0.005] * len(x_data), color='navy', s=30, marker=2,
                        label="Randomly generated data points")
            # Draw true pdf
            plt.plot(bins, get_pdf(bins, mu1, sigma1) + get_pdf(bins, mu2, sigma2) + get_pdf(bins, mu3, sigma3),
                     color='red', lw=2, label="True pdf")
            # Draw pdf after training
            plt.plot(bins, get_pdf(bins, means[0], variances[0]) + get_pdf(bins, means[1], variances[1]) +
                     get_pdf(bins, means[2], variances[2]), color='blue', lw=2, label="Trained pdf")
            plt.legend(loc='upper left')
            plt.savefig('pdf_after_training.png')
            plt.show()
            break
        iterations_completed += 1


if __name__ == "__main__":
    # Generate plots to show successful implementation of expectation maximization algorithm
    expectation_maximization()
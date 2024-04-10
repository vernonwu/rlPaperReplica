import numpy as np
from alive_progress import alive_it
import random
import matplotlib.pyplot as plt

random_seed = 42
np.random.seed(random_seed)
# start at D
start_pos = 2
# training sets
batch = 100
batch_size = 10
# theoretical probability
P_ideal = np.array([1/6, 1/3, 1/2, 2/3, 5/6])

lambdas = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]

def prediction(w, current_pos):
    return w[current_pos]

def state_value(w, current_pos):
    if current_pos == -1:
        return 0
    elif current_pos == 5:
        return 1
    else:
        return prediction(w, current_pos)

def update_weights(w, current_pos, next_pos, alpha, lamb, e):
    delta_e = np.zeros(5)
    delta_e[current_pos] = 1
    e = lamb * e + delta_e
    delta_w = alpha * (state_value(w, next_pos) - state_value(w, current_pos)) * e
    return delta_w, e

def generate_sequence(start_pos = 2):
    current_pos = start_pos
    sequence = [current_pos]
    while True:
        action = np.random.choice([-1,1])
        current_pos += action
        sequence.append(current_pos)
        if current_pos == -1 or current_pos == 5:
            return sequence

def generate_data(batch, batch_size):
    data = []
    for i in range(batch):
        batch_data = []
        for j in range(batch_size):
            batch_data.append(generate_sequence())
        data.append(batch_data)
    return data

data = generate_data(batch, batch_size)

random.sample(data[0], 3)

def rms_error(batch, w, P_ideal):
    err = 0
    for sequence in batch:
        err_list = []
        for x in sequence[:-1]:
            err_list.append(prediction(w, x) - P_ideal[x])
        err += np.sqrt(np.mean(np.square(err_list)))

    err /= len(batch)

    return err

def train(data, alpha, lamb,convergence_threshold):

    # initialize weights
    err = 0
    for batch in alive_it(data):
        w = np.random.uniform(0,1,5)

        while True:
            delta_w = np.zeros(5)
            for sequence in batch:
                e = 0
                for idx in range(len(sequence)-1):
                    res =  update_weights(w, sequence[idx], sequence[idx+1], alpha, lamb, e)
                    delta_w += res[0]
                    e = res[1]
            w += delta_w
            if np.linalg.norm(delta_w) < convergence_threshold:
                err += rms_error(batch, w, P_ideal)
                break
    err /= len(data)

    return err

def plot_rms_error(lambdas, alpha, data,convergence_threshold):
    rms_errors = []
    for lamb in lambdas:
        err = train(data, alpha, lamb,convergence_threshold)
        rms_errors.append(err)
    plt.plot(lambdas, rms_errors)
    plt.xlabel('$\lambda$')
    plt.ylabel('ERROR')
    plt.scatter(lambdas, rms_errors, color='b')
    plt.show()

# hyperparams
alpha = 0.01
convergence_threshold = 0.01
plot_rms_error(lambdas, alpha, data,convergence_threshold)

lrs = np.linspace(0, 0.6, 13)
lambdas_2 = [0.0, 0.3, 0.8, 1.0]
def train_2(data, alpha, lamb):

    # initialize weights to 0.5
    err = 0
    for batch in data:
        w = np.full(5, 0.5)
        for sequence in batch:
            delta_w = np.zeros(5)
            e = 0
            for idx in range(len(sequence)-1):
                res =  update_weights(w, sequence[idx], sequence[idx+1], alpha, lamb, e)
                delta_w += res[0]
                e = res[1]
            w += delta_w
        err += rms_error(batch, w, P_ideal)
    err /= len(data)

    return err


def plot_rms_error_2(lambdas_2, lrs, data):
    threshold = 0.8
    for lamb in lambdas_2:
        rms_errors = []
        for lr in alive_it(lrs):
            err = train_2(data, lr, lamb)
            if err > threshold:
                break
            rms_errors.append(err)
        plt.plot(lrs[:len(rms_errors)], np.array(rms_errors), label='$\lambda$ = ' + str(lamb))
        plt.scatter(lrs[:len(rms_errors)], np.array(rms_errors))
    plt.legend()
    plt.xlabel('α')
    plt.ylabel('ERROR')
    plt.show()

    lambdas_3 = np.linspace(0, 1, 11)
def plot_rms_error_3(lambdas_3, lrs, data):
    err_list = []
    # choose lr that yields the least error for each lambda
    for lamb in lambdas_3:
        least_err = 1
        for lr in alive_it(lrs):
            err = train_2(data, lr, lamb)
            if err < least_err:
                least_err = err
        err_list.append(least_err)
    print('The best lambda is around', round(lambdas_3[np.argmin(err_list)],3))
    plt.plot(lambdas_3, err_list)
    plt.scatter(lambdas_3, err_list)
    plt.xlabel('$\lambda$')
    plt.ylabel('ERROR USING BEST α')
    plt.show()

plot_rms_error_3(lambdas_3, lrs, data)
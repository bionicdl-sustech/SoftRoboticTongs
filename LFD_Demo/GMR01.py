from Utils.gmr import Gmr, plot_gmm


def loadData(dataSet, fileName, dataSize):
    index = 1
    timeStep = 0
    x = []
    y = []
    z = []
    position = []
    with open(fileName, 'r', encoding='utf8') as fp:
        json_data = json.load(fp)
        while index <= len(json_data) - len(json_data) % dataSize:
            string = "serialNum" + str(index)
            index += 1

            # change the data used
            if (index % ((len(json_data) - len(json_data) % dataSize) / dataSize) == 0):
                x_data = json_data[string][1]['level_1'][0]['X'] * 400
                y_data = -json_data[string][1]['level_1'][0]['Y'] * 400
                bdlData = [0.01 * timeStep, x_data, y_data]
                position.append(bdlData)
                x.append(x_data)  # X data
                y.append(y_data)
                z.append(0)
                timeStep += 1
    dataSet.append(position)
    dataSet = np.array(dataSet, dtype=np.float64)

    plt.plot(x, y, color=[.7, .7, .7])
    plt.show()
    return 0


# GMR on 2D trajectories with time as input
if __name__ == '__main__':
    import json
    import matplotlib.pyplot as plt
    import numpy as np

    # Parameters
    nb_data = 200
    nb_data_sup = 15
    nb_samples = 5
    dt = 0.01
    input_dim = 1
    output_dim = 2
    in_idx = [0]
    out_idx = [1, 2]
    nb_states = 6

    dataSet = []
    fileName = 'Trajectory data/L1.json'
    loadData(dataSet, fileName, nb_data)
    fileName = 'Trajectory data/L2.json'
    loadData(dataSet, fileName, nb_data)
    fileName = 'Trajectory data/L3.json'
    loadData(dataSet, fileName, nb_data)
    fileName = 'Trajectory data/L4.json'
    loadData(dataSet, fileName, nb_data)
    fileName = 'Trajectory data/L5.json'
    loadData(dataSet, fileName, nb_data)

    # Create time data
    # Stack time and position data
    demos_tx = dataSet

    # Stack demos
    demos_np = demos_tx[0]

    for i in range(1, nb_samples):
        demos_np = np.vstack([demos_np, demos_tx[i]])

    X = demos_np[:, 0][:, None]
    Y = demos_np[:, 1:]
    # Test data
    Xt = dt * np.arange(nb_data + nb_data_sup)[:, None]
    # GMM
    gmr_model = Gmr(nb_states=nb_states, nb_dim=input_dim + output_dim, in_idx=in_idx, out_idx=out_idx)
    gmr_model.init_params_kbins(demos_np.T, nb_samples=nb_samples)
    gmr_model.gmm_em(demos_np.T)

    # GMR
    mu_gmr = []
    sigma_gmr = []
    for i in range(Xt.shape[0]):
        mu_gmr_tmp, sigma_gmr_tmp, H_tmp = gmr_model.gmr_predict(Xt[i])
        mu_gmr.append(mu_gmr_tmp)
        sigma_gmr.append(sigma_gmr_tmp)

    mu_gmr = np.array(mu_gmr)
    sigma_gmr = np.array(sigma_gmr)

    plt.figure(figsize=(5, 5))
    for p in range(nb_samples):
        plt.plot(Y[p * nb_data:(p + 1) * nb_data, 0], Y[p * nb_data:(p + 1) * nb_data, 1], color=[.7, .7, .7])
        plt.scatter(Y[p * nb_data, 0], Y[p * nb_data, 1], color=[.7, .7, .7], marker='X', s=80)
    plt.plot(mu_gmr[:, 0], mu_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=2)
    plt.scatter(mu_gmr[0, 0], mu_gmr[0, 1], color=[0.20, 0.54, 0.93], marker='X', s=80)
    plot_gmm(mu_gmr, sigma_gmr, alpha=0.05, color=[0.20, 0.54, 0.93])
    axes = plt.gca()
    plt.xlabel('$y_1$', fontsize=20)
    plt.ylabel('$y_2$', fontsize=20)
    plt.locator_params(nbins=3)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig("./Output/L.png")
    plt.show()

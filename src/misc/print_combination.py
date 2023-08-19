from itertools import combinations
import os


if __name__ == "__main__":
    # entry point main

    data_flag = 'pneumoniamnist'
    real_flag = 0
    init_w = "uniform"
    n_samples = 1
    early_stop_metric = 'auc'
    suffix_dir = 'exp_0'
    configs_dir = 'src/configs/'
    gan_list = os.listdir(os.path.join(configs_dir, data_flag))
    gan_list = [x.split(".yaml")[0] for x in gan_list]
    gan_iter = [str(x) for x in range(5000, 100000, 10000)] #define here the step

    tot = 0
    for gan_ensemble in gan_list:
        for num_iter in range(len(gan_iter) + 1):
            for iter_combination in combinations(gan_iter, num_iter):

                if len(iter_combination) < 1:
                    continue
                num_iter = ",".join(iter_combination)
                print(
                    f"data_flag='{data_flag}' early_stop_metric='{early_stop_metric}' suffix_dir='{suffix_dir}' real_flag={real_flag} init_w='{init_w}' gan_ensemble='{gan_ensemble}' gan_epochs='{num_iter}' n_samples={n_samples}")
                tot += 1
    print(f"Number of combinations: {tot - 1}")
    print("\n")
from math import sqrt
import json
import numpy as np
import matplotlib.pyplot as plt


def t_test(mu1, mu2, s1, s2, n: int):
    '''
    Welch's t-test: independent two-sample t-test, one-sided
    sample group 1: attn wrt patches inside bboxes
    sample group 2: attn wrt patches outside bboxes
    H_0: mu_1 <= mu_2, and we want to reject it
    '''
    unbiased_s = (s1**2+s2**2)/n
    t = (mu1-mu2)/sqrt(unbiased_s)
    # degree of freedom
    df = unbiased_s**2/(((s1**2/n)**2+(s1**2/n)**2)/(n-1))
    return t, df


def plot_CI(surprisal_simple, surprisal_normal):
    if isinstance(surprisal_normal, list):
        surprisal_simple = np.array(surprisal_simple)
        surprisal_normal = np.array(surprisal_normal)

    layers = np.arange(1, 13)

    def compute_ci(data, z=1.96):
        assert data.shape[0] == 12
        mean = np.mean(data, axis=1)  # (12,)
        std = np.std(data, axis=1, ddof=1)
        n = data.shape[1]
        ci = z * std / np.sqrt(n)
        return mean, ci

    mean_simple, ci_simple = compute_ci(surprisal_simple)
    mean_normal, ci_normal = compute_ci(surprisal_normal)

    plt.figure(figsize=(10, 6))

    # 绘图 - 用误差棒 error bars 表示 CI，而不是 fill_between
    plt.errorbar(layers, mean_simple, yerr=ci_simple, label='simple context',
                 fmt='-o', capsize=4, color='royalblue')
    plt.errorbar(layers, mean_normal, yerr=ci_normal, label='normal context',
                 fmt='-o', capsize=4, color='darkorange')

    # 添加每个数据点上的数值标签
    for i, val in enumerate(mean_simple):
        plt.text(layers[i], val + 0.05, f'{val:.4f}', fontsize=8,
                 ha='center', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1.5))
    for i, val in enumerate(mean_normal):
        plt.text(layers[i], val + 0.05, f'{val:.4f}', fontsize=8,
                 ha='center', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1.5))

    plt.xlabel('Layer')
    plt.ylabel('Surprisal')
    plt.title('Surprisal with 95% Confidence Interval')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figure/Attached_Lens_CE_Loss_with_CI.png')
    plt.show()


def read_surprisal(option = 'normal'):
    context_file = ['', '2', '5_0', '5_1', '5_2', '5_3', '5_4', '6_0', '6_1', '6_2']
    with open(f'surprisal_result/vocab_prob/{option}_context.json') as f:
        data = json.load(f)

    stat_ls = []  # every layer has 1000 samples
    layer_all = []
    for layer in range(12):
        layer_ls = []
        for cf in context_file:
            surprisal = data[str(layer)][f'context{cf}'].values()
            layer_ls += list(surprisal)
        layer_arr = np.array(layer_ls)
        layer_all.append(layer_ls)
        stat_ls.append([np.mean(layer_arr), np.std(layer_arr), len(layer_ls)])
    
    return stat_ls, layer_all


if __name__ == '__main__':
    # H0 hypothesis: layer 8 surprisal < layer 7 surprisal
    # we want to reject it. One-sided Welch's t-test
    stat_normal, layer_normal = read_surprisal()
    stat_simple, layer_simple = read_surprisal(option='simple')
    plot_CI(layer_simple, layer_normal)

    # for i in range(1, 11):
    #     print(f'Now comparing layer {i+1} and {i+2}...')
    #     t, df = t_test(mu1=stat_ls[i][0], mu2=stat_ls[i+1][0], s1=stat_ls[i][1], s2=stat_ls[i+1][1], n=stat_ls[i][2])
    #     print(t, df)
    
    # for (i, j) in [(7, 9), (7, 10)]:
    #     print(f'Now comparing layer {i+1} and {j+1}...')
    #     t, df = t_test(mu1=stat_ls[i][0], mu2=stat_ls[j][0], s1=stat_ls[i][1], s2=stat_ls[j][1], n=stat_ls[i][2])
    #     print(t, df)
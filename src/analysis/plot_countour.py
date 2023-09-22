import sys
sys.path.extend([
    "./",
])

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc

# For aspect ratio 4:3.
column_width_pt = 516.0
pt_to_inch = 1 / 72.27
column_width_inches = column_width_pt * pt_to_inch
aspect_ratio = 4 / 3
sns.set(style="whitegrid", font_scale=1.6, rc={"figure.figsize": (column_width_inches, column_width_inches / aspect_ratio)})
#sns.set_context("paper")
#sns.set_theme(style="ticks")
# For Latex.
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


# main
if __name__ == "__main__":

    reports_dir = './reports/'
    dataset_name = 'pneumoniamnist'
    filename =  'umap_imgs_train_ResNet50_torch_friendly_50000' # 'umap_train_InceptionV3_torch_friendly_50000' # 'umap_train_SwAV_torch_friendly_50000' # 'umap_train_cnn_resnet_50_pneumoniamnist_friendly_50000' # 'umap_train_resnet_ae_50_pneumoniamnist_friendly_50000'

    # Load dataframe.
    df = pd.read_excel(os.path.join(os.path.join(reports_dir, dataset_name, 'umap_embeddings', filename + '.xlsx')), engine='openpyxl')

    # Select df_real.
    df_real = df[df['exp'] == 'real']
    df_naive = df[df['exp'] == 'naive']
    df_DGE = df[df['exp'] == 'DGE']

    # Countour plot.
    # Single plot.
    levels_list= [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        [0.6, 0.7, 0.8, 0.9, 1.0],
        [0.7, 0.8, 0.9, 1.0],
        [0.8, 0.9, 1.0],
        [0.9, 1.0],
    ]
    alpha_val = 0.7
    for levels in levels_list:

        fig = plt.figure()
        sns.kdeplot(data=df_real, x="umap-1", y="umap-2", levels=levels, color='blue', fill=True,  alpha=.5, label=r'Real')
        sns.kdeplot(data=df_naive, x="umap-1", y="umap-2", levels=levels, color='red', fill=True,  alpha=.5, label=r'Naive')
        sns.kdeplot(data=df_DGE, x="umap-1", y="umap-2",levels=levels, color='green', fill=True, alpha=.5, label=r'DGE')
        leg_ele = [
           plt.Line2D([0], [0], color=['blue', 'red', 'green'][i], lw=2, label=label, alpha=.5) for i, label in enumerate(['Real', 'Naive', 'DGE'])]
        plt.legend(handles=leg_ele)
        plt.xlabel(r'umap-1')
        plt.ylabel(r'umap-2')
        plt.tight_layout()
        fig.savefig(os.path.join(reports_dir, f'kde_{filename}_{levels}.pdf'), dpi=400, format='pdf', bbox_inches='tight')
        plt.show()

        g = sns.jointplot(data=df_real, x='umap-1', y='umap-2', kind='kde', color="blue", fill=True, alpha=alpha_val, zorder=0, marginal_kws=dict(fill=False))

        sns.kdeplot(x=df_naive['umap-1'], y=df_naive['umap-2'], ax=g.ax_joint, color="red", fill=True, alpha=alpha_val, levels=levels, zorder=1)
        sns.kdeplot(x=df_naive['umap-1'], ax=g.ax_marg_x, color='red', alpha=alpha_val, levels=levels)
        sns.kdeplot(y=df_naive['umap-2'], ax=g.ax_marg_y, color='red', alpha=alpha_val, levels=levels)

        sns.kdeplot(x=df_DGE['umap-1'], y=df_DGE['umap-2'], ax=g.ax_joint, color="green", fill=True, alpha=0.5,  levels=levels, zorder=2)
        sns.kdeplot(x=df_DGE['umap-1'], ax=g.ax_marg_x, color='green', alpha=alpha_val, levels=levels)
        sns.kdeplot(y=df_DGE['umap-2'], ax=g.ax_marg_y, color='green',alpha=alpha_val, levels=levels)

        #g.ax_joint.grid(False)
        #g.set_axis_labels("", "")
        g.ax_joint.set_xticks([])
        g.ax_joint.set_yticks([])
        g.ax_marg_x.set_xticks([])
        g.ax_marg_y.set_yticks([])

        leg_ele = [
                plt.Line2D([0], [0], color=['blue', 'red', 'green'][i], lw=2, label=label, alpha=.5) for i, label in enumerate(['Real', 'Naive', 'DGE'])]
        plt.legend(handles=leg_ele)
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, f'joint_{filename}_{levels}.pdf'), dpi=400, format='pdf',  bbox_inches='tight')
        plt.show()

    # Subplots.
    # 3 plots, one for each df.
    colors = ['blue', 'red', 'green']
    labels = ['Real', 'Naive', 'DGE']
    levels = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    sns.kdeplot(data=df_real, x="umap-1", y="umap-2", levels=levels, color=colors[0], fill=True, ax=axes[0])
    axes[0].set_title(labels[0])

    sns.kdeplot(data=df_naive, x="umap-1", y="umap-2", levels=levels, color=colors[1], fill=True, ax=axes[1])
    axes[1].set_title(labels[1])

    sns.kdeplot(data=df_DGE, x="umap-1", y="umap-2", levels=levels, color=colors[2], fill=True, ax=axes[2])
    axes[2].set_title(labels[2])

    leg_ele = [plt.Line2D([0], [0], color=color, lw=2, label=label, alpha=.5) for color, label in zip(colors, labels)]
    axes[2].legend(handles=leg_ele, loc='upper left')

    # Set common labels
    for ax in axes:
        ax.set_xlabel(r'umap-1')
    axes[0].set_ylabel(r'umap-2')
    plt.tight_layout()  # To adjust layout to accommodate the legend
    fig.savefig(os.path.join(reports_dir, f'kde_subplots_{filename}_{levels}.pdf'), dpi=400, format='pdf', bbox_inches='tight')
    plt.show()

    # fig = px.density_contour(df_real, x="umap-1", y="umap-2")
    # fig.update_traces(contours_coloring="fill", contours_showlabels=True)
    # fig.show()
    # fig.write_html(os.path.join(cache_dir, f'{filename}_contour.html'))

    print('May the force be with you.')
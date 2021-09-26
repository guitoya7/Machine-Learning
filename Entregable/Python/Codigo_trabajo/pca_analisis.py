from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
def pca_analisis(data):
    pca_pipe = make_pipeline(PCA())
    pca_pipe.fit(data[data.columns[:-1]])
    modelo_pca = pca_pipe.named_steps['pca']
    pca=pd.DataFrame(
    data=modelo_pca.components_,
    columns=data.columns[:-1])
    # Porcentaje de varianza explicada acumulada
    # ==============================================================================
    prop_varianza_acum = modelo_pca.explained_variance_ratio_.cumsum()
    print('------------------------------------------')
    print('Porcentaje de varianza explicada acumulada')
    print('------------------------------------------')
    print(prop_varianza_acum)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    ax.plot(
        np.arange(len(data.columns[:-1])) + 1,
        prop_varianza_acum,
        marker='o'
    )

    for x, y in zip(np.arange(len(data.columns[:-1])) + 1, prop_varianza_acum):
        label = round(y, 2)
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )

    ax.set_ylim(0, 1.1)
    ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
    ax.set_title('Porcentaje de varianza explicada acumulada')
    ax.set_xlabel('Componente principal')
    ax.set_ylabel('Por. varianza acumulada');
    plt.show()
    fig_2, ax_2 = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    componentes = modelo_pca.components_
    plt.imshow(componentes[0:6].T, cmap='PuOr', aspect='auto', vmin=-1, vmax=1)
    plt.yticks(range(len(data.columns[:-1])), data.columns[:-1])
    plt.xticks(range(6))
    plt.grid(False)
    plt.colorbar();
    plt.show()
    data = data.drop(['A_mentions_sent','B_mentions_sent','A_retweets_sent','B_retweets_sent','A_posts','B_posts'], axis=1)
    return data
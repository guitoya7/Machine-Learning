import seaborn as sns
import matplotlib.pyplot as plt
def funar_variables(data):
    plt.figure(figsize=(30, 30))
    sns.heatmap(data.corr(),
                vmin=-1,
                vmax=1,
                cmap=sns.diverging_palette(145, 280, s=85, l=25, n=7),
                annot=True,
                square=True,
                linewidths=.5)
    plt.show()
    data=data.drop(['B_network_feature_1','A_network_feature_1','B_listed_count','A_listed_count','A_mentions_received','B_mentions_received'],axis=1)
    return(data)
import matplotlib.pyplot as plt
import seaborn as sns
def Analisis_errores(simple_data,modelo_df,data_todo):
    fallos=data_todo[modelo_df.predict(simple_data.drop('Choice',1))!=simple_data['Choice']]
    plt.figure(figsize=(20, 20))
    sns.scatterplot(x=fallos['A_follower_count'],
                    y=fallos['B_follower_count'],
                    s=100,
                    hue=fallos['Choice']);
    plt.show()
    plt.figure(figsize=(20, 20))
    sns.scatterplot(x=fallos['A_retweets_received'],
                    y=fallos['B_retweets_received'],
                    s=100,
                    hue=fallos['Choice']);
    plt.show()
    sns.scatterplot(x=fallos['A_retweets_received'],
                    y=fallos['B_retweets_received'],
                    s=100,
                    hue=fallos['Choice']);
    plt.show()
    plt.figure(figsize=(20, 20))
    sns.scatterplot(x=fallos['A_listed_count'],
                    y=fallos['B_listed_count'],
                    s=100,
                    hue=fallos['Choice']);
    plt.show()
    plt.figure(figsize=(20, 20))
    sns.scatterplot(x=fallos['A_posts'],
                    y=fallos['B_posts'],
                    s=100,
                    hue=fallos['Choice']);
    plt.show()
    plt.figure(figsize=(20, 20))
    sns.scatterplot(x=fallos['A_mentions_sent'],
                    y=fallos['B_mentions_sent'],
                    s=100,
                    hue=fallos['Choice']);
    plt.show()
    plt.figure(figsize=(20, 20))
    sns.scatterplot(x=fallos['A_retweets_sent'],
                    y=fallos['B_retweets_sent'],
                    s=100,
                    hue=fallos['Choice']);
    plt.show()
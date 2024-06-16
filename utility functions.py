'''plot'''
# from sklearn.metrics import silhouette_score
def plot_latent_space(encoder, x, y, name='mu'):
    s_mean, s_var, s = encoder.predict(x, batch_size=1280)
    ss_mean = round(silhouette_score(s_mean, y), 2)
    ss_s = round(silhouette_score(s, y), 2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2,  figsize=(12, 5))
    
    for label in np.unique(y):
        ax1.scatter(s_mean[y == label, 0], s_mean[y == label, 1], label=label, alpha = 0.5)
    #ax1.set_title(name + ' salient latent variable_mu, Silhouette score: ' + str(ss_mean))
    ax1.set_title(name + ' salient latent variable_mu, SS: ' + str(ss_mean))
    ax1.set_xlabel("mu 1")
    ax1.set_ylabel("mu 2")
    ax1.legend()
    

    for label in np.unique(y):
        ax2.scatter(s[y == label, 0], s[y == label, 1], label=label, alpha = 0.5)
    #ax2.set_title(name + ' salient latent variable, Silhouette score: ' + str(ss_s))
    ax2.set_title(name)
    ax2.set_xlabel("Salient latent variable 1")
    ax2.set_ylabel("Salient latent variable 2")
    ax2.legend()
    
    #plt.scatter(z_mean[:, 0],z_mean[:, 1], c=y, cmap='Accent')
    plt.tight_layout()
    plt.show()
    #return ss

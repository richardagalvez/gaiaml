# coding: utf-8
import numpy as np; import pickle as pkl

#x, likelihoods, sigma_square, parallax = pkl.load(open('quickdumpdskfjsdlkjfs_v2.pkl', 'rb'))
#x, likelihoods, sigma_square, parallax = pkl.load(open('quickdumpdskfjsdlkjfs_array.pkl', 'rb'))
data = pkl.load(open('quickdump_new_slidjflsdjlf.pkl', 'rb'))

x, likelihoods, sigma_square, parallax = [], [], [], []

for data_chunk in data:
    x.append(data_chunk[0])
    likelihoods.append(data_chunk[1])
    sigma_square.append(data_chunk[2])
    parallax.append(data_chunk[3])

# x has this!
#cols_in_use = ['l', 'b', 'phot_g_mean_mag', 'g_bp', 'g_rp', 'pmra', 'pmdec', 'visibility_periods_used', 'parallax']
x = np.concatenate(x)
og_parallax = x[:, -1]
likelihoods = np.concatenate(likelihoods)
sigma_square = np.concatenate(sigma_square)
parallax = np.concatenate(parallax)

print(x.shape, flush=True)
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

cleaned_parallax = parallax.copy()

for parallax, name in zip([og_parallax, cleaned_parallax], ['og_parallax', 'cleaned_parallax']):

    print("Running on", name, flush=True)
    good_parallax_mask = (parallax >0) & (parallax < 10000) & (parallax > sigma_square)
    print("Number good parallaxes:", good_parallax_mask.sum())
    bkupparallax = parallax.copy(); sigmabackup = sigma_square.copy(); xbkup = x.copy();
    parallax = parallax[good_parallax_mask].copy(); x_curr=x[good_parallax_mask].copy(); sigma_square_curr=sigma_square[good_parallax_mask].copy()
    smin = np.min(sigma_square_curr)
    smax = np.max(sigma_square_curr)
    #smin=np.min(sigma_square_curr[(x_curr[:, -1]>-1)&(x_curr[:, 1]<1)&(parallax>-1)&(parallax<1)])
    #smax=np.max(sigma_square_curr[(x_curr[:, -1]>-1)&(x_curr[:, 1]<1)&(parallax>-1)&(parallax<1)])

    #plt.figure(); plt.scatter(x_curr=x_curr[:, -1][::1000], y=parallax[::1000], s=0.5, c=sigma_square_curr[::1000], norm=mpl.colors.LogNorm(vmin=smin,vmax=smax)); plt.colorbar(); plt.plot(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)); plt.xlim(-1, 1); plt.ylim(-1, 1); plt.savefig('every_thousandth_parallaxv2.png')
    #print("X is ra,dec,g,g-bp,g-rp,pmra,pmdec,visperiods")
    #plt.close()

    distances_pc = 1/(parallax*1e-3)

    x_mean = np.r_[134.704, 3.842, 18.3667, -0.4447, 0.70889, -2.29007232, -1.24711091, 12.53824764, 0.185974]
    x_scale = np.r_[28.7076653, 23.58792102, 1.79013134, 0.22338684, 0.22823019, 4.2702091, 5.29038589, 2.46977188, 0.69567963]

    G = x_curr[:, 2]*x_scale[2]+x_mean[2]; g_bp = x_curr[:, 3]*x_scale[3]+x_scale[3]
    g = G - 5*np.log10(distances_pc/10 + 1)


    g_rp = x_curr[:, 4]*x_scale[4] + x_mean[4]
    g_bp = x_curr[:, 3]*x_scale[3] + x_mean[3]
    bp_rp = g_rp - g_bp

    plt.figure(); plt.scatter(x=bp_rp[::10], y=g[::10], s=0.1, c=parallax, alpha=0.2);plt.gca().invert_yaxis(); plt.savefig('colordiagram_' + name + '.png'); 
    print("Plotting!", flush=True)
    plt.close()


# Make for old parallax!

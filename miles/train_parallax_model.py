import numpy as np
import tensorflow as tf
import pickle as pkl
from astropy.table import Table
import pandas as pd
import tables
import sys
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
assert torch.cuda.device_count() > 0

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
data_parallel = True

t = Table.read('../../GD-1/gd1-with-masks.fits')

df = t.to_pandas()
df['g_bp'] = df['phot_g_mean_mag'] - df['phot_bp_mean_mag']
df['g_rp'] = df['phot_g_mean_mag'] - df['phot_rp_mean_mag']
#HACK use l and b here instead of ra and dec
cols_in_use = ['ra', 'dec', 'phot_g_mean_mag', 'g_bp', 'g_rp', 'pmra', 'pmdec', 'visibility_periods_used', 'parallax']
df = df[cols_in_use]
data = np.array(df)

# Convert ra/dec to l/b
from astropy import units as u
from astropy.coordinates import SkyCoord
c = SkyCoord(ra=np.array(df['ra'])*u.degree, dec=np.array(df['dec'])*u.degree, frame='icrs')
data[:, 0] = np.array(c.galactic.l.degree)
data[:, 1] = np.array(c.galactic.b.degree)

# Available columns:
#Index(['source_id', 'ra', 'dec', 'parallax', 'parallax_error', 'pmra',
#'pmra_error', 'pmdec', 'pmdec_error', 'ra_parallax_corr',
#'ra_pmra_corr', 'ra_pmdec_corr', 'dec_parallax_corr', 'dec_pmra_corr',
#'dec_pmdec_corr', 'parallax_pmra_corr', 'parallax_pmdec_corr',
#'pmra_pmdec_corr', 'visibility_periods_used', 'phot_g_mean_mag',
#'phot_g_mean_flux_over_error', 'phot_bp_mean_mag',
#'phot_bp_mean_flux_over_error', 'phot_rp_mean_mag',
#'phot_rp_mean_flux_over_error', 'phot_bp_rp_excess_factor',
#'astrometric_chi2_al', 'astrometric_n_good_obs_al', 'g', 'g_error',
#'g0', 'r', 'r_error', 'r0', 'i', 'i_error', 'i0', 'z', 'z_error', 'z0',
#'y', 'y_error', 'y0', 'pm_mask', 'gi_cmd_mask', 'phi1', 'phi2',
#'pm_phi1_cosphi2_no_reflex', 'pm_phi2_no_reflex', 'pm_phi1_cosphi2',
#'pm_phi2', 'stream_track_mask'],

print("New run over architectures with split over streams not stars")

# These work the best!
n_nodes = 100
n_layers = 10

#for (n_nodes, n_layers) in [(x, y) for x in [20, 50, 100, 150] for y in [2, 5, 7, 10, 15, 30]]:
# HACK This goes all the way done to the end of the file!

scaler = preprocessing.StandardScaler()
scaler.fit(data)
print(scaler.mean_, scaler.scale_)
data = scaler.transform(data)
dtype = torch.cuda.FloatTensor
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def generate_more_data():

    N_train = 800000
    N_test = 200000

    train_idx = np.random.randint(low=0, high=len(data)-1, size=N_train)
    valid_idx = np.random.randint(low=0, high=len(data)-1, size=N_test)
    x = data[train_idx].copy()
    


    #x[:, -1] = np.random.normal(size=len(x))[:] #Put in random numbers instead of parallax.
    #HACK - leave in parallax

    #y = data[train_idx, -1] # Predict parallax!
    #HACK - make 1s as output!
    #y = data[train_idx, -1] # Predict parallax!
    y = np.zeros(len(train_idx))  

    x_valid = data[valid_idx].copy()
    #x_valid[:, -1] = np.random.normal(size=len(x_valid))[:] #Put in random numbers instead of parallax.


    #y_valid = data[valid_idx, -1] # Predict parallax!
    #HACK!
    y_valid = np.zeros(len(valid_idx))  

    x = Variable(torch.from_numpy(x)).type(dtype)
    y = Variable(torch.from_numpy(y), requires_grad=False).type(dtype)
    x_valid = Variable(torch.from_numpy(x_valid)).type(dtype)
    y_valid = Variable(torch.from_numpy(y_valid),requires_grad=False).type(dtype)
    #------------------------------------------------------------------------------
    # run cuda
    kwargs = {}
    #HACK - need this for the following to get multi-device!
    print("Done generating data", flush=True)
    if data_parallel:
        with torch.cuda.device(0): #Bottom 4 snippets!
            x = x.cuda(**kwargs)
            y=y.cuda(**kwargs)
            x_valid=x_valid.cuda(**kwargs)
            y_valid=y_valid.cuda(**kwargs)
    else:
        x = x.cuda(**kwargs)
        y=y.cuda(**kwargs)
        x_valid=x_valid.cuda(**kwargs)
        y_valid=y_valid.cuda(**kwargs)
    return x, y, x_valid, y_valid

# # Move into validation set:
x, y, x_valid, y_valid = generate_more_data()
# Let's batch this!

flatten = lambda l: [subitem for item in l for subitem in item]
import time
import torch.nn.functional as F
print("Running with n_layers, n_nodes", n_layers, n_nodes, flush=True)

#-----------------------------------------------------------------------------
# make model
print(data.shape)
class BasicNet(torch.nn.Module):
    def __init__(self, n_layers=5, n_nodes=70):
        super(BasicNet, self).__init__()
        self.fc1 = torch.nn.Linear(data.shape[1]-1, n_nodes)
        self.fc_middle = torch.nn.ModuleList(
            [torch.nn.Linear(n_nodes, n_nodes) for _ in range(n_layers)]
        )
        self.fc_before_out = torch.nn.Linear(n_nodes, 2)
        self.n_layers = n_layers
    def get_distance_prior(self, l_mod, b_mod, true_distance):
        x = true_distance * torch.sin(b_mod) * torch.cos(l_mod)
        y = true_distance * torch.sin(b_mod) * torch.sin(l_mod)
        z = true_distance * torch.cos(b_mod)
        distance_to_galaxy_center = 7.8 # Along x

        x_mod = x - distance_to_galaxy_center
        r = torch.sqrt(x_mod**2 + y**2 + z**2)#Should be distance from galactic center
        #distance_log_prior = 2*torch.log(r) - r/self.length_scale(l, b) - (true_parallax - global_parallax_zeropoints - 1.0/r)**2/sigma_square**2
        L = 200#10* 0.200 # Should be 200 pc!
        #distance_log_prior = 2*torch.log(r) - r/L
        distance_log_prior = 0.0
        #Equation 18 https://arxiv.org/pdf/1804.09376.pdf
        return distance_log_prior
    def forward(self, x):
        # Max pooling over a (2, 2) window
        activation = F.leaky_relu
        input_vec = x
        x = activation(self.fc1(x[:, :-1]))
        x = activation(self.fc_middle[0](x))

        for i in range(1, self.n_layers):
            x = activation(self.fc_middle[i](x))

        x = self.fc_before_out(x)

        true_parallax = x[:, 1]
        sigma_square = x[:, 0]**2
        constant_offset_to_make_l1_work = 10000.0

        #true_distance = 1.0/(F.relu(true_parallax) + 1e-10)
        #TODO: Make sure these are correct!
        #to_rad = lambda deg: np.pi*deg/180.0
        #l_mod = to_rad(input_vec[:, 0])#phi - angle from center of galaxy
        #b_mod = to_rad(90 - input_vec[:, 1]) #theta

        #distance_log_prior = self.get_distance_prior(l_mod, b_mod, true_distance)

        x = -(x[:, 1] - input_vec[:, -1])**2/sigma_square - 0.5*torch.log(2*np.pi*sigma_square + 1e-7) - constant_offset_to_make_l1_work 
        #x += distance_log_prior

        # Add distance prior here?
        return x, sigma_square, true_parallax

class Net(torch.nn.Module):
    def __init__(self, n_layers=5, n_nodes=70):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(data.shape[1]-1, n_nodes)
        self.fc_middle = torch.nn.ModuleList(
            [torch.nn.Linear(n_nodes, n_nodes) for _ in range(n_layers)]
        )
        self.fc_before_middles = torch.nn.ModuleList(
            [torch.nn.Linear(n_nodes, n_nodes) for _ in range(n_layers - 1)]
        )
        self.fc_before_out = torch.nn.Linear(n_nodes, 2)
        self.fc_before_before_out = torch.nn.Linear(n_nodes, 2)
        self.n_layers = n_layers

    def get_distance_prior(self, l_mod, b_mod, true_distance):
        x = true_distance * torch.sin(b_mod) * torch.cos(l_mod)
        y = true_distance * torch.sin(b_mod) * torch.sin(l_mod)
        z = true_distance * torch.cos(b_mod)
        distance_to_galaxy_center = 7.8 # Along x

        x_mod = x - distance_to_galaxy_center
        r = torch.sqrt(x_mod**2 + y**2 + z**2)#Should be distance from galactic center
        #distance_log_prior = 2*torch.log(r) - r/self.length_scale(l, b) - (true_parallax - global_parallax_zeropoints - 1.0/r)**2/sigma_square**2
        L = 0.200 # Should be 200 pc!
        distance_log_prior = 2*torch.log(r) - r/L
        #Equation 18 https://arxiv.org/pdf/1804.09376.pdf
        return distance_log_prior

    def forward(self, x):
        # Max pooling over a (2, 2) window
        activation = F.leaky_relu
        input_vec = x
        x = activation(self.fc1(x[:, :-1]))
        network_input = x
        x = activation(self.fc_middle[0](x))

        for i in range(1, self.n_layers):
            x = activation(self.fc_middle[i](x) + self.fc_before_middles[i - 1](network_input))

        go_into_last = self.fc_before_before_out(network_input)
        x = self.fc_before_out(x)
        x = x + go_into_last

        true_parallax = x[:, 1]
        sigma_square = x[:, 0]**2
        constant_offset_to_make_l1_work = 10000.0

        #global_parallax_zeropoints = -0.029 #https://arxiv.org/pdf/1804.10121.pdf
        true_distance = 1.0/(F.relu(true_parallax) + 1e-10)
        #TODO: Make sure these are correct!
        to_rad = lambda deg: np.pi*deg/180.0
        l_mod = to_rad(input_vec[:, 0])#phi - angle from center of galaxy
        b_mod = to_rad(90 - input_vec[:, 1]) #theta

        distance_log_prior = self.get_distance_prior(l_mod, b_mod, true_distance)

        x = -(x[:, 1] - input_vec[:, -1])**2/sigma_square - 0.5*torch.log(2*np.pi*sigma_square + 1e-7) - constant_offset_to_make_l1_work 
        x += distance_log_prior
        #Needs 1s as L1loss

        # Add distance prior here?
        return x, sigma_square, true_parallax


tmodel2 = BasicNet(n_layers=n_layers, n_nodes=n_nodes)
model2 = []

if data_parallel:
    model2 = torch.nn.DataParallel(tmodel2, device_ids=list(range(torch.cuda.device_count())))
else:
    model2 = tmodel2
if data_parallel:
    with torch.cuda.device(0):
        model2.cuda()
else:
    model2.cuda()
#model2.cuda()
#loss_fn = torch.nn.MSELoss(size_average=True)
#HACK requires -1, 1 in y!
loss_fn = torch.nn.L1Loss(size_average=True)
# define optimizer
learning_rate = 0.001#0.001
optimizer = torch.optim.Adam(model2.parameters(), lr=learning_rate)
#-----------------------------------------------------------------------------
# convergence counter
current_loss = []
if data_parallel:
    with torch.cuda.device(0):
        current_loss = torch.tensor(10.**8)
else:
    current_loss = torch.tensor(10.**8)
count = 0
t = 0
print(current_loss, flush=True)
# record time
start_time = time.time()
losses_valid = []
#-----------------------------------------------------------------------------
# train the neural network
while t < 2000:
    # training
    y_pred, sigma_square, true_parallax = model2(x)
    loss_error = loss_fn(y_pred, y)
    # The model is only sent once, thus the division by
    # the number of datapoints used to train
    #loss_prior = model2.prior_loss() / (x.shape[0])
    loss = loss_error# + loss_prior
    # optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # validation
    y_pred_valid, sigma_square, true_parallax = model2(x_valid)
    loss_valid = loss_fn(y_pred_valid, y_valid)# + loss_prior
#-----------------------------------------------------------------------------
    # check convergence
    if t % 100 == 0:
        #More data!
        x, y, x_valid, y_valid = generate_more_data()
    if t % 200 == 1:
        #More data!
        #x, y, x_valid, y_valid = generate_more_data()
        if loss_valid >= current_loss:
            pass
        else:
            current_loss = loss_valid
            save_path = 'save_%d_epoch_%d_layers_%d_nodes_ultimate_resnet.pt' % (t, n_nodes, n_layers)
            torch.save(model2.state_dict(), save_path)
        print('Likelihood loss: %d nodes, %d layers, ' % (n_nodes, n_layers), flush=True, end='')
        print("Epoch", t, "loss", loss_valid.cpu().detach().numpy(), "best loss", current_loss.cpu().detach().numpy(), flush=True)

#-----------------------------------------------------------------------------
    # add counter
    t += 1
data_chunks = []
for i in range(int(0.5+data.shape[0]/800000)):
    cur_data = data[i*800000:np.min([(i+1)*800000, data.shape[0]])].copy()

    x = Variable(torch.from_numpy(cur_data)).type(dtype)
    # run cuda
    kwargs = {}
    #HACK - need this for the following to get multi-device!
    if data_parallel:
        with torch.cuda.device(0): #Bottom 4 snippets!
            x = x.cuda(**kwargs)
    else:
        x = x.cuda(**kwargs)


    likelihood, sigma_square, true_parallax = model2(x)
    data_chunks.append([x.cpu().detach().numpy(), likelihood.cpu().detach().numpy(), sigma_square.cpu().detach().numpy(), true_parallax.cpu().detach().numpy()])
#pkl.dump(data_chunks, open('quickdumpdskfjsdlkjfs.pkl', 'wb'))
pkl.dump(data_chunks, open('quickdump_new_slidjflsdjlf.pkl', 'wb'))
master_losses = current_loss.cpu().detach().numpy()
print('Average 5-fold loss (50000 epoch): %d nodes, %d layers, ' % (n_nodes, n_layers), np.average(master_losses), 'loss', flush=True)

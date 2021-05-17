import matplotlib.pyplot as plt
import mpl_scatter_density
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_scatter_density import \
    ScatterDensityArtist  # adds projection='scatter_density'


def get_plot_func(env_id):
    if 'Point' in env_id:
        return plot_obs_point
    elif 'Ant' in env_id:
        return plot_obs_ant
    elif 'DD2B' or 'C2B' in env_id:
        return plot_obs_DD2B
    elif 'HC' in env_id or 'Walker' in env_id:
        return plot_obs_hc
    elif 'LapGrid' in env_id:
        return plot_obs_DD2B
    else:
        return lambda *x: None

# ================================================================== #
# Scatter plot with color representing density                       #
# ================================================================== #

# "Viridis-like" colormap with white background
white_viridis = LinearSegmentedColormap.from_list('white_viridis',
    [(0, '#ffffff'),
     (1e-20, '#440053'),
     (0.2, '#404388'),
     (0.4, '#2a788e'),
     (0.6, '#21a784'),
     (0.8, '#78d151'),
     (1, '#fde624')], N=256)

def using_mpl_scatter_density(fig, x, y):
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    density = ax.scatter_density(x, y, cmap=white_viridis)
    #a = ScatterDensityArtist(ax, x, y)
    #ax.add_artist(a)
    fig.colorbar(density, label='Number of points per pixel')
    return ax
# ================================================================== #

def plot_circle(fig, ax, radius):
    ax.add_artist(plt.Circle((0,0), radius, fill=False, color='k'))

def plot_vertical_lines(fig, ax, x_coords):
    if len(x_coords) == 1:
        x_coords = x_coords[0]
    plt.vlines(x_coords, color='red',ymin=-10, ymax=10, linestyle='dashed')

def plot_obs_ant(obs, save_name):
    # Clip observations
    obs = np.clip(obs, -10, 10)

    if len(obs.shape) > 2:
        obs.reshape(-1, obs.shape[-1])

    if obs.shape[0] > 100000: # Use density plotting only for large number of points
        fig = plt.figure(figsize=(15,15))
        ax = using_mpl_scatter_density(fig, obs[:,0], obs[:,1])
    else:
        fig, ax = plt.subplots(1,1, figsize=(15,15))
        ax.scatter(obs[:,0], obs[:,1])
    plot_circle(fig, ax, radius=10)
    plot_vertical_lines(fig, ax, [-3, 3])
    ax.set_xlim([-10,10])
    ax.set_ylim([-10,10])
    fig.savefig(save_name)
    plt.close(fig=fig)

def plot_obs_point(obs, save_name):
    # Clip observations
    obs = np.clip(obs, -10, 10)

    # check size of obs, if too many, restrict to 5000
    # Also select first two dimensions (corresponding to 
    # x and y positions)
    obs = obs[:5000, [0,1]]
    fig, ax = plt.subplots(1,3,figsize=(45,15))
    ax[0].scatter(obs[:,0], obs[:,1])
    plot_circle(fig, ax[0], radius=10)
    plot_vertical_lines(fig, ax[0], [-3, 3])
    ax[0].set_xlim([-10,10])
    ax[0].set_ylim([-10,10])
    ax[1].hist(obs[:,0], bins=20, range=(-10, 10))
    ax[1].plot(np.ones(1000)*-3, np.arange(1000), 'r')
    ax[1].plot(np.ones(1000)*3, np.arange(1000), 'r')
    ax[2].hist(obs[:,1], bins=20, range=(-10, 10))
    ax[2].plot(np.ones(1000)*-3, np.arange(1000), 'r')
    ax[2].plot(np.ones(1000)*3, np.arange(1000), 'r')
    fig.savefig(save_name)
    plt.close(fig=fig)

def plot_obs_hc(obs, save_name):
    obs = np.clip(obs, -20, 20)
    if len(obs.shape) > 2:
        obs.reshape(-1, obs.shape[-1])
    fig, ax = plt.subplots(1,1, figsize=(15,15))
    ax.scatter(obs[:,0], 0.2 + np.zeros(obs.shape[0]))
    #ax.scatter(obs[:,0], obs[:,1])
    plot_vertical_lines(fig, ax, [-3, 3])
    ax.set_xlim([-20,20])
    ax.set_ylim([-2,2])
    fig.savefig(save_name)
    plt.close(fig=fig)

def plot_obs_DD2B(obs, save_name):
#    obs = obs[:5000, [0,1]]
    fig, ax = plt.subplots(1,1,figsize=(15,15))
    ax.scatter(obs[:,0], obs[:,1])
    ax.set_xlim(0,20)
    ax.set_ylim(0,20)
    fig.savefig(save_name)
    plt.close(fig=fig)

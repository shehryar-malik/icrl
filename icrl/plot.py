"""Generate nice plots for paper."""

import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

import wandb
api = wandb.Api()
entity = 'spiderbot'

from itertools import cycle

MARKERSIZE = 10
LINEWIDTH = 4

# ============================================================================
# Utils
# ============================================================================

def smooth_data(scalars, weight=0.):
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return np.array(smoothed)

def tsplot(data, x=None, smooth=0., marker=None, label=None, **kw):
    if x is None:
        x = np.arange(data.shape[0])
    # Plot data's smoothed mean
    y = np.mean(data, axis=1)
    y = smooth_data(y, weight=smooth)
    # Find standard deviation and error
    sd = np.std(data, axis=1)
    se = sd/np.sqrt(data.shape[1])
    # Plot
    plt.plot(x, y, marker=marker, markersize=MARKERSIZE, linewidth=LINEWIDTH, label=label, **kw)
    # Show error on graph
    cis = (y-se, y+se)
    plt.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)

def plot_legend(legends, colors, markers, save_name):
    # Dummy plots
    for legend, color, marker in zip(legends, colors, markers):
        plt.plot([0,0,0], [0,0,0], color=color, label=legend, marker=marker, markersize=MARKERSIZE, linewidth=LINEWIDTH)
    # Get legend separately
    handles, labels = plt.gca().get_legend_handles_labels()
    leg = plt.legend(handles, labels, loc='center', ncol=len(legends))
    plt.axis('off')
    fig = leg.figure
    fig.canvas.draw()
    bbox = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(save_name, bbox_inches=bbox, pad_inches=0, dpi=500)
    plt.close('all')

# ============================================================================
# Main plotting
# ============================================================================

def retrieve_group(project, group, metric, x_axis, prepend=None):
    # Get runs
    path = os.path.join(entity, project)
    runs = api.runs(path=path, filters={"config.group": group})
    # Get data
    data = [run.history()[metric] for run in runs]
    min_length = min([d.shape[0] for d in data])
    data = np.concatenate([datum.to_numpy()[:min_length,None] for datum in data], axis=-1)
    # Just get x-axis of one run since all runs should be identical
    x_axis = runs[0].history()[x_axis].to_numpy()[:min_length]

    # Filter out nans
    c_data, c_x_axis = [], []
    for datum, x in zip(data, x_axis):
        if np.sum(np.isnan(datum)) == 0:
            c_data += [datum]
            c_x_axis += [x]
    data, x_axis = np.array(c_data), np.array(c_x_axis)
    if prepend is not None:
        data, x_axis = prepend_missing_points(data, x_axis, prepend)
    return data, x_axis

def prepend_missing_points(data, x_axis, points):
    x_axis = np.concatenate([[0], x_axis])
    points = points[:data.shape[1]]
    points = np.reshape(points, [1,data.shape[1]])
    data = np.concatenate([points, data], axis=0)
    return data, x_axis

def plot(data, x_axis=None, min_x_axis=None, smooth=0., legend=None, color=None, marker=None):
    if x_axis is not None and min_x_axis is not None:
        # Take evenly spaced points to match mix_x_axis
        r = int(x_axis.shape[0]/min_x_axis.shape[0])
        indices = np.arange(0, x_axis.shape[0], r)
        #indices = list(filter(lambda idx: x_axis[idx] >= min_x_axis[0], indices))
        x_axis = x_axis[indices]
        data = data[indices]

    tsplot(data, x=x_axis, smooth=smooth, marker=marker, label=legend, color=color)

def plot_graph(project, groups, metrics, x_axes, save_name, xlim=None, ylim=None, legends=None, smooth=0.,
               colors=None, markers=None, horizontal_lines=None, horizontal_lines_colors=None, horizontal_lines_legends=None,
               horizontal_lines_markers=None, ylabel_length=None, prepend=None, x_label=None, y_label=None, correct_x_axis=False,
               show_legend=False):
    # Retrieve data
    metrics = [metrics]*len(groups) if type(metrics) != list else metrics
    x_axes = [x_axes]*len(groups) if type(x_axes) != list else x_axes
    data = [retrieve_group(project, *args, prepend=prepend) for args in zip(groups, metrics, x_axes)]

    # Take value at equally spaced intervals
    min_x_axis = min([x_axis for _, x_axis in data], key=lambda x: x.shape[0])

    # Plot any horizontal lines
    if horizontal_lines is not None:
        hcolors = [horizontal_lines_colors]*len(groups) if type(horizontal_lines_colors) != list else horizontal_lines_colors
        hmarkers = [horizontal_lines_markers]*len(groups) if type(horizontal_lines_markers) != list else horizontal_lines_markers
        for line, color, legend, marker in zip(horizontal_lines, hcolors, horizontal_lines_legends, hmarkers):
            plt.plot(min_x_axis, line*np.ones(min_x_axis.shape), linewidth=LINEWIDTH, marker=marker, markersize=MARKERSIZE, color=color, label=legend)

    # Plot data
    legends = [legends]*len(groups) if type(legends) != list else legends
    colors = [colors]*len(groups) if type(colors) != list else colors
    markers = [markers]*len(groups) if type(markers) != list else markers
    for (datum, x_axis), legend, color, marker in zip(data, legends, colors, markers):
        plot(datum, x_axis, min_x_axis, smooth, legend, color, marker)

    # Format plot
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    if plt.yticks()[0][-1] >= 2000:
        ylabels = ['%d' % y + 'k' for y in plt.yticks()[0]/1000]
        plt.gca().set_yticklabels(ylabels)

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

    # Correct ylabels length (will cast to int)
    if ylabel_length is not None:
        ax = plt.gca()
        ylabels = [' '*(ylabel_length-len(str(int(y))))+'%d'%y for y in plt.yticks()[0]]
        ax.set_yticklabels(ylabels)
    plt.margins(x=0)
    plt.gca().grid(which='major', linestyle='-', linewidth='0.2', color='#d3d3d3')
    plt.grid('on')

    # Label axes
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)

    if show_legend:
        plt.legend(loc='upper left', prop={'size': 12})

    # Save
    #plt.show()
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0, dpi=500)

    plt.close()

# ============================================================================
# What to plot?
# ============================================================================

def main_results(save_dir):
    project = 'ICRL-FE2'
    smooth = 0.5
    #colors = ['r', '#006400', '#adadad', '#00008b', '#ff8c00']
    #colors = ['r', '#006400', 'y', '#fed8b1', '#add8e6']
    colors = ['r', '#006400', 'y', '#9932a8', '#1f5fc4']
    #markers = ['s', '^', 'p', '*', 'X']
    markers = [None, None, None, None, None]

    # ========================================================================
    # Legends
    # ========================================================================

    plot_legend(['ICRL', 'GC', 'BC', 'nominal', 'expert'], colors, markers, os.path.join(save_dir, 'legend.png'))

    # ========================================================================
    # Learning constraints
    # ========================================================================

    def lgw():
        sd = os.path.join(save_dir, 'lgw')
        os.makedirs(sd, exist_ok=True)

        # Data
        lgw_reward_at_zero = np.array([
            -1.,  2., -1., -1., -1., -1., -1., -1., -1., -1.,
        ])
        lgw_cost_at_zero = np.array([
            0.575, 0.515, 0.48, 0.535, 0.5, 0.495, 0.525, 0.55, 0.525, 0.515
        ])
        lgw_nominal_reward = -1
        lgw_expert_reward = 60
        lgw_nominal_cost = 1.
        lgw_expert_cost = 0.

        # LGW Reward
        plot_graph(
                project=project,
                groups=['LapGrid-ICRL', 'LapGrid-GLC', 'LapGrid-Glag'],
                metrics=['true/reward', 'eval/mean_reward', 'true/reward'],
                x_axes=['timesteps', 'time/total_timesteps', 'timesteps'],
                save_name=os.path.join(sd, 'reward.png'),
                xlim=[0,120000],
                ylim=[-3,62],
                legends=[None,None,None],
                smooth=smooth,
                colors=colors[:3],
                markers=markers[:3],
                horizontal_lines=[lgw_nominal_reward, lgw_expert_reward],
                horizontal_lines_colors=colors[3:],
                horizontal_lines_markers=markers[3:],
                horizontal_lines_legends=[None,None],
                prepend=lgw_reward_at_zero,
                x_label=None,
                y_label=None,
                correct_x_axis=True
        )

        # LGW Cost
        plot_graph(
                project=project,
                groups=['LapGrid-ICRL', 'LapGrid-GLC', 'LapGrid-Glag'],
                metrics=['true/cost', 'eval/mean_cost', 'true/cost'],
                x_axes=['timesteps', 'time/total_timesteps', 'timesteps'],
                save_name=os.path.join(sd, 'violations.png'),
                xlim=[0,120000],
                ylim=[-0.05,1.05],
                legends=[None,None,None],
                smooth=smooth,
                colors=colors[:3],
                markers=markers[:3],
                horizontal_lines=[lgw_nominal_cost, lgw_expert_cost],
                horizontal_lines_colors=colors[3:],
                horizontal_lines_markers=markers[3:],
                horizontal_lines_legends=[None,None],
                prepend=lgw_cost_at_zero,
                x_label=None,
                y_label=None,
                correct_x_axis=True
        )


    def hc_lc():
        sd = os.path.join(save_dir, 'hc')
        os.makedirs(sd, exist_ok=True)

        # Data
        hc_reward_at_zero = np.array([
            196.97503954, 216.46464462, 279.05939259, 227.10430118, 295.01714065,
            54.937775480, 065.93740182, 230.63063724, 050.31560911, 049.84911195,
        ])
        hc_cost_at_zero = np.array([
            0.083, 0.0, 0.683, 0.857, 0.0, 0.0, 0.355, 0.0, 0.136, 0.713
        ])
        hc_nominal_reward = 56
        hc_expert_reward = 3800
        hc_nominal_cost = 1.
        hc_expert_cost = 0.

        # HC Reward
        plot_graph(
                project=project,
                groups=['HC-ICRL', 'HC-GLC', 'HC-Glag'],
                metrics=['true/reward', 'eval/mean_reward', 'true/reward'],
                x_axes=['timesteps', 'time/total_timesteps', 'timesteps'],
                save_name=os.path.join(sd, 'reward.png'),
                xlim=[0,3.85e6],
                ylim=[-200,4000],
                legends=[None,None,None],
                smooth=smooth,
                colors=colors[:3],
                markers=markers[:3],
                horizontal_lines=[hc_nominal_reward, hc_expert_reward],
                horizontal_lines_colors=colors[3:],
                horizontal_lines_markers=markers[3:],
                horizontal_lines_legends=[None,None],
                prepend=hc_reward_at_zero,
                x_label=None,
                y_label=None,
        )

        # HC Cost
        plot_graph(
                project=project,
                groups=['HC-ICRL', 'HC-GLC', 'HC-Glag'],
                metrics=['true/cost', 'eval/mean_cost', 'true/cost'],
                x_axes=['timesteps', 'time/total_timesteps', 'timesteps'],
                save_name=os.path.join(sd, 'violations.png'),
                xlim=[0,3.85e6],
                ylim=[-0.05,1.05],
                legends=[None,None,None],
                smooth=smooth,
                colors=colors[:3],
                markers=markers[:3],
                horizontal_lines=[hc_nominal_cost, hc_expert_cost],
                horizontal_lines_colors=colors[3:],
                horizontal_lines_markers=markers[3:],
                horizontal_lines_legends=[None,None],
                prepend=hc_cost_at_zero,
                x_label=None,
                y_label=None,
        )

    def ant_lc():
        sd = os.path.join(save_dir, 'ant')
        os.makedirs(sd, exist_ok=True)

        # Data
        ant_reward_at_zero = np.array([
            1342.11199848, 1207.24967682, -51.52055938, 1521.19819447, 452.17171373,
            42.61464075, 575.98478714, 651.30991591, 344.7494582, 703.13014031
        ])
        ant_cost_at_zero = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])
        ant_nominal_reward = 3000
        ant_expert_reward = 9000
        ant_nominal_cost = 0.3
        ant_expert_cost = 0.

        # Ant Reward
        plot_graph(
                project=project,
                groups=['AntWall-ICRL', 'AntWall-GLC', 'AntWall-GLag'],
                metrics=['true/reward', 'eval/mean_reward', 'true/reward'],
                x_axes=['timesteps', 'time/total_timesteps', 'timesteps'],
                save_name=os.path.join(sd, 'reward.png'),
                xlim=[0,3.85e6],
                ylim=[-200,9200],
                legends=[None,None,None],
                smooth=smooth,
                colors=colors[:3],
                markers=markers[:3],
                horizontal_lines=[ant_nominal_reward, ant_expert_reward],
                horizontal_lines_colors=colors[3:],
                horizontal_lines_markers=markers[:3],
                horizontal_lines_legends=[None,None],
                prepend=ant_reward_at_zero,
                x_label=None,
                y_label=None,
        )

        # Ant Cost
        plot_graph(
                project=project,
                groups=['AntWall-ICRL', 'AntWall-GLC', 'AntWall-GLag'],
                metrics=['true/cost', 'eval/mean_cost', 'true/cost'],
                x_axes=['timesteps', 'time/total_timesteps', 'timesteps'],
                save_name=os.path.join(sd, 'violations.png'),
                xlim=[0,3.85e6],
                ylim=[-0.02,0.32],
                legends=[None,None,None],
                smooth=smooth,
                colors=colors[:3],
                markers=markers[:3],
                horizontal_lines=[ant_nominal_cost, ant_expert_cost],
                horizontal_lines_colors=colors[3:],
                horizontal_lines_markers=markers[:3],
                horizontal_lines_legends=[None,None],
                prepend=ant_cost_at_zero,
                x_label=None,
                y_label=None,
        )

    # ========================================================================
    # Transferring constraints
    # ========================================================================

    def ant_to_point():
        sd = os.path.join(save_dir, 'point')
        os.makedirs(sd, exist_ok=True)

        # Data
        point_reward_at_zero = np.array([
            0.21352264, -0.06529675, -0.0413248, -0.12043785, 0.10773159, -0.22479139,
            0.34000134, -0.04010579, -0.10166518, -2.06870481
        ])
        point_cost_at_zero = np.array([
            0.0, 0.153, 0.0, 0.247, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])
        point_nominal_reward = 50
        point_expert_reward = 150
        point_nominal_cost = 0.6
        point_expert_cost = 0.

        # Point reward
        plot_graph(
                project=project,
                groups=['Point-CT-ICRL', 'Point-CT-GLC', 'Point-CT-Glag'],
                metrics=['eval/mean_reward', 'eval/mean_reward', 'eval/mean_reward'],
                x_axes=['time/total_timesteps', 'time/total_timesteps', 'time/total_timesteps'],
                save_name=os.path.join(sd, 'reward.png'),
                xlim=[0,1.5e6],
                ylim=[-10,160],
                legends=[None,None,None],
                smooth=smooth,
                colors=colors[:3],
                markers=markers[:3],
                horizontal_lines=[point_nominal_reward, point_expert_reward],
                horizontal_lines_colors=colors[3:],
                horizontal_lines_markers=markers[3:],
                horizontal_lines_legends=[None,None],
                prepend=point_reward_at_zero,
                x_label='timesteps',
                y_label='reward',
        )

        # Point cost
        plot_graph(
                project=project,
                groups=['Point-CT-ICRL', 'Point-CT-GLC', 'Point-CT-Glag'],
                metrics=['eval/true_cost', 'eval/mean_cost', 'eval/true_cost'],
                x_axes=['time/total_timesteps', 'time/total_timesteps', 'time/total_timesteps'],
                save_name=os.path.join(sd, 'violations.png'),
                xlim=[0,1.5e6],
                ylim=[-0.05,0.65],
                legends=[None,None,None],
                smooth=smooth,
                colors=colors[:3],
                markers=markers[:3],
                horizontal_lines=[point_nominal_cost, point_expert_cost],
                horizontal_lines_colors=colors[3:],
                horizontal_lines_markers=markers[3:],
                horizontal_lines_legends=[None,None],
                prepend=point_cost_at_zero,
                x_label='timesteps',
                y_label='constraint violations'
        )

    def ant_to_ant_broken():
        sd = os.path.join(save_dir, 'ant_broken')
        os.makedirs(sd, exist_ok=True)

        # Data
        ant_broken_reward_at_zero = np.array([
            1546.02287911, 1375.13323366, 1249.51326171, 1126.52033152, 708.51953463,
            802.53545791, 1182.28379381, 1550.51381021,  838.4939416,  1139.96627391,
        ])
        ant_broken_cost_at_zero = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])
        ant_broken_nominal_reward = 1100
        ant_broken_expert_reward = 3300
        ant_broken_nominal_cost = 0.3
        ant_broken_expert_cost = 0.

        # Ant broken reward
        plot_graph(
                project=project,
                groups=['AntBroken-CT-ICRL3', 'AntBroken-CT-GLC', 'AntBroken-CT-Glag'],
                metrics=['eval/mean_reward', 'eval/mean_reward', 'eval/mean_reward'],
                x_axes=['time/total_timesteps', 'time/total_timesteps', 'time/total_timesteps'],
                save_name=os.path.join(sd, 'reward.png'),
                xlim=[0,3e6],
                ylim=[-200,3500],
                legends=[None,None,None],
                smooth=smooth,
                colors=colors[:3],
                markers=markers[:3],
                horizontal_lines=[ant_broken_nominal_reward, ant_broken_expert_reward],
                horizontal_lines_colors=colors[3:],
                horizontal_lines_markers=markers[3:],
                horizontal_lines_legends=[None,None],
                prepend=ant_broken_reward_at_zero,
                x_label='timesteps',
                y_label='reward'
        )

        # Ant broken cost
        plot_graph(
                project=project,
                groups=['AntBroken-CT-ICRL3', 'AntBroken-CT-GLC', 'AntBroken-CT-Glag'],
                metrics=['eval/true_cost', 'eval/mean_cost', 'eval/true_cost'],
                x_axes=['time/total_timesteps', 'time/total_timesteps', 'time/total_timesteps'],
                save_name=os.path.join(sd, 'violations.png'),
                xlim=[0,1.5e6],
                ylim=[-0.02,0.32],
                legends=[None,None,None],
                smooth=smooth,
                colors=colors[:3],
                markers=markers[:3],
                horizontal_lines=[ant_broken_nominal_cost, ant_broken_expert_cost],
                horizontal_lines_colors=colors[3:],
                horizontal_lines_markers=markers[3:],
                horizontal_lines_legends=[None,None],
                prepend=ant_broken_cost_at_zero,
                x_label='timesteps',
                y_label='constraint violations'
        )

    # ========================================================================
    # Make all plots
    # ========================================================================

    lgw()
    hc_lc()
    ant_lc()
    ant_to_point()
    ant_to_ant_broken()

def ablation_studies(save_dir):
    project = 'Ablations'
    smooth = 0.
    colors = ['r', '#006400', 'y', '#9932a8', '#1f5fc4']
    markers = [None, None, None, None, None]
    show_legend = False

    # Data
    hc_reward_at_zero = np.array([
        196.97503954, 216.46464462, 279.05939259, 227.10430118, 295.01714065,
        54.937775480, 065.93740182, 230.63063724, 050.31560911, 049.84911195,
    ])
    hc_cost_at_zero = np.array([
        0.083, 0.0, 0.683, 0.857, 0.0, 0.0, 0.355, 0.0, 0.136, 0.713
    ])
    best_icrl_reward = 3800
    best_icrl_cost = 0.

    plot_legend([r'$B=1$', r'$B=5$', r'$B=10$', r'$B=20$', 'ICRL'], colors, markers, os.path.join(save_dir, 'legend.png'))

    def NoIS_NoES():
        sd = os.path.join(save_dir, 'no-is_no-es')
        os.makedirs(sd, exist_ok=True)

        groups = ['A-NoIS-NoES-BI1', 'A-NoIS-NoES-BI5', 'A-NoIS-NoES-BI10', 'A-NoIS-NoES-BI20']
        # Reward
        plot_graph(
                project=project,
                groups=groups,
                metrics='true/reward',
                x_axes='timesteps',
                save_name=os.path.join(sd, 'reward.png'),
                xlim=[0,3.85e6],
                ylim=[-200,4000],
                legends=['1', '5', '10', '20'],
                smooth=smooth,
                colors=colors[:4],
                markers=markers[:4],
                horizontal_lines=[best_icrl_reward],
                horizontal_lines_colors=colors[4:],
                horizontal_lines_markers=markers[4:],
                horizontal_lines_legends=['ICRL'],
                prepend=hc_reward_at_zero,
                x_label=None,
                y_label=None,
                show_legend=show_legend
        )

        # Cost
        plot_graph(
                project=project,
                groups=groups,
                metrics='true/cost',
                x_axes='timesteps',
                save_name=os.path.join(sd, 'violations.png'),
                xlim=[0,3.85e6],
                ylim=[-0.05,1.05],
                legends=['1', '5', '10', '20'],
                smooth=smooth,
                colors=colors[:4],
                markers=markers[:4],
                horizontal_lines=[best_icrl_cost],
                horizontal_lines_colors=colors[4:],
                horizontal_lines_markers=markers[4:],
                horizontal_lines_legends=['ICRL'],
                prepend=hc_cost_at_zero,
                x_label=None,
                y_label=None,
                show_legend=show_legend
        )

    def NoIS_ES():
        sd = os.path.join(save_dir, 'no-is_es')
        os.makedirs(sd, exist_ok=True)

        groups = ['A-NoIS-ES-BI1', 'A-NoIS-ES-BI5', 'A-NoIS-ES-BI10', 'A-NoIS-ES-BI20']
        # Reward
        plot_graph(
                project=project,
                groups=groups,
                metrics='true/reward',
                x_axes='timesteps',
                save_name=os.path.join(sd, 'reward.png'),
                xlim=[0,3.85e6],
                ylim=[-200,4000],
                legends=['1', '5', '10', '20'],
                smooth=smooth,
                colors=colors[:4],
                markers=markers[:4],
                horizontal_lines=[best_icrl_reward],
                horizontal_lines_colors=colors[4:],
                horizontal_lines_markers=markers[4:],
                horizontal_lines_legends=['ICRL'],
                prepend=hc_reward_at_zero,
                x_label=None,
                y_label=None,
                show_legend=show_legend
        )

        # Cost
        plot_graph(
                project=project,
                groups=groups,
                metrics='true/cost',
                x_axes='timesteps',
                save_name=os.path.join(sd, 'violations.png'),
                xlim=[0,3.85e6],
                ylim=[-0.05,1.05],
                legends=['1', '5', '10', '20'],
                smooth=smooth,
                colors=colors[:4],
                markers=markers[:4],
                horizontal_lines=[best_icrl_cost],
                horizontal_lines_colors=colors[4:],
                horizontal_lines_markers=markers[4:],
                horizontal_lines_legends=['ICRL'],
                prepend=hc_cost_at_zero,
                x_label=None,
                y_label=None,
                show_legend=show_legend
        )

    def IS_NoES():
        sd = os.path.join(save_dir, 'is_no-es')
        os.makedirs(sd, exist_ok=True)

        groups = ['A-IS-NoES-BI1', 'A-IS-NoES-BI5', 'A-IS-NoES-BI10', 'A-IS-NoES-BI20']
        # Reward
        plot_graph(
                project=project,
                groups=groups,
                metrics='true/reward',
                x_axes='timesteps',
                save_name=os.path.join(sd, 'reward.png'),
                xlim=[0,3.85e6],
                ylim=[-200,4000],
                legends=['1', '5', '10', '20'],
                smooth=smooth,
                colors=colors[:4],
                markers=markers[:4],
                horizontal_lines=[best_icrl_reward],
                horizontal_lines_colors=colors[4:],
                horizontal_lines_markers=markers[4:],
                horizontal_lines_legends=['ICRL'],
                prepend=hc_reward_at_zero,
                x_label=None,
                y_label=None,
                show_legend=show_legend
        )

        # Cost
        plot_graph(
                project=project,
                groups=groups,
                metrics='true/cost',
                x_axes='timesteps',
                save_name=os.path.join(sd, 'violations.png'),
                xlim=[0,3.85e6],
                ylim=[-0.05,1.05],
                legends=['1', '5', '10', '20'],
                smooth=smooth,
                colors=colors[:4],
                markers=markers[:4],
                horizontal_lines=[best_icrl_cost],
                horizontal_lines_colors=colors[4:],
                horizontal_lines_markers=markers[4:],
                horizontal_lines_legends=['ICRL'],
                prepend=hc_cost_at_zero,
                x_label=None,
                y_label=None,
                show_legend=show_legend
        )

    def IS_ES():
        sd = os.path.join(save_dir, 'is_es')
        os.makedirs(sd, exist_ok=True)

        groups = ['A-IS-ES-BI1', 'A-IS-ES-BI5', 'A-IS-ES-BI10', 'A-IS-ES-BI20']
        # Reward
        plot_graph(
                project=project,
                groups=groups,
                metrics='true/reward',
                x_axes='timesteps',
                save_name=os.path.join(sd, 'reward.png'),
                xlim=[0,3.85e6],
                ylim=[-200,4000],
                legends=['1', '5', '10', '20'],
                smooth=smooth,
                colors=colors[:4],
                markers=markers[:4],
                horizontal_lines=[best_icrl_reward],
                horizontal_lines_colors=colors[4:],
                horizontal_lines_markers=markers[4:],
                horizontal_lines_legends=['ICRL'],
                prepend=hc_reward_at_zero,
                x_label=None,
                y_label=None,
                show_legend=show_legend
        )

        # Cost
        plot_graph(
                project=project,
                groups=groups,
                metrics='true/cost',
                x_axes='timesteps',
                save_name=os.path.join(sd, 'violations.png'),
                xlim=[0,3.85e6],
                ylim=[-0.05,1.05],
                legends=['1', '5', '10', '20'],
                smooth=smooth,
                colors=colors[:4],
                markers=markers[:4],
                horizontal_lines=[best_icrl_cost],
                horizontal_lines_colors=colors[4:],
                horizontal_lines_markers=markers[4:],
                horizontal_lines_legends=['ICRL'],
                prepend=hc_cost_at_zero,
                x_label=None,
                y_label=None,
                show_legend=show_legend
        )

    def ER():
        sd = os.path.join(save_dir, 'er')
        os.makedirs(sd, exist_ok=True)

        groups = ['A-ER1', 'A-ER5', 'A-ER10', 'A-ER20']
        # Reward
        plot_graph(
                project=project,
                groups=groups,
                metrics='true/reward',
                x_axes='timesteps',
                save_name=os.path.join(sd, 'reward.png'),
                xlim=[0,3.85e6],
                ylim=[-200,4000],
                legends=['1', '5', '10', '20'],
                smooth=smooth,
                colors=colors[:4],
                markers=markers[:4],
                horizontal_lines=[best_icrl_reward],
                horizontal_lines_colors=colors[4:],
                horizontal_lines_markers=markers[4:],
                horizontal_lines_legends=['ICRL'],
                prepend=hc_reward_at_zero,
                x_label=None,
                y_label=None,
                show_legend=show_legend
        )

        # Cost
        plot_graph(
                project=project,
                groups=groups,
                metrics='true/cost',
                x_axes='timesteps',
                save_name=os.path.join(sd, 'violations.png'),
                xlim=[0,3.85e6],
                ylim=[-0.05,1.05],
                legends=['1', '5', '10', '20'],
                smooth=smooth,
                colors=colors[:4],
                markers=markers[:4],
                horizontal_lines=[best_icrl_cost],
                horizontal_lines_colors=colors[4:],
                horizontal_lines_markers=markers[4:],
                horizontal_lines_legends=['ICRL'],
                prepend=hc_cost_at_zero,
                x_label=None,
                y_label=None,
                show_legend=show_legend
        )


    NoIS_NoES()
    NoIS_ES()
    IS_NoES()
    IS_ES()
    ER()

if __name__=='__main__':
    start = time.time()
    main_results('icrl/plots/main_results')
    ablation_studies('icrl/plots/ablations')
    print('Time taken: ', (time.time()-start))

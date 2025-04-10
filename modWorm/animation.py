"""
modWorm: Modular simulation of neural connectomics, dynamics and biomechanics of Caenorhabditis elegans
Copyright (c) 2024-2025 University of Washington. Developed in UW NeuroAI Lab by Jimin Kim.
"""

__author__ = 'Jimin Kim: jk55@u.washington.edu'

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from imageio import imread
from modWorm import sys_paths as paths
from modWorm import muscle_body_params as mb_params
from IPython.display import Video

#####################################################################################################################################################
# MASTER FUNCTIONS ##################################################################################################################################
#####################################################################################################################################################

def animate_body(x, y, filename,
                 xmin, xmax, ymin, ymax,
                 figsize_x, figsize_y, background_img_path = False, animation_config = mb_params.CE_animation):

    if background_img_path != False:
        img = imread(background_img_path)

    segment_count = len(x[0, :])

    fig, ax = initialize_figure(figsize_x, figsize_y, xmin, xmax, ymin, ymax)

    patch_list = render_body(x, y, animation_config, segment_count)

    def init():

        for k in range(len(patch_list)):

            patch = patch_list[k]
            patch.center = x[0, k], y[0, k]
            ax.add_patch(patch)

        return patch_list

    def animate(i):

        for k in range(0, len(patch_list)):

            patch = patch_list[k]
            pos_x, pos_y = patch.center
            pos_x = x[i, k]
            pos_y = y[i, k]
            patch.center = (pos_x, pos_y)

        if animation_config.display_trail == True:

            patch_list.append(plt.Circle((x[i,animation_config.trail_point], y[i,animation_config.trail_point]), 
                                          animation_config.trail_width, color = animation_config.trail_color, alpha = 0.7))

            patch = patch_list[-1]
            patch.center = x[i, animation_config.trail_point], y[i, animation_config.trail_point]

            ax.add_patch(patch)

        return patch_list

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=animation_config.fps, metadata=dict(artist='Me'), bitrate=1800)

    anim = animation.FuncAnimation(fig, func = animate, init_func = init, 
                                        frames = len(x), interval = animation_config.interval, blit = True)

    ax.axis(animation_config.display_axis)

    os.chdir(paths.videos_dir)

    if background_img_path != False:

        ax.imshow(img, zorder=0, extent=[xmin, xmax, ymin, ymax])

    anim.save(filename + '.mp4',savefig_kwargs={'facecolor':animation_config.facecolor})

    os.chdir(paths.default_dir)

    return Video(paths.videos_dir + '/' + filename + '.mp4', embed = True, height = 500, width = 500)

#####################################################################################################################################################
# PREPARATION FUNCTIONS #############################################################################################################################
#####################################################################################################################################################

def initialize_figure(figsize_x, figsize_y, xmin, xmax, ymin, ymax):

    fig = plt.figure(figsize=(figsize_x, figsize_y))
    fig.set_dpi(100)

    ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax))

    return fig, ax

def render_body(x, y, animation_config, segment_count):

    patch_list = []

    diameters = animation_config.diameter_scaler * mb_params.CE_animation.h_interp(np.linspace(0, mb_params.CE.h_num, segment_count))
    radius = np.divide(diameters, 1.5)

    for k in range(0, segment_count):

        if type(animation_config.worm_seg_color) == str:

            patch_list.append(plt.Circle((x[0,k], y[0,k]), radius[k], color = animation_config.worm_seg_color))

        else:

            patch_list.append(plt.Circle((x[0,k], y[0,k]), radius[k], color = animation_config.worm_seg_color[k]))

    return patch_list
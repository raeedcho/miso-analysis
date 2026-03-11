import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Ellipse
from matplotlib import animation
import numpy as np
import pandas as pd
import seaborn as sns

from trialframe import get_index_level, slice_by_time

def animate_trial(trial, save_path=None):
    fig,[neural_ax, monitor_ax] = plt.subplots(1,2,figsize=(20,10))

    center_target=monitor_ax.add_patch(Circle((0,0),10,color='0.3'))
    target_direction = get_index_level(trial,'target direction').iloc[0]
    target_dist = 70
    outer_target = monitor_ax.add_patch(Circle(
        (target_dist * np.cos(target_direction*np.pi/180), target_dist * np.sin(target_direction*np.pi/180)),
        4.1,
        color='0.3',
        visible=False,
        fill=False,
        linestyle='--',
    ))

    target_ring = {
        angle: monitor_ax.add_patch(Circle(
            (target_dist * np.cos(angle*np.pi/180), target_dist * np.sin(angle*np.pi/180)),
            4,
            color='0.5',
            fill=True,
            visible=False,
        ))
        for angle in range(0,360,45)
    }
    
    stimmed_channels = get_index_level(trial,'stimulated channel').iloc[0]
    if stimmed_channels != frozenset():
        stim_indicator = {
            chan: monitor_ax.add_patch(Rectangle(
                (-40 + chan_num*10, 15),8,8,
                color='purple',
                visible=False,
            ))
            for chan_num, chan in enumerate(stimmed_channels)
        }
        # Add channel name annotations above the boxes
        for chan_num, chan in enumerate(stimmed_channels):
            monitor_ax.text(
                -40 + chan_num*10 + 4,  # Center of the box
                10,  # below the box
                chan,
                ha='center',
                va='bottom',
                fontsize=8,
                color='purple',
            )

        stim_activity = trial['stim activity'][stimmed_channels]
    else:
        stim_indicator = {}

    hand_position =  trial['hand position']
    cursor = monitor_ax.add_patch(Circle(hand_position[['x','y']].values[0],2,zorder=100,color='r'))

    monitor_ax.set_xlim((-90.,90.))
    monitor_ax.set_ylim((-90.,90.))
    monitor_ax.set_xticks([])
    monitor_ax.set_yticks([])
    sns.despine(ax=monitor_ax,left=True,bottom=True)

    neural_pca = trial['neural pca']
    # Use ellipse to account for non-equal axis limits
    # x_range = 4, y_range = 2, so width should be 2x height for circular appearance
    ellipse_width = 0.04
    ellipse_height = 0.02
    neural_state = neural_ax.add_patch(Ellipse(neural_pca[[0,1]].values[0], ellipse_width, ellipse_height, zorder=100, color=[0,104/255,55/255]))
    # Add trail line for showing past 200ms of neural state
    trail_line, = neural_ax.plot([], [], '-', color=[0,104/255,55/255], alpha=0.4, linewidth=2, zorder=99)

    neural_ax.set_xlim((-2.,2.))
    neural_ax.set_ylim((-1.,1.))
    neural_ax.set_xticks([])
    neural_ax.set_yticks([])
    sns.despine(ax=neural_ax,left=True,bottom=True)

    plt.tight_layout()

    def animate(frame_time):
        state = (
            get_index_level(trial,'state')
            .xs(level='time',key=frame_time)
            .values[0]
        )
        if state == 'Cheat Period':
            center_target.set(visible=False)
            for ring in target_ring.values():
                ring.set(visible=True)

        if state == 'Reach Target On':
            outer_target.set(visible=True, fill=True)
        else:
            outer_target.set(visible=True, fill=False)

        cursor.set(center=(
            hand_position
            .xs(level='time',key=frame_time)
            [['x','y']]
            .values[0]
        ))

        for chan,indicator in stim_indicator.items():
            if stim_activity.xs(level='time',key=frame_time).values[0] > 0:
                indicator.set(visible=True)
            else:
                indicator.set(visible=False)

        # Update trail to show past 200ms
        trail_start_time = frame_time - pd.Timedelta(milliseconds=200)
        trail_data = slice_by_time(neural_pca, slice(trail_start_time, frame_time))[[0,1]]
        if len(trail_data) > 0:
            trail_line.set_data(trail_data[0].values, trail_data[1].values)
        else:
            trail_line.set_data([], [])

        neural_state.set_center(
            neural_pca
            .xs(level='time',key=frame_time)
            [[0,1]]
            .values[0]
        )

        return [cursor,center_target,neural_state,trail_line] + list(stim_indicator.values()) + list(target_ring.values())

    frames = get_index_level(trial,'time')
    frame_interval =  frames.dt.total_seconds().diff().mode()[0]
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames = frames,
        interval = frame_interval * 1000,  # in milliseconds
        blit = True,
    )

    if save_path is not None:
        anim.save(save_path, writer='ffmpeg', fps=1/frame_interval, dpi=400)

    return anim
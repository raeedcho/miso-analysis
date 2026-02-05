import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib import animation
import numpy as np
import seaborn as sns

from src.time_slice import get_index_level

def animate_trial(trial, save_path=None):
    fig,monitor_ax = plt.subplots(1,1,figsize=(10,10))

    # resampled_trial = (
    #     trial
    #     .resample('10ms', level='time')
    #     .mean()
    # )
    resampled_trial = trial

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

        stim_activity = resampled_trial['stim activity'][stimmed_channels]
    else:
        stim_indicator = {}

    hand_position =  resampled_trial['hand position']
    cursor = monitor_ax.add_patch(Circle(hand_position[['x','y']].values[0],2,zorder=100,color='r'))

    monitor_ax.set_xlim((-90.,90.))
    monitor_ax.set_ylim((-90.,90.))
    monitor_ax.set_xticks([])
    monitor_ax.set_yticks([])
    sns.despine(ax=monitor_ax,left=True,bottom=True)

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

        return [cursor,center_target] + list(stim_indicator.values()) + list(target_ring.values())

    frames = get_index_level(resampled_trial,'time')
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
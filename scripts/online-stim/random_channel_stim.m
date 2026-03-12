%% Script to run random channel stim across arrays
% This script will stimulate across all channels sequentially
% There will be 10 seconds between each stimulation.

% Stimulation parameters
num_simul_stim_chans = 1;
all_chans = [1:96 129:160]; % available channels to loop through sequentially
active_chans = [2,3,4,5,9,11,12,33,40,52,53,57,61,73,75,77,79,80,88,146,158,160]';
inactive_chans = setdiff(all_chans,active_chans);
stim_chans = nchoosek(all_chans,num_simul_stim_chans);

fast_settle_option = 3; % 1=None, 2=Any, 3=Same port, 4= Same front end
fast_settle_duration = 0.5; %ms

pulse_width=250; % us
stim_freq=350; % Hz
stim_duration=150; % ms
stim_amplitude=25; %uA
time_between_stim=1.5; %s
num_stim_repeats=10;
catch_trials_per_block=2;

% initialize and check xippmex
addpath(genpath('C:\Program Files (x86)\Ripple\Trellis\Tools\xippmex'))
xippmex('close')
status = xippmex;
if status ~= 1
    error('unable to initialize xippmex')
end
available_stim_chans = xippmex('elec','stim');
    
unavailable_stim_chans = setdiff(stim_chans,available_stim_chans);
if any(unavailable_stim_chans)
    error('unable to stimulate on requested channels %d',unavailable_stim_chans)
end

% % record a baseline period before stim
% xippmex('trial',205,'recording',[],60,1) % record 60 seconds of baseline
% fprintf('recording baseline\n')
% pause(70) % wait for recording to finish
% % in a new file, record stim responses
% xippmex('trial',205,'recording',[],[],1)

for i = 1:num_stim_repeats
    stim_chan_order = randperm(length(stim_chans)+catch_trials_per_block);
    for channum = 1:length(stim_chans)+catch_trials_per_block
        if stim_chan_order(channum) > length(stim_chans)
            xippmex('digout', 1:2, [1, 1]); pause(0.001); xippmex('digout', 1:2, [0,0]);
            fprintf('catch trial - no stim\n')
            pause(time_between_stim)
            continue
        end
        chosen_chan = stim_chans(stim_chan_order(channum),:);
        
        xippmex('stim','enable',0) % disable stim first so step size can be set
        stim_cmd = xippmexStimCmd(chosen_chan,pulse_width,stim_freq,stim_duration,stim_amplitude);
        xippmex('stim','enable',1) % re-enable stim
        xippmex('signal',chosen_chan,'stim',chosen_chan)
        
        if fast_settle_option>0
            xippmex('fastsettle','stim',chosen_chan,fast_settle_option,fast_settle_duration);
        end
        
        % send a digital pulse to mark stim timing, then trigger stim
        xippmex('digout', 1:2, [1, 1]); pause(0.001); xippmex('digout', 1:2, [0,0]);
        xippmex('stimseq',stim_cmd)
        pause(time_between_stim)
    end
end
% xippmex('trial',205,'stopped')
xippmex('close')
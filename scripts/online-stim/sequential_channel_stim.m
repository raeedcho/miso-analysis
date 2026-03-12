%% Script to run sequential channel stim across arrays
% This script will stimulate across all channels sequentially
% There will be 10 seconds between each stimulation.

% Stimulation parameters
stim_chans = [1:96 129:160]; % available channels to loop through sequentially
fast_settle_option = 3; % 1=None, 2=Any, 3=Same port, 4= Same front end
fast_settle_duration = 0.5; %ms

pulse_width=250; % us
stim_freq=350; % Hz
stim_duration=150; % ms
stim_amplitude=25; %uA
time_between_stim=1; %s

% initialize and check xippmex
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

% start "trials"
for channum = 1:length(stim_chans)
    chosen_chan = stim_chans(channum);
    
    xippmex('stim','enable',0) % disable stim first so step size can be set
    stim_cmd = xippmexStimCmd(chosen_chan,pulse_width,stim_freq,stim_duration,stim_amplitude);
    xippmex('stim','enable',1) % re-enable stim
    xippmex('signal',chosen_chan,'stim',chosen_chan)
    
    if fast_settle_option>0
        xippmex('fastsettle','stim',chosen_chan,fast_settle_option,fast_settle_duration);
    end
    
    xippmex('stimseq',stim_cmd)
    
    pause(time_between_stim) % wait for 10 seconds
end

xippmex('close')
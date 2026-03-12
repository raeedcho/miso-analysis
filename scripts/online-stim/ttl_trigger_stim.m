%% Script to run TTL-triggered stim
% This script will wait for a TTL pulse on digital input,
% at which point it will choose a random channel from a given list
% and stimulate on the channel with given stimulation parameters.
% Settings for this script are at the beginning.

% Stimulation parameters
num_simul_stim_chans = 2;
% All channels
% avail_chans = [1:96 129:160]; % available channels to pick from randomly
% Sulley
% avail_chans = [1 2 3 5 6 9 12 33 34 35 40 43 44 50 52 53 57 59 60 61 63 64 67 69 70 72 73 77 80 81 82 83 84 87 88 90 91 95 133 154 156 158]';
% Prez
% avail_chans = [6 9 10 13 16 22 23 29 32 34 36 46 50 56 78 85 87 91 132 134 135 136 138 139 141 145 147 148 149 150 152 153 154 155 156 157 158 159]';
avail_chans = [36 46 50 83 85 6 18 22 25 132 145 149 150 155 157 159]';
stim_chans = nchoosek(avail_chans,num_simul_stim_chans);

fast_settle_option = 3; % 1=None, 2=Any, 3=Same port, 4= Same front end
fast_settle_duration = 0.5; %ms

pulse_width=250; % ms
stim_freq=350; % Hz
stim_duration=150; % ms
stim_amplitude=25; %uA

% initialize and check xippmex
addpath(genpath('C:\Program Files (x86)\Ripple\Trellis\Tools\xippmex'))
status = xippmex;
if status ~= 1
    error('unable to initialize xippmex')
end
available_stim_chans = xippmex('elec','stim');

unavailable_stim_chans = setdiff(stim_chans,available_stim_chans);
if any(unavailable_stim_chans)
    error('unable to stimulate on requested channels %d',unavailable_stim_chans)
end

[~, ~, ~] = xippmex('digin'); % clear digital buffer

while 1 % keep it alive
%     try % error catcher
        [~, ~, events] = xippmex('digin');
        if ~isempty(events) && any([events.reason] == 4) && any([events.sma2] > 0)
%             try
                %% use Ex stim command generation
                chosen_chan = stim_chans(randi(size(stim_chans,1)),:);
                
                xippmex('stim','enable',0) % disable stim first so step size can be set
                stim_cmd = xippmexStimCmd(chosen_chan,pulse_width,stim_freq,stim_duration,stim_amplitude);
                xippmex('stim','enable',1) % re-enable stim
                xippmex('signal',chosen_chan,'stim',chosen_chan)
                
                if fast_settle_option>0
                    xippmex('fastsettle','stim',chosen_chan,fast_settle_option,fast_settle_duration);
                end
                
                xippmex('stimseq',stim_cmd)
                
%             catch e
%                 disp(['Error: ' e.message]);
%             end
        end
        [~, ~, ~] = xippmex('digin'); % clear digital buffer
        pause(0.00001)
%     catch e
%         disp(['Main loop error: ' e.message]);
%         pause(0.5)
%     end
end
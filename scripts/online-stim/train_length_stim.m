%% Script to run random channel + variable length of stim train across arrays
% This script will stimulate active channels at varying lengths the train
% (i.e. varying number of pulses)
% (1, 2, 10, 35, 53 pulses)
%                       Trial Structure
%   [- - pre stim - -][- - stim - -][- - post stim - -]
%   [- -  300 ms  - -][- -(VARIABLE) ms- -][- -  800 ms   - -]
% E __________________|___|___|___|____________________
% ***Time intervals not to scale

% file parameters
addpath('/opt/Trellis/Tools/xippmex')
% file parameters
base_data_folder = '/home/collinlehmann';
monkey = 'Sulley';
date = datestr(now,'yyyy-mm-dd');
year = date(1:4);
data_path = fullfile(base_data_folder, monkey, year, date);
if ~exist(data_path, 'dir')
    mkdir(data_path)
end
filename_prefix = sprintf('%s_%s', monkey, date);
stim_paradigm = 'train-length-stim';

% Stimulation parameters
num_simul_stim_chans = 1;
all_chans = [1:96 129:160]; % available channels to loop through sequentially
active_chans = [1,3,4,5,6,7,8,9,11,12,15,31,40,53,57,73,77,79,80,88,129,137,147,160]';
inactive_chans = setdiff(all_chans,active_chans);
stim_chans = nchoosek(active_chans,num_simul_stim_chans);

fast_settle_option = 3; % 1=None, 2=Any, 3=Same port, 4= Same front end
fast_settle_duration = 0.5; %ms

pulse_width=250; % us
stim_freq=350; % Hz
cycle_num = [1 2 10 35 53]; %number of cycles (i.e. stim pulses) to deliver
stim_duration = cycle_num*1000/stim_freq; % ms
stim_amplitude=[25].'; %uA
stim_offset = [0] * 1e6 / stim_freq; % us
prestim_time=0.3; %s
poststim_time=0.8; %s
num_stim_repeats=10;
catch_trials_per_block=2;
baseline_recording_time = 30; % seconds to record baseline neural activity before stim

n_stim_elecs = length(stim_chans);
n_durations = length(stim_duration);

total_trials = num_stim_repeats*(catch_trials_per_block + n_stim_elecs*n_durations); % total number of planned stim trials
total_time = baseline_recording_time + total_trials*(prestim_time + poststim_time + (mean(stim_duration)/1000)); %total planned session time in s

stim_chans = repelem(stim_chans, n_durations,1);
stim_durations = repmat(stim_duration, n_stim_elecs, 1);

input(sprintf('Planned session includes %d trials, projected to last a total of %.1f minutes. Ok to proceed?',total_trials,total_time/60),'s');

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

% record a baseline period before stim
xippmex('trial','recording',fullfile(data_path, sprintf('%s_baseline_neural_', filename_prefix)),baseline_recording_time,1,1) % record baseline
fprintf('recording baseline\n')
pause(baseline_recording_time + 5) % wait for recording to finish
% in a new file, record stim responses (each trial gets its own file)
xippmex('trial','recording',fullfile(data_path, sprintf('%s_%s_neural_', filename_prefix, stim_paradigm)),0,1)
stim_record = fopen(fullfile(data_path,sprintf('%s_%s_trial_order.txt',filename_prefix,stim_paradigm)), 'w');
fprintf(stim_record,'%s\t%s\t%s\n','trial_id','channel','train_length');
for i = 1:num_stim_repeats
    stim_chan_order = randperm(length(stim_chans)+catch_trials_per_block);
    for channum = 1:length(stim_chans)+catch_trials_per_block
        % trial start
        fprintf('Running stimulation trial %d/%d.\n',channum + (length(stim_chans)+catch_trials_per_block)*(i-1),total_trials)
        pause(prestim_time)
        if stim_chan_order(channum) > length(stim_chans)
            xippmex('digout', 1:2, [1, 1]); pause(0.001); xippmex('digout', 1:2, [0,0]);
            fprintf('catch trial - no stim\n')
            fprintf(stim_record,'%d\t%s\t%s\n',channum + (i-1)*(length(stim_chans)+catch_trials_per_block),' ',' ');
            pause(poststim_time)
            continue
        end
        chosen_chan = stim_chans(stim_chan_order(channum),:);
        chosen_duration = stim_durations(stim_chan_order(channum));
        fprintf(stim_record,'%d\t%s\t%d\n',channum + (i-1)*(length(stim_chans)+catch_trials_per_block),num2str(chosen_chan),chosen_duration);
        
        xippmex('stim','enable',0) % disable stim first so step size can be set
        stim_cmd = xippmexStimCmd(chosen_chan,pulse_width,stim_freq,chosen_duration,stim_amplitude,stim_offset);
        xippmex('stim','enable',1) % re-enable stim
        xippmex('signal',chosen_chan,'stim',chosen_chan)

        if fast_settle_option>0
            xippmex('fastsettle','stim',chosen_chan,fast_settle_option,fast_settle_duration);
        end

        % send a digital pulse to mark stim timing, then trigger stim
        xippmex('digout', 1:2, [1, 1]); pause(0.001); xippmex('digout', 1:2, [0,0]);
        xippmex('stimseq',stim_cmd)
        pause(poststim_time)
    end
end
fclose(stim_record);
xippmex('trial','stopped')
xippmex('close')
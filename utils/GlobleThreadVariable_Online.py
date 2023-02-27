# Multi-threaded global variables for real environment
# created by zt
gl_need_sleep_stage_rl = False  # Whether sleep detection is required, and the target and policy will be updated later
gl_need_rl = False  # Need to start reinforcement learning
gl_need_white_noise = False  # Whether to start playing white noise
gl_need_realtime_sleep = False  # Whether it is necessary to monitor sleep in real time, time interval
gl_switch_is_open = False  # Whether the tgam sensor switch is on
gl_realtime_switch = False  # Whether to terminate sleep real-time monitoring
gl_stop_tgam_thread = True  # Whether to stop the thread switch for collecting data

gl_eeg_rawdata = []  # Save the data after the policy to update the target
gl_ai_eeg_raw = []  # Save data of normal sleep for 15 minutes
gl_real_time_sleep_stage_eeg = []  # Real-time sleeping data
# gl_eeg_start_time = 0  # The time domain action of the policy and the start time of white noise
gl_model_path = ""  # model path
gl_is_fall_in_sleep = 0  # real time sleep stage
user_name = "xxx"  # subject code

stop_wn_player_thread = False
stop_rl_thread = False
stop_real_time_is_sleep = False

gl_need_update_fd_target = False  # Update template in frequency domain
gl_need_update_td_target = False  # Update template in time domain

gl_start_timedomain_id = 0  # Time mark of this time domain action
gl_start_timedomain = 0  # Time of this time domain action
gl_frequency_action = []
gl_log_time = "0"
gl_mod_end_time = 0  # Last playback time
gl_switch_open_time = "0"  # Time to open the switch
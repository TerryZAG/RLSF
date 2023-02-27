# Multi-threaded global variables for simulation environment
# created by zt
gl_need_sleep_stage_rl = False  # Whether sleep detection is required, and the target and policy will be updated later
gl_need_rl = False  # Need to start reinforcement learning
gl_need_white_noise = False  # Whether to start playing white noise
gl_need_make_td_a0_data = False  # Whether target data (a0_target_eeg) for normal sleep is needed
gl_need_start_td_a0_data_thread = False  # Start a0_target_eeg after getting data
gl_need_realtime_sleep = False  # Whether it is necessary to monitor sleep in real time, time interval
# gl_switch_is_open = False  # Whether the tgam sensor switch is on
gl_realtime_switch = False  # Whether to terminate sleep real-time monitoring
gl_stop_tgam_thread = True  # Whether to stop the thread switch for collecting data
gl_make_td_a0_data_cont_time = 0
gl_policy = {}  # Save white noise playback strategy
gl_eeg_rawdata = []  # Save the data after the policy to update the target
gl_a0_eeg_raw = []  # Save data of normal sleep for 15 minutes
gl_real_time_sleep_stage_eeg = []  # Real-time detection of sleeping data
# gl_eeg_start_time = 0  # The time domain action of the policy and the start time of white noise
gl_model_path = ""  # Model path
gl_is_fall_in_sleep = 0  # sleep stage
user_name = "xxx"  # Patient code

stop_wn_player_thread = False
stop_rl_thread = False
stop_a0_target_update = False
stop_real_time_is_sleep = False
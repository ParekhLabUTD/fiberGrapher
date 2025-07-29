import numpy as np
import matplotlib.pyplot as plt
from OHRBETsEventExtractor import parse_ohrbets_serial_log, load_data 
from nxtEventExtrator import readfile, get_events

# Load saved photometry data
data = np.load('fiber_photometry_data5.npz')
signal_raw = data['signal']
control_raw = data['control']
time_raw = data['time']
sr = data['sr']
#pct_times = data['pct']

# Load and parse event data
path = r"D:\Aryan\python scripts\fiberGrapher\OHRBETsample.csv"
path_nxt = r'C:\Users\ParekhLab\Downloads\18_2dLightSucroseTest\18_2dLightSucroseTest\CohortTwoDignalTesting-250724-140624_18_2-250724-150026_SavedNotes1.nxt'
events_data = readfile(path_nxt)
code_map = {
    1:"Foraging",
    2:"Reward Approach",
    3:"Entering Foraging Arena",
    4:"Exiting Foraging Arena",
    5:"Home Approach",
    6:"Entering Home Arena",
    7:"Exiting Home Arena",
    8:"Grooming",
    9:"Eating"
}
events = get_events(events_data,code_map)

#events_data = load_data(path)
#events = parse_ohrbets_serial_log(events_data)

print(events)

# ----------------------------------------
# Downsample (average every N samples)
# ----------------------------------------
N = 100  # downsampling factor

F465 = [np.mean(signal_raw[i:i+N]) for i in range(0, len(signal_raw), N)]
F405 = [np.mean(control_raw[i:i+N]) for i in range(0, len(control_raw), N)]

signal = np.array(F465)
control = np.array(F405)

time = time_raw[::N]
time = time[:len(signal)]  # Ensure same length as downsampled signal

# ----------------------------------------
# ΔF/F using Lerner et al., 2015 method
# ----------------------------------------
bls = np.polyfit(control, signal, 1)  # linear fit of 405 to 465
fitted_baseline = bls[0] * control + bls[1]

Y_dF = signal - fitted_baseline
dFF = 100 * (Y_dF / fitted_baseline)

# Z-score normalization
dFF_mean = np.mean(dFF)
dFF_std = np.std(dFF)
dFF_z = (dFF - dFF_mean) / dFF_std

# ----------------------------------------
# Trim to start from 2nd PCT trial
# ----------------------------------------
#cutoff_time = pct_times[1]  # in seconds
cutoff_time = 2
cutoff_index = np.searchsorted(time, cutoff_time)

signal = signal[cutoff_index:]
control = control[cutoff_index:]
time = time[cutoff_index:] - time[cutoff_index]
dFF = dFF[cutoff_index:]
dFF_z = dFF_z[cutoff_index:]

# Adjust PCT times to match trimmed time
#pct_times = pct_times - cutoff_time
#pct_times = pct_times[pct_times >= 0]
trial_duration = 2.0  # seconds

# ----------------------------------------
# Plotting
# ----------------------------------------
plt.figure(figsize=(12, 5))

choice = int(input("Enter 0 for ΔF/F or 1 for Z-scored ΔF/F: "))

if choice == 0:
    plt.plot(time, dFF, label='ΔF/F')
    plt.ylabel('ΔF/F (%)')
elif choice == 1:
    plt.plot(time, dFF_z, label='Z-scored ΔF/F')
    plt.ylabel('Z-scored ΔF/F')


#plt.plot(time,control,label='CONTROLraw')
#plt.plot(time,signal)
# Plot OHRBETS events
'''
for event in events:
    t = event['timestamp_s']
    name = event['event']

    if t >= 0 and name == "tone":
        plt.axvline(t, color='red', linestyle='--', alpha=0.6, label='Tone')
    elif t >= 0 and name == "lick":
        plt.axvline(t, color='green', linestyle='--', alpha=0.6, label='Lick')
    elif t>= 0 and name == "trial_end":
        plt.axvline(t, color='black', linestyle='--', alpha=0.6, label='trial_end')
'''

for event in events:
    t=event['timestamp_s']
    name = event['event']
    code = event['code']

    if t>=0 and code == 1:
        plt.axvline(t-2, color='red', linestyle='--', alpha=0.6, label='sucrose admin')
    elif t>= 0 and code == 6:
        plt.axvline(t, color='green', linestyle='--', alpha=0.6, label='Entering Home Arena')
    elif t>= 0 and code == 4:
        plt.axvline(t, color='purple', linestyle='--', alpha=0.6, label='Exiting Foraging Arena')
# Plot PCT trial spans
#for start in pct_times:
#    plt.axvspan(start, start + trial_duration, color='orange', alpha=0.3, label='PCT Trial')

# Prevent duplicate labels
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

# Final touches
plt.xlabel('Time (s)')
plt.title('Fiber Photometry with PCT and OHRBETS Events')
plt.tight_layout()
plt.show()

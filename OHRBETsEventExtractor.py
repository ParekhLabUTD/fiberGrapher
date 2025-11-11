import csv

code_map = {
    51: 'tone',
    31: 'lick',
    13: 'spout extended',
    15: 'spout retracted',
    12: 'brake disengaged',
    688: 'lick onset',
    689: 'lick offset'
}

def extract_code(cell):
    return int(str(cell).strip().split()[0])

def parse_ohrbets_serial_log(file_path):
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        data = [row for row in reader if row]

    events = []
    codes = [extract_code(row[0]) for row in data]

    for i, code in enumerate(codes):
        if code in code_map:
            event_name = code_map[code]
            timestamp_s = float(data[i][1]) / 1000
            events.append({
                'code': code,
                'event': event_name,
                'timestamp_s': timestamp_s
            })

    # Add trial end
    events.append({
        'code': code,
        'event': 'trial_end',
        'timestamp_s': float(data[-1][1]) / 1000
    })
    return events


def lickBoutFilter(events):
    # --- Filter fake licks ---
    spoutExtendedTime = 0
    filtered_events = []

    for event in events:
        code = event["code"]
        ts = event["timestamp_s"]

        if code == 13 and ts != 0.0:
            spoutExtendedTime = ts

        if not (code == 31 and ts > spoutExtendedTime + 3.2):
            filtered_events.append(event)
        else:
            print(f"Removed fake lick at {ts}")

    events = filtered_events

    # --- Bout logic with relabeling ---
    bouts = []
    current_bout = {"start": None, "end": None, "lick_indices": []}

    for i, event in enumerate(events):
        code = event["code"]
        ts = event["timestamp_s"]

        if code == 13:  # spout extend
            current_bout = {"start": ts, "end": None, "lick_indices": []}

        elif code == 31 and current_bout["start"] is not None and current_bout["end"] is None:
            current_bout["lick_indices"].append(i)

        elif code == 15 and current_bout["start"] is not None:
            current_bout["end"] = ts
            if current_bout["lick_indices"]:
                bouts.append(current_bout)
            current_bout = {"start": None, "end": None, "lick_indices": []}

    for bout in bouts:
        lick_indices = bout["lick_indices"]
        if lick_indices:
            first_idx = lick_indices[0]
            last_idx = lick_indices[-1]
            events[first_idx]["code"] = 688
            events[first_idx]["event"] = "lick onset"
            events[last_idx]["code"] = 689
            events[last_idx]["event"] = "lick offset"

    return events

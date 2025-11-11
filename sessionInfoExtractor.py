import os
from datetime import datetime
from OHRBETsEventExtractor import *
from nxtEventExtrator import *
import tdt
#path = r'D:\Fiberphotometry'

def readfile(path):
    with open(path, 'r') as file:
        data = file.read()
    return data.split()      

def get_tdt_block_paths_with_events(path):
    tdt_block_paths = []

    for root, subdirs, files in os.walk(path):
        if files: 
            count_nxt = 0 
            try:
                files.index("StoresListing.txt")
                for file in files:
                    if file.split(".")[1] == "nxt" or file.split(".")[1] == "csv":
                        count_nxt = count_nxt + 1         
                if count_nxt > 0:
                    tdt_block_paths.append([root,count_nxt])
            except:
                 continue
    return tdt_block_paths              

#tdt_paths = get_tdt_block_paths_with_events(path)

def get_sessions_by_mouseIDs(tdt_block_paths):
    session_info = []

    for block in tdt_block_paths:
        data = readfile(os.path.join(block[0], "StoresListing.txt"))

        # Safely extract metadata fields
        mouse_id = data[3] if len(data) > 3 else "Unknown"
        experiment = data[1] if len(data) > 1 else "Unknown"
        date_str = data[7] if len(data) > 7 else ""
        time_str = data[9] if len(data) > 9 else ""

        # Try parsing date/time into datetime object
        dt_obj = None
        if date_str and time_str:
            try:
                dt_obj = datetime.strptime(f"{date_str} {time_str}", "%m/%d/%Y %I:%M:%S%p")
            except ValueError:
                # Handle possible alternative formats or malformed entries
                try:
                    dt_obj = datetime.strptime(f"{date_str} {time_str}", "%m/%d/%Y %H:%M:%S")
                except Exception:
                    dt_obj = None

        session_info.append({
            'mouseID': mouse_id,
            'Experiment': experiment,
            'date': date_str,
            'time': time_str,
            'datetime': dt_obj,
            'event_files_present': block[1],
            'path': block[0],
            'signalChannelSet': 1,
            'event_file_path': None,
            'event_interpretor': None,
            'events': None
        })
    session_info = split_combined_mouse_ids(session_info)
    session_info = define_event_file_paths(session_info)
    session_info = finalize_event_path_assignment(session_info)
    session_info = add_events_to_sessions(session_info)
    return session_info

def add_events_to_sessions(metadata):
    for data in metadata:
        if data['event_interpretor'] == 1:
            print("not made yet")
        elif data['event_interpretor'] == 2:
            data['events']  = parse_ohrbets_serial_log(data['event_file_path'])
            data['events'] = lickBoutFilter(data['events'])
        else:
            data_temp = readfile(data['event_file_path'])
            code_map = {
                1: "event 1",
                2: "event 2",
                3: "event 3",
                4: "event 4",
                5: "event 5",
                6: "event 6",
                7: "event 7",
                9: "event 9",
                8: "event 8"
            }

            data['events'] = get_events(data_temp,code_map)
        
    return metadata

def split_combined_mouse_ids(session_info):
    expanded_sessions = []
    for entry in session_info:
        # Split by underscore if it seems like multiple mouse IDs
        if ("M" in entry["mouseID"] or "F" in entry["mouseID"]) and "_" in entry["mouseID"]:
            mouse_ids = entry["mouseID"].split("_")
            channel_ind = 1
            for mid in mouse_ids:
                new_entry = entry.copy()
                new_entry["mouseID"] = mid
                new_entry["signalChannelSet"] = channel_ind
                channel_ind = channel_ind + 1
                expanded_sessions.append(new_entry)
        else:
            expanded_sessions.append(entry)
    return expanded_sessions

def define_event_file_paths(metadata):
    """
    Assigns an event file path and interpretor to each session folder.
    Priority:
        1. all_events.csv -> interpretor 1
        2. OHRBETS .csv     -> interpretor 2
        3. .nxt           -> interpretor 3
    """
    for data in metadata:
        data["event_file_path"] = None
        data["event_interpretor"] = None

        files = os.listdir(data["path"])
        # Priority 1: all_events.csv
        if "all_events.csv" in files:
            data["event_file_path"] = os.path.join(data["path"], "all_events.csv")
            data["event_interpretor"] = 1
            continue

        # Priority 2: other csv
        csv_files = [f for f in files if f.lower().endswith(".csv") and f != "all_events.csv"]
        if csv_files:
            data["event_file_path"] = os.path.join(data["path"], csv_files[0])
            data["event_interpretor"] = 2
            continue

        # Priority 3: nxt files
        nxt_files = [f for f in files if f.lower().endswith(".nxt")]
        if nxt_files:
            data["event_file_path"] = os.path.join(data["path"], nxt_files[0])
            data["event_interpretor"] = 3
            continue

    return metadata

def finalize_event_path_assignment(metadata):
    '''
    Need to handle cases with 2 OHRBETS mice and both raw csv are in same tdt block folder
    get mID and compare - yes 
    Need to handle case where 2 EF mice have one events file associated -- future, currently remove trace 2
    '''
    for data in metadata:
        if data["event_interpretor"] == 2 and data["event_files_present"] == 2:
            files = os.listdir(data["path"])
            csv_files = [f for f in files if f.lower().endswith(".csv") and f != "all_events.csv"]
            for file in csv_files:
                if (file.split(".")[0])[-3:]==data['mouseID']:
                    data["event_file_path"] = os.path.join(data["path"], file)
    return metadata

#get_sessions_by_mouseIDs(tdt_paths)

def get_tdt_block_paths(path):
    tdt_block_paths = []

    for root, subdirs, files in os.walk(path):
        if files:  
            try:
                files.index("StoresListing.txt")
                tdt_block_paths.append([root,0])
            except:
                 continue
    return tdt_block_paths        

def get_date_from_tdt_name(date_str):
    year ='20'+date_str[0]+date_str[1]
    mon = date_str[2]+date_str[3]
    day = date_str[4]+date_str[5]
    return mon+'/'+day+'/'+year

def get_sessions_by_mouseIDs_dumb(tdt_block_paths):
    mouseIDs = [os.path.basename(p[0]).split("-")[0] for p in tdt_block_paths]
    raw_dates = [os.path.basename(p[0]).split("-")[1] for p in tdt_block_paths]
    print(raw_dates)
    dates = [get_date_from_tdt_name(d) for d in raw_dates]
    print(dates)
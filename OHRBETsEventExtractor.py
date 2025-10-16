import csv

code_map = {
    51: 'tone',
    31: 'lick',
    13: 'spout extended',
    15: 'spout retracted',
}

# Utility function to get event codes
def extract_code(cell):
        return int(str(cell).strip().split()[0])

# Input the file path of the OHRBETS .csv 
def parse_ohrbets_serial_log(file_path):
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        data = [row for row in reader if row]
    
    events =[]

    codes = [extract_code(row[0]) for row in data]

    i=0
    for code in codes:
        i+=1
        if code in code_map:
            event_name = code_map[code]
            timestamp_s = float(data[i][1])/1000

            events.append({
                    'code': code,
                    'event': event_name,
                    'timestamp_s': timestamp_s
                })
    
    events.append({
         'code': code,
         'event' : "trial_end",
         "timestamp_s" : float(data[len(data)-1][1])/1000})
    return events

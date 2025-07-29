import csv

def extract_code(cell):
        return int(str(cell).strip().split()[0])

def load_data(file_path):
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        data = [row for row in reader if row]
    return data


code_map = {
    51: 'tone',
    31: 'lick'
}

def parse_ohrbets_serial_log(data):
    
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

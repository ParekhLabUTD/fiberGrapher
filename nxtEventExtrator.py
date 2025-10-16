def readfile(path):
    with open(path, 'r') as file:
        data = file.read()
    return data.split()

def get_event_codes(data):
    fifth_values = data[4::5]
    unique_sorted = sorted(set(fifth_values))
    i=0
    while i <len(unique_sorted):
        unique_sorted[i] = int(unique_sorted[i])
        i=i+1
        
    return unique_sorted

def get_events(data,code_map):
    i=2
    extracted = []

    while i<len(data):
        extracted.append(float(data[i]))
        i+=2
        extracted.append(int(data[i]))
        i+=3

    events =[]
    ind = 1
    while ind<len(extracted):
        events.append({
                    'code': extracted[ind],
                    'event': code_map[extracted[ind]],
                    'timestamp_s': extracted[ind-1]
                })
        ind+=2

    return events
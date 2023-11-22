import re

def getTrajs(file_path):
    ret_records = []
    buffer_records = []
    data = open(file_path, 'r')
    for item in data:
        line = item.strip()
        line = re.split(' |,', line)
        if line[0][0] == '-':
            ret_records.append(buffer_records)
            buffer_records = []
        else:
            buffer_records.append(line)
    return ret_records
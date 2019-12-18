import os
import csv

def revision_metafile(meta_file):
    with open(meta_file, 'r') as f:
        lines = f.readlines()[1:]
    new_item = {}
    label_list = []

    for line in lines:
        filepath, name, label_id = line.rstrip().split(',')
        ffolder, fname = filepath.split('/')[-2:]
        new_filepath = os.path.join("/media/Datasets/Fbank-train", ffolder, fname)
        new_item['filepath'] = new_filepath
        label_list.append({'filepath': new_filepath, 'name': name, 'label_id': label_id})
    headers = ['filepath','name','label_id']
    with open('new_train.csv', 'w') as f:
        f_scv = csv.DictWriter(f,headers)
        f_scv.writeheader()
        f_scv.writerows(label_list)


def get_dataset_voice(csv_file):
    voice_list = []

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader,None)
        for rows in reader:
            voice_list.append({'filepath': rows[0], 'name': rows[1],'label_id':int(rows[2])})
    
    names = {item['name'] for item in voice_list}
    names = list(names)
    
    return voice_list, len(names)





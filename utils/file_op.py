import json

def write_json(e, file:str):
    with open(file, 'w') as f:
        json.dump(e, f, indent=4)


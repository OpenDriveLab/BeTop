import pandas as pd 
import numpy as np

def token_checker(tok):
    if not isinstance(tok, str):
        tok = str(tok)
    assert len(tok) == 16
    return tok

df = pd.read_csv('/cpfs01/user/liuhaochen/behavioro/test14_filtered.csv')
tokens = df['token'].values.tolist()

new_tokens = [token_checker(tok) for tok in tokens]

print(len(tokens))

import yaml

filter_path = '/cpfs01/user/liuhaochen/behavioro/planning/config/scenario_filter/'

# Load the existing YAML configuration
with open(filter_path + 'test14-hard.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Update the configuration with the new list of strings under the desired argument
config['scenario_tokens'] = new_tokens

# Write the updated configuration back to the YAML file
with open(filter_path + 'test14-filter.yaml', 'w') as file:
    yaml.safe_dump(config, file)
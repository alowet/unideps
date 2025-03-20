import json

import requests

for layer in range(26):
    print(f"Downloading layer {layer}")
    url = f'https://www.neuronpedia.org/api/explanation/export?modelId=gemma-2-2b&saeId={layer}-gemmascope-res-16k'
    response = requests.get(url)
    data = response.json()
    with open(f'data/neuronpedia/layer_{layer}.json', 'w') as f:
        json.dump(data, f)

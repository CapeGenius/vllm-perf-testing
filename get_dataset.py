import requests
import pandas as pd

# decides the size of the dataframe
num_requests = 10
total_df = pd.DataFrame()

# dataset options: theblackcat102/evol-codealpaca-v1
#                  HuggingFaceH4/no_robots

# requests the dataset and gets the first 100*num_requests rows into a csv
for i in range(num_requests):

    url = "https://datasets-server.huggingface.co/rows"
    params = {
        "dataset": "theblackcat102/evol-codealpaca-v1",
        "config": "default",
        "split": "train",
        "offset": i * 100,
        "length": 100,
    }

    headers = {
        "Accept": "application/json",
    }

    response = requests.get(url, params=params, headers=headers)
    data = response.json()

    # replace "instruction" with prompt for no-robots dataset
    prompts = [row["row"]["instruction"] for row in data["rows"]]

    df = pd.DataFrame({"prompt": prompts})

    total_df = pd.concat(
        [total_df, df],
        axis=0,
    )

total_df.to_csv("src/datasets/evol-codealpaca-v1.csv")

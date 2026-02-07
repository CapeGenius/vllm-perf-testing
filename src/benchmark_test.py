import asyncio
import requests
import json
import numpy as np
import os
from dotenv import load_dotenv


async def run_request():

    load_dotenv()

    IP_ADDRESS = os.getenv("IP_ADDRESS")

    url = f"http://{IP_ADDRESS}/v1/completions"

    headers = {"Content-Type": "application/json"}

    data = {
        "model": "Qwen/Qwen3-0.6B",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0,
    }

    print(url)

    response = requests.post(url, json=data, headers=headers, verify=False)
    print(response)
    # response_content = json.loads(response.content)
    # latency_times = response_content["kv_transfer_params"]
    # print(f"Prefill Time: {latency_times["prefill_time"]}")
    # print(f"Decode Time: {latency_times["decode_time"]}")


if __name__ == "__main__":
    asyncio.run(run_request())

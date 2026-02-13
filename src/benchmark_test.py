import asyncio
import requests
import json
import numpy as np
import os
from dotenv import load_dotenv
import time
import pandas as pd
import threading

load_dotenv()


def run_request(
    model: str = "Qwen/Qwen3-0.6B",
    dataset: str = "San Francisco is a",
    max_tokens: int = 200,
    temperature: float = 0,
):
    """
    Creates an API request to the vLLM based on the specified model, dataset, max token window, and temperature.

    Returns a JSON response from REST API request.

    model: str - which vLLM model --> depends on the vLLM model that's currently serving
    dataset: str - string of the dataset that will be sent to vLLM
    max_tokens: int - the integer of maximum tokens for response window
    temperature: float value for model temperature
    """

    IP_ADDRESS = os.getenv("IP_ADDRESS")

    url = f"http://{IP_ADDRESS}/v1/completions"

    headers = {"Content-Type": "application/json"}

    data = {
        "model": f"{model}",
        "prompt": f"{dataset}",
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    print(url)

    response = requests.post(url, json=data, headers=headers, verify=False)

    if response.status_code == 200:
        return response
    return None

    # response_content = json.loads(response.content)
    # latency_times = response_content["kv_transfer_params"]
    # print(f"Prefill Time: {latency_times["prefill_time"]}")
    # print(f"Decode Time: {latency_times["decode_time"]}")


def get_statistics(response):
    """
    Gets vLLM metrics from the "kv_transfer_params" from the API request response, by loading
    json into a dict -- and getting dictionary matching "kv_transfer_params".



    response: Response object - response object from api rquest
    vllm_request_metrics: dict - dictionary with vllm request metrics
    """

    response_content = json.loads(response.content)
    vllm_request_metrics = response_content["kv_transfer_params"]
    return vllm_request_metrics


def send_to_fluentbit(metrics, host="fluent-bit", port=9880, tag="vllm"):
    print("sending to fluent-bit")
    url = f"http://{host}:{port}/{tag}"
    r = requests.post(url, json=metrics, timeout=2)
    r.raise_for_status()


def send_event(response):
    # get environment variables
    host = os.getenv("host")
    port = int(os.getenv("port"))
    tag = os.getenv("tag")

    vllm_request_metrics = get_statistics(response)
    send_to_fluentbit(vllm_request_metrics, host=host, port=port, tag=tag)


def create_event():
    # get environment variables
    model = os.getenv("model")
    dataset = os.getenv("dataset")
    max_tokens = int(os.getenv("max_tokens"))
    temperature = float(os.getenv("temperature"))

    dataset = pd.read_csv(f"/app/datasets/{dataset}.csv")
    random_prompt = dataset.sample()["prompt"]

    # runs the request
    response = run_request(
        model=model,
        dataset=random_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if response is None:
        print("Request failed")
        return None

    # gets the statistic from the response, creates an event, and then sends it to fluent bit
    send_event(response)


def send_multiple_requests(interval: float = 10, num_requests: int = 5):
    """
    send_multiple_requests sends multiple requests every n seconds, defined by variable interval,
    and will be sent num_request times

    interval: int - period of time between requests
    num_requests: int - number of requests
    :type num_requests: int
    """

    threads = []

    for i in range(int(num_requests)):
        print(f"number of requests {i+1}/{num_requests}")
        t = threading.Thread(
            target=create_event,
        )
        t.start()
        threads.append(t)
        time.sleep(interval)

    for t in threads:
        t.join()
        print("finished execution")


if __name__ == "__main__":
    interval = float(os.getenv("interval"))
    num_requests = int(os.getenv("num_requests"))
    send_multiple_requests(interval=interval, num_requests=num_requests)

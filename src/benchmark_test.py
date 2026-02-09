import asyncio
import requests
import json
import numpy as np
import os
from dotenv import load_dotenv
import time

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


def send_to_fluentbit(metrics, host="fluentbit", port=9880, tag="vllm"):
    url = f"http://{host}:{port}/{tag}"
    r = requests.post(url, json=metrics, timeout=2)
    r.raise_for_status()


def send_event(response):
    # get environment variables
    host = os.getenv("host")
    port = os.getenv("port")
    tag = os.getenv("tag")

    vllm_request_metrics = get_statistics(response)
    send_to_fluentbit(vllm_request_metrics, host=host, port=port, tag=tag)


def create_event():
    # get environment variables
    model = os.getenv("model")
    dataset = os.getenv("dataset")
    max_tokens = os.getenv("max_tokens")
    temperature = os.getenv("temperature")

    # runs the request
    response = run_request(
        model=model, dataset=dataset, max_tokens=max_tokens, temperature=temperature
    )

    # gets the statistic from the response, creates an event, and then sends it to fluent bit
    send_event(response)


def send_multiple_requests(interval: int = 10, num_requests: int = 5):
    """
    send_multiple_requests sends multiple requests every n seconds, defined by variable interval,
    and will be sent num_request times

    interval: int - period of time between requests
    num_requests: int - number of requests
    :type num_requests: int
    """

    for _ in range(num_requests):
        asyncio.run(create_event())
        time.sleep(interval)


if __name__ == "__main__":
    interval = os.getenv("interval")
    num_requests = os.getenv("num_requests")
    asyncio.run(send_multiple_requests(interval=interval, num_requests=num_requests))

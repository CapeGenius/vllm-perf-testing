import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

load_dotenv()

# get env

dataset = os.getenv("dataset")
caption_note = os.getenv("caption")
interval = int(os.getenv("interval"))
num_requests = int(os.getenv("num_requests"))
model = os.getenv("model")
caption_txt = f"(Note: {caption_note}). dataset: {dataset}, model: {model}, arrival rate (requests/minute): {60 / interval}, number of requests: {num_requests}"

log_file = "/logger/vllm.log"

rows = []

pattern = re.compile(r"vllm:\s*\[\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*(\{.*\})\s*\]")

with open(log_file) as f:
    for line in f:
        m = pattern.search(line)
        if not m:
            continue

        response_ts = float(m.group(1))
        payload = json.loads(m.group(2))

        arrival_ts = payload.get("arrival_time")
        if arrival_ts is None:
            continue

        payload["response_ts"] = response_ts
        payload["e2e_latency"] = response_ts - arrival_ts

        rows.append(payload)

# print(rows)
df = pd.DataFrame(rows)

print(df.columns)

plt.figure()
plt.scatter(df["throughput"], df["e2e_latency"], alpha=0.6)
plt.xlabel("Throughput (tokens/sec)")
plt.ylabel("End-to-End Latency (s)")
plt.title("Throughput vs End-to-End Latency")
plt.figtext(
    0.5, -0.2, caption_txt, wrap=True, horizontalalignment="center", fontsize=11
)
plt.show()

plt.savefig("plots/throughput_vs_latency.png", dpi=150, bbox_inches="tight")

plt.figure()
plt.scatter(df["prefill_time"], df["decode_time"], alpha=0.6)
plt.xlabel("Prefill Time (s)")
plt.ylabel("Decode Time (s)")
plt.title("Prefill vs Decode Time")
plt.show()
plt.figtext(
    0.5, -0.2, caption_txt, wrap=True, horizontalalignment="center", fontsize=11
)
plt.savefig("plots/prefill_vs_decode.png", dpi=150, bbox_inches="tight")

plt.figure()
plt.scatter(df["average_itl"], df["time_to_first_token"], alpha=0.6)
plt.xlabel("Average Inter-Token Latency (s)")
plt.ylabel("Time to First Token (s)")
plt.title("Average Inter-Token Latency vs Time to First Token")
plt.show()
plt.figtext(
    0.5, -0.2, caption_txt, wrap=True, horizontalalignment="center", fontsize=11
)
plt.savefig("plots/itl_vs_ttft.png", dpi=150, bbox_inches="tight")

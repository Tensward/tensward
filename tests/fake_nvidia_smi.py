"""A stand-in for ``nvidia-smi --query-gpu=... --loop-ms=...``: prints a warning line, then
FAKE_SMI_ROW (the fields after the index, comma separated) for device 0 every 20 ms until
stopped. With FAKE_SMI_OLD set, a query naming clocks_event_reasons is refused, as older
drivers do."""

import os
import sys
import time

query = next(word for word in sys.argv if word.startswith("--query-gpu="))
if os.environ.get("FAKE_SMI_OLD") and "clocks_event_reasons" in query:
    refused = "clocks_event_reasons.sw_power_cap"
    print(f'Field "{refused}" is not a valid field to query.', file=sys.stderr)
    sys.exit(2)
fields = query.split("=", 1)[1].split(",")
row = os.environ["FAKE_SMI_ROW"].split(",")
print("WARNING: infoROM is corrupted at gpu 00000000:00:1E.0", flush=True)
while True:
    print(", ".join(["0", *row[: len(fields) - 1]]), flush=True)
    time.sleep(0.02)

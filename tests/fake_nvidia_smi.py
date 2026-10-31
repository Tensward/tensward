"""A stand-in for ``nvidia-smi --query-gpu=... --loop-ms=...``: prints a warning line, then
FAKE_SMI_ROW (the fields after the index, comma separated) for device 0 every 20 ms until
stopped; the memory fields come from FAKE_SMI_MEMORY ("used,total" in MiB, default
"1024,24576"). A query for the PCIe link answers generation 3, width 16. With FAKE_SMI_OLD
set, a query naming clocks_event_reasons is refused, as older drivers do."""

import os
import sys
import time

query = next(word for word in sys.argv if word.startswith("--query-gpu="))
if "pcie.link" in query:
    print("3, 16")
    sys.exit(0)
if os.environ.get("FAKE_SMI_OLD") and "clocks_event_reasons" in query:
    refused = "clocks_event_reasons.sw_power_cap"
    print(f'Field "{refused}" is not a valid field to query.', file=sys.stderr)
    sys.exit(2)
fields = query.split("=", 1)[1].split(",")
row = os.environ["FAKE_SMI_ROW"].split(",")
used, total = os.environ.get("FAKE_SMI_MEMORY", "1024,24576").split(",")
memory = {"memory.used": used, "memory.total": total}
print("WARNING: infoROM is corrupted at gpu 00000000:00:1E.0", flush=True)
while True:
    others = iter(row)
    values = [memory[f] if f in memory else next(others) for f in fields[1:]]
    print(", ".join(["0", *values]), flush=True)
    time.sleep(0.02)

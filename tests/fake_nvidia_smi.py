"""A stand-in for ``nvidia-smi --query-gpu=... --loop-ms=...``: prints a warning line, then
FAKE_SMI_ROW (the fields after the index, comma separated) for device 0 every 20 ms until
stopped; the memory fields come from FAKE_SMI_MEMORY ("used,total" in MiB, default
"1024,24576"). A query for the PCIe link answers generation 3, width 16. With FAKE_SMI_OLD
set, a query naming clocks_event_reasons is refused, as older drivers do. A query for compute
apps answers the rows of FAKE_SMI_APPS ("pid,MiB,uuid;..."), none when it is unset. A query for
name, memory and identity without --loop-ms answers once, with no warning line, one row for the
device named by FAKE_SMI_DEVICE (default "NVIDIA L4") and the memory of FAKE_SMI_MEMORY."""

import os
import sys
import time

if any(word.startswith("--query-compute-apps=") for word in sys.argv):
    for app in filter(None, os.environ.get("FAKE_SMI_APPS", "").split(";")):
        print(", ".join(app.split(",")))
    sys.exit(0)
query = next(word for word in sys.argv if word.startswith("--query-gpu="))
if "pcie.link" in query:
    print("3, 16")
    sys.exit(0)
if "--loop-ms" not in " ".join(sys.argv) and query.startswith("--query-gpu=name,"):
    used, total = os.environ.get("FAKE_SMI_MEMORY", "1024,24576").split(",")
    name = os.environ.get("FAKE_SMI_DEVICE", "NVIDIA L4")
    print(
        f"{name}, {total}, {used}, 0, GPU-0f1e2d3c, 00000000:00:1E.0, 550.54.15, 0x27B810DE, 95.04"
    )
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

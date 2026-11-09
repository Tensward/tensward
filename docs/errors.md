# Refusal codes

A refused command prints `error: <message> [<code>]` on stderr, with a `fix:` line when the message does not already name a command. With `--json`, or `TENSWARD_JSON=1`, it prints `{"code": ..., "message": ...}` instead. Exit 2 means the environment refuses; exit 3 means a declared input does not meet its contract.

| Code | Exit | Fix |
| --- | --- | --- |
| `checkpoint_act_order` | 3 | see the message |
| `checkpoint_changed` | 2 | see the message |
| `checkpoint_inventory_unexpected` | 3 | see the message |
| `checkpoint_inventory_unsafe` | 3 | see the message |
| `checkpoint_layout_invalid` | 3 | see the message |
| `checkpoint_precision_unsupported` | 3 | see the message |
| `checkpoint_unsupported` | 3 | see the message |
| `engine_unavailable` | 2 | run `tensward env` to see what is installed |
| `gpu_busy` | 2 | stop the process named in the message, or run on another machine |
| `gpu_mismatch` | 2 | see the message |
| `no_metal_build` | 2 | install llama.cpp with `brew install llama.cpp` |
| `project_busy` | 2 | wait for the other tensward command to finish |
| `project_config_unsupported` | 3 | see the message |
| `project_inputs_changed` | 3 | use `--project ./another-folder`, or remove `.tensward` to start over |
| `project_inputs_invalid` | 3 | see the message |
| `project_layout_invalid` | 3 | see the message |
| `project_not_registered` | 2 | run `tensward init --model PATH` |
| `project_publication_failed` | 2 | see the message |
| `project_record_invalid` | 3 | see the message |
| `project_snapshot_mismatch` | 3 | see the message |
| `project_state_unsafe` | 2 | see the message |
| `runner_failure` | 2 | see the message |

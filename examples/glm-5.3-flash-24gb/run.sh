#!/usr/bin/env bash
# Run GLM-5.3-Flash (2-bit GGUF) on one 24 GB NVIDIA card with llama.cpp.
# ./run.sh help   shows all commands.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="$HERE/.state"
mkdir -p "$STATE"
PIDFILE="$STATE/server.pid"
LOGFILE="$STATE/server.log"
HOST=127.0.0.1
PORT="${PORT:-8080}"

TAG="b12722"
COMMIT="e6443bd7eb970d940b0669cd9ea0d82d91d620a1"
ARCHIVE="${ARCHIVE:-llama-${TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz}"
ARCHIVE_URL="${ARCHIVE_URL:-https://github.com/ggml-org/llama.cpp/releases/download/${TAG}/${ARCHIVE}}"
SOURCE_URL="${SOURCE_URL:-https://github.com/ggml-org/llama.cpp/archive/${COMMIT}.tar.gz}"
MODEL_REV="${MODEL_REV:-a38483c8cd5df544f53d70fb281afe97369d5ab6}"
MODEL_BASE="${MODEL_BASE:-https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/resolve/${MODEL_REV}/UD-Q2_K_XL}"
SHARDS=(
  GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf
  GLM-5.3-Flash-UD-Q2_K_XL-00002-of-00004.gguf
  GLM-5.3-Flash-UD-Q2_K_XL-00003-of-00004.gguf
  GLM-5.3-Flash-UD-Q2_K_XL-00004-of-00004.gguf
)
PINNED_MIN_GIB=118
PROBE_CACHE_MIB=1024
PEAK_PROMPT_TOKENS=7000
CTX=40960

say() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
NEXT_QUIET=0
next() { # prints the menu of likely next commands
  (( NEXT_QUIET )) && return 0
  say ""; say "Next:"; local l; for l in "$@"; do say "  $l"; done
}
ask_yes() { # question; returns 0 on yes. Without a terminal the answer is no.
  [[ -t 0 ]] || return 1
  local a; read -r -p "$1 [y/N] " a || return 1
  [[ "$a" == y || "$a" == Y || "$a" == yes ]]
}

get_dir() { [[ -f "$STATE/dir" ]] || die "No download folder is set yet, so there is nothing to run. Run ./run.sh (it sets everything up) or ./run.sh fetch /path/to/folder"; cat "$STATE/dir"; }
ram_gib() { awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo; }
card_field() { nvidia-smi --query-gpu="$1" --format=csv,noheader,nounits | head -1 | tr -d ' '; }
phys_cores() {
  local n; n="$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)"
  [[ "$n" -ge 1 ]] && echo "$n" || nproc
}
sysctl_value() { cat /proc/sys/vm/compaction_proactiveness 2>/dev/null || echo unknown; }

pick_mode() {
  local ram; ram="$(ram_gib)"
  if [[ -n "${MODE_OVERRIDE:-}" ]]; then
    MODE="$MODE_OVERRIDE"; MODE_WHY="chosen by --mode"
  elif (( ram >= PINNED_MIN_GIB )); then
    MODE=pinned; MODE_WHY="${ram} GiB RAM is enough to keep the whole expert part of the model pinned in RAM"
  else
    MODE=mapped; MODE_WHY="${ram} GiB RAM is not enough to pin the model; the file is mapped and read from the SSD as needed"
  fi
  case "$MODE" in pinned|mapped) ;; *) die "--mode takes pinned or mapped." ;; esac
}

cuda_runtime_found() {
  local lib missing=0
  for lib in libcudart.so.12 libcublas.so.12; do
    if ldconfig -p 2>/dev/null | grep -q "$lib"; then continue; fi
    if [[ -n "${CUDA_LIB_DIR:-}" && -e "$CUDA_LIB_DIR/$lib" ]]; then continue; fi
    if [[ -f "$STATE/dir" ]] && find "$(cat "$STATE/dir")" -name "$lib*" 2>/dev/null | grep -q .; then continue; fi
    say "  missing: $lib"; missing=1
  done
  return $missing
}

cmd_check() {
  while (( $# )); do case "$1" in --mode) MODE_OVERRIDE="${2:-}"; shift 2 ;; *) die "Unknown option $1" ;; esac; done
  say "== Card =="
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,compute_cap --format=csv,noheader
  else
    say "nvidia-smi not found: no NVIDIA driver on this machine."
  fi
  say "== Machine =="
  say "RAM: $(ram_gib) GiB    CPU threads: $(nproc)    physical cores: $(phys_cores)"
  if [[ -f "$STATE/dir" ]]; then
    local dir; dir="$(cat "$STATE/dir")"
    say "Folder: $dir"
    df -h "$dir" | tail -1 | awk '{print "Disk free there: " $4}'
    if [[ -f "$dir/${SHARDS[1]}" ]]; then
      say "Sequential read speed of the model file (1 GiB, bypassing the page cache):"
      dd if="$dir/${SHARDS[1]}" of=/dev/null bs=1M count=1024 skip=2048 iflag=direct 2>&1 | tail -1 || say "  could not measure"
    fi
  else
    df -h "$HERE" | tail -1 | awk '{print "Disk free here: " $4 "  (the model needs about 120 GB)"}'
  fi
  say "== CUDA 12 runtime =="
  if cuda_runtime_found; then say "found"; else
    say "The prebuilt release needs the CUDA 12 runtime libraries. Without them it silently runs on the CPU."
    say "Install them, or set CUDA_LIB_DIR to the folder holding libcudart.so.12 and libcublas.so.12."
  fi
  pick_mode
  say "== Mode =="
  say "Mode: $MODE. Reason: $MODE_WHY."
  say "Sizes of RAM between about 64 and ${PINNED_MIN_GIB} GiB are untested. Use --mode to choose."
  if [[ "$MODE" == pinned ]]; then
    say "== Kernel setting for the pinned mode =="
    local v; v="$(sysctl_value)"
    say "vm.compaction_proactiveness is now: $v"
    if [[ "$v" != 0 ]]; then
      say "With this much memory pinned, the kernel's memory-compaction thread can keep one core busy and slow replies."
      say "Run this once (the kit never runs sudo itself; it lasts until reboot):"
      say "  sudo sysctl -w vm.compaction_proactiveness=0"
    else
      say "Already 0. Nothing to do."
    fi
  fi
  next "./run.sh          (does everything: download, start, speed test)" "./run.sh fetch ./glm-data   (only download)"
}

remote_size() { curl -sIL "$1" | tr -d '\r' | awk 'tolower($1)=="content-length:" {v=$2} END {print v}'; }

download_resumable() { # url file
  local url="$1" file="$2" remote
  remote="$(remote_size "$url")"
  if [[ -f "$file" && -n "$remote" && "$(stat -c %s "$file")" == "$remote" ]]; then
    say "Already complete: $(basename "$file")"; return
  fi
  curl -L --fail -C - --retry 5 -o "$file" "$url"
}

cmd_fetch() {
  local dir="" source=0 yes=0
  while (( $# )); do case "$1" in --source) source=1; shift ;; -y) yes=1; shift ;; *) dir="$1"; shift ;; esac; done
  [[ -n "$dir" ]] || { [[ -f "$STATE/dir" ]] && dir="$(cat "$STATE/dir")" || dir="$HERE/glm-data"; }
  mkdir -p "$dir"; dir="$(cd "$dir" && pwd)"; echo "$dir" > "$STATE/dir"
  local bad=0 s
  local need_dl=0
  for s in "${SHARDS[@]}"; do [[ -f "$dir/$s" ]] || need_dl=1; done
  if (( need_dl )); then
    say "Download size: about 109 GB for the model plus a small engine file. Free disk space in $dir: $(df -h --output=avail "$dir" | tail -1 | tr -d ' ')."
    say "Time: depends on your line; at 100 MB/s about 20 minutes. Downloads can be resumed: run the same command again."
    if (( ! yes )); then
      ask_yes "Download about 109 GB to $dir?" || die "Nothing was downloaded. To download, run ./run.sh again from a terminal and answer y, or add -y: ./run.sh fetch $dir -y"
    fi
  fi
  if (( source )); then
    say "Downloading the llama.cpp source ($TAG)..."
    # The source host does not support resuming, so this is one plain download.
    curl -L --fail --retry 3 -o "$dir/llama.cpp-$COMMIT.tar.gz" "$SOURCE_URL"
  else
    say "Downloading the llama.cpp release ($ARCHIVE)..."
    download_resumable "$ARCHIVE_URL" "$dir/$ARCHIVE"
    mkdir -p "$dir/llama"
    case "$ARCHIVE" in
      *.zip) if command -v unzip >/dev/null; then unzip -oq "$dir/$ARCHIVE" -d "$dir/llama"
             else python3 -I -c 'import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' "$dir/$ARCHIVE" "$dir/llama"; fi ;;
      *) tar -xzf "$dir/$ARCHIVE" -C "$dir/llama" ;;
    esac
    find "$dir/llama" -type f -name 'llama-*' ! -name '*.so*' -exec chmod +x {} + || true
  fi
  for s in "${SHARDS[@]}"; do
    say "Downloading $s ..."
    download_resumable "$MODEL_BASE/$s" "$dir/$s"
  done
  say "Checking sizes..."
  for s in "${SHARDS[@]}"; do
    local r; r="$(remote_size "$MODEL_BASE/$s")"
    if [[ -n "$r" && "$r" == "$(stat -c %s "$dir/$s")" ]]; then say "  ok: $s"; else say "  SIZE MISMATCH: $s (run fetch again)"; bad=1; fi
  done
  say "Checksums (reads the whole download, takes a while):"
  (cd "$dir" && sha256sum "${SHARDS[@]}" $( (( source )) && echo "llama.cpp-$COMMIT.tar.gz" || echo "$ARCHIVE"))
  (( bad == 0 )) || die "A file has the wrong size, so the model may be damaged. Run ./run.sh fetch $dir again: it resumes and repairs."
  (( source )) || touch "$dir/.fetched-ok"
  next "./run.sh serve" "./run.sh build --patches routing,warm   (optional, needs: ./run.sh fetch $dir --source)"
}

cmd_build() {
  local want=""
  while (( $# )); do case "$1" in --patches) want="${2:-}"; shift 2 ;; *) die "Unknown option $1" ;; esac; done
  local dir; dir="$(get_dir)"
  local tarball="$dir/llama.cpp-$COMMIT.tar.gz"
  [[ -f "$tarball" ]] || die "The engine source is not downloaded yet, and the build needs it. Run: ./run.sh fetch $dir --source"
  for t in cmake nvcc patch make; do command -v "$t" >/dev/null || die "The build needs the tool $t, which is not installed. Install it with your package manager, or skip the build: the release works without it (./run.sh serve)."; done
  local cc; cc="$(card_field compute_cap)"; cc="${cc/./}"
  [[ "$cc" =~ ^[0-9]+$ ]] || die "Could not read the card's compute capability, so the build cannot pick the right code. Check that nvidia-smi works (./run.sh check)."
  say "Card compute capability: $cc"
  rm -rf "$dir/src"; mkdir -p "$dir/src"
  tar -xzf "$tarball" -C "$dir/src" --strip-components=1
  local applied="" p name
  # Patches are always applied in this fixed order.
  for name in advise routing warm; do
    if [[ ",$want," == *",$name,"* ]]; then
      p="$(ls "$HERE"/patches/*-"$name".diff)"
      say "Applying patch: $name"
      (cd "$dir/src" && patch -p1 --no-backup-if-mismatch < "$p") || die "Patch $name did not apply to this source. Run ./run.sh fetch $dir --source again for a clean copy, then build again."
      applied="$applied $name"
    fi
  done
  for name in ${want//,/ }; do case "$name" in advise|routing|warm) ;; *) die "Unknown patch: $name. Choose from advise, routing, warm." ;; esac; done
  say "Building takes about 10 to 20 minutes and uses all cores."
  (cd "$dir/src" && cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
      -DLLAMA_CURL=OFF -DCMAKE_CUDA_ARCHITECTURES="$cc" >/dev/null && \
    cmake --build build --config Release -j "$(nproc)" --target llama-server)
  mkdir -p "$dir/llama-built"
  cp "$dir/src/build/bin/llama-server" "$dir/llama-built/"
  echo "${applied# }" > "$STATE/patches"
  say "Built: $dir/llama-built/llama-server   patches:${applied:- none}"
  next "./run.sh serve   (the built engine is used when it exists; add --engine release to use the release)"
}

has_patch() { [[ "$ENGINE_KIND" == built && -f "$STATE/patches" ]] && grep -qw "$1" "$STATE/patches"; }

find_engine() {
  local dir; dir="$(get_dir)"
  if [[ "${ENGINE_PREF:-auto}" != release && -x "$dir/llama-built/llama-server" ]]; then
    ENGINE="$dir/llama-built/llama-server"; ENGINE_KIND=built
  else
    ENGINE="$(find "$dir/llama" -type f -name llama-server 2>/dev/null | head -1 || true)"; ENGINE_KIND=release
  fi
  [[ -n "$ENGINE" ]] || die "The engine (llama-server) is not in the download folder, so the server cannot start. Run ./run.sh fetch"
}

server_alive() { [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

start_server() { # args = server flags; env comes from the caller
  local dir model; dir="$(get_dir)"; model="$dir/${SHARDS[0]}"
  [[ -f "$model" ]] || die "The model file is missing: $model. Run ./run.sh fetch to download it."
  server_alive && die "A server is already running (pid $(cat "$PIDFILE")), and a second one would not fit on the card. Use it (./run.sh chat), or run ./run.sh stop first."
  find_engine
  export LD_LIBRARY_PATH="$(dirname "$ENGINE")${CUDA_LIB_DIR:+:$CUDA_LIB_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  card_field memory.used > "$STATE/used_before"
  : > "$LOGFILE"
  nohup "$ENGINE" -m "$model" --host "$HOST" --port "$PORT" "$@" >> "$LOGFILE" 2>&1 &
  echo $! > "$PIDFILE"
  say "Server starting (pid $(cat "$PIDFILE")), engine: $ENGINE_KIND, log: $LOGFILE"
  say "Loading takes about 3 to 4 minutes on a fast SSD, longer on a slow disk."
  local i code=""
  for (( i = 0; i < 1800; i++ )); do
    if ! server_alive; then tail -20 "$LOGFILE" >&2; die "The server stopped while starting (last log lines above, full log: $LOGFILE). Often the card is busy or too small. Check with ./run.sh check, then try ./run.sh serve exact."; fi
    code="$(curl -s -o /dev/null -w '%{http_code}' "http://$HOST:$PORT/health" || true)"
    [[ "$code" == 200 ]] && break
    sleep 2
  done
  [[ "$code" == 200 ]] || { cmd_stop; die "The server did not answer within an hour, so it was stopped. See $LOGFILE, then try ./run.sh serve again."; }
  # The engine's default log does not name the card, so ask the driver how much card memory this server holds.
  local used; used="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', *' -v p="$(cat "$PIDFILE")" '$1==p {print $2}' | head -1)"
  if [[ -n "$used" && "$used" -ge 1024 ]]; then
    say "Card in use: this server holds $used MiB on the card."
  else
    cmd_stop
    die "This server holds no memory on the card, so it would run on the CPU only and be very slow.
The release needs the CUDA 12 runtime libraries (libcudart.so.12, libcublas.so.12).
Install them or set CUDA_LIB_DIR=/folder/with/them, then run ./run.sh check and ./run.sh serve."
  fi
}

# Cache size = probe cache + card memory still free at the lowest point while a long
# prompt is read - headroom. Probes with the same flags the real server will use.
size_cache() { # headroom_mib args...
  local headroom="$1"; shift
  say "Sizing the expert cache: one start with a ${PROBE_CACHE_MIB} MiB cache and a long prompt..." >&2
  start_server "$@" --moe-cache-mib "$PROBE_CACHE_MIB" >&2
  ( while :; do card_field memory.free >> "$STATE/free_samples"; sleep 1; done ) &
  local sampler=$!
  : > "$STATE/free_samples"
  python3 -I "$HERE/bench.py" --host "$HOST" --port "$PORT" --peak-probe "$PEAK_PROMPT_TOKENS" >&2 || true
  kill "$sampler" 2>/dev/null || true
  local minfree; minfree="$(sort -n "$STATE/free_samples" | head -1)"
  cmd_stop >&2
  [[ -n "$minfree" ]] || die "Could not read the card's free memory, so the cache cannot be sized. Check that nvidia-smi works (./run.sh check), or set the size yourself: ./run.sh serve --cache-mib 6000"
  local c=$(( PROBE_CACHE_MIB + minfree - headroom ))
  (( c >= 512 )) || die "Not enough free card memory for a useful cache (lowest free: ${minfree} MiB). Close other programs that use the card (nvidia-smi shows them), or try ./run.sh serve exact --ctx 16384"
  say "Lowest free card memory: ${minfree} MiB -> cache ${c} MiB (headroom ${headroom} MiB)." >&2
  echo "$c"
}

next_serve() { next "./run.sh chat" "./run.sh agent" "./run.sh bench" "./run.sh stop"; }

cmd_serve() {
  local preset="" experts="" cache="" routing=0 ctx="$CTX"
  while (( $# )); do
    case "$1" in
      exact|fast|jobs4|mapped) preset="$1"; shift ;;
      --routing)   routing=1; shift ;;
      --experts)   experts="${2:-}"; shift 2 ;;
      --cache-mib) cache="${2:-}"; shift 2 ;;
      --ctx)       ctx="${2:-}"; shift 2 ;;
      --mode)      MODE_OVERRIDE="${2:-}"; shift 2 ;;
      --engine)    ENGINE_PREF="${2:-}"; shift 2 ;;
      --resize)    RESIZE=1; shift ;;
      *) die "Unknown option: $1. Run ./run.sh help." ;;
    esac
  done
  pick_mode
  [[ -n "$preset" ]] || { [[ "$MODE" == pinned ]] && preset=fast || preset=mapped; }
  if [[ "$MODE" == mapped && "$preset" != mapped ]]; then die "Preset $preset needs about ${PINNED_MIN_GIB} GiB of RAM and this machine has $(ram_gib). Use the low-RAM mode: ./run.sh serve mapped"; fi
  if [[ "$MODE" == pinned && "$preset" == mapped ]]; then die "Preset mapped is for machines with less RAM; this one can pin the model, which is faster. Run ./run.sh serve, or force it with ./run.sh serve mapped --mode mapped"; fi
  find_engine
  say "Mode: $MODE   preset: $preset   ($MODE_WHY)"

  local extra=() envs=()
  if [[ -n "$experts" ]]; then
    [[ "$experts" =~ ^[0-9]+$ ]] && (( experts >= 1 && experts <= 8 )) || die "--experts takes a number from 1 to 8 (the model's own value is 8). Example: ./run.sh serve fast --experts 6"
    say "WARNING: --experts $experts uses fewer experts per token than the model's own 8. Answers WILL change. Check them yourself."
    extra+=(--override-kv "glm5-next.expert_used_count=int:$experts")
  fi

  if [[ "$MODE" == mapped ]]; then
    local args=(--jinja -fa on -np 1 -c "$ctx" --fit on -b 4096 -ub 4096 -t "$(phys_cores)")
    if has_patch advise; then
      envs+=(LLAMA_MOE_ADVISE=1)
      say "Read-ahead patch is on: it speeds up the first reply after a prompt. Steady reply speed does not change."
    else
      say "No read-ahead patch built. This mode is slow, especially the first prompt. One stream only."
    fi
    (( routing )) && say "Note: --routing is for the pinned mode and is ignored here."
    say "Loading from the SSD can take many minutes, depending on the disk."
    start_server "${args[@]}" "${extra[@]}"
    echo "$MODE $preset" > "$STATE/preset"
    say "Ready at http://$HOST:$PORT  (API at /v1)."
    next_serve
    return
  fi

  # Pinned mode
  local v; v="$(sysctl_value)"
  if [[ "$v" != 0 ]]; then
    say "NOTE: vm.compaction_proactiveness is $v. Replies can be slower. This is an optional speed-up; continuing anyway. To set it: sudo sysctl -w vm.compaction_proactiveness=0"
  fi
  local split=0.80 np=(-np 1) headroom=600 draft=()
  case "$preset" in
    exact) ;;
    fast)  draft=(--spec-type draft-mtp --spec-draft-n-max 1); headroom=1024 ;;
    jobs4) np=(-np 4 -kvu); split=0.95 ;;
  esac
  if (( routing )); then
    [[ "$preset" == jobs4 ]] && die "--routing works on one stream, not with jobs4. Use ./run.sh serve fast --routing or ./run.sh serve jobs4"
    has_patch routing || die "--routing needs the engine built with our routing patch, and this engine has none. Run: ./run.sh fetch ./glm-data --source and then ./run.sh build --patches routing"
    if [[ ${#draft[@]} -gt 0 ]]; then
      say "Both the draft head and --routing were asked. They do not combine: using --routing, draft head off."
      draft=(); headroom=600
    fi
    say "ROUTING ON: answers change slightly (the router may pick a cached expert with a close score)."
    envs+=(LLAMA_MOE_STICKY_D=0.03 LLAMA_MOE_STICKY_KEEP=4 LLAMA_MOE_STICKY_CREDIT=router)
  fi
  if (( ctx > CTX )) && [[ ${#draft[@]} -gt 0 ]]; then
    say "WARNING: a context above $CTX with the draft head can run the card out of memory on a long prompt."
  fi
  local e
  for e in "${envs[@]+"${envs[@]}"}"; do export "$e"; done
  local common=(--jinja -fa on "${np[@]}" -c "$ctx" -ngl all --cpu-moe --fit off -lm none
                -b 4096 -ub 4096 --moe-cache-cpu-share "$split"
                --chat-template-kwargs '{"reasoning_effort":"low"}')
  [[ ${#draft[@]} -gt 0 ]] && common+=("${draft[@]}")
  [[ ${#extra[@]} -gt 0 ]] && common+=("${extra[@]}")

  if [[ -z "$cache" ]]; then
    local cardslug; cardslug="$(card_field name | tr -c 'A-Za-z0-9\n' '_')-$(card_field memory.total)"
    local key="$preset-$ctx-${routing}-${experts:-8}-$cardslug"
    if [[ -f "$STATE/cache_$key" && "${RESIZE:-0}" != 1 ]]; then
      cache="$(cat "$STATE/cache_$key")"
      say "Reusing the saved cache size. Loading takes about 3 to 4 minutes on a fast SSD. (--resize measures again.)"
    else
      say "The first start measures the card once and then starts again: about 8 minutes in all. Later starts reuse the size and take about 3 to 4 minutes."
      cache="$(size_cache "$headroom" "${common[@]}")"; echo "$cache" > "$STATE/cache_$key"
    fi
  fi
  say "Expert cache on the card: ${cache} MiB"
  local hot="$STATE/hot-$preset-$cache.bin"
  if has_patch warm; then
    export LLAMA_MOE_WARM_SAVE="$hot"
    if [[ -f "$hot" ]]; then export LLAMA_MOE_WARM_LOAD="$hot"; say "Loading the saved list of hot experts."; fi
  fi
  start_server "${common[@]}" --moe-cache-mib "$cache"
  echo "$MODE $preset" > "$STATE/preset"
  say "Ready at http://$HOST:$PORT  (API at /v1)."
  next_serve
}

cmd_warm() {
  local file=""
  while (( $# )); do case "$1" in --file) file="${2:-}"; shift 2 ;; *) die "Unknown option $1. Run ./run.sh help." ;; esac; done
  server_alive || die "No server is running, so there is nothing to warm. Start it first: ./run.sh serve"
  find_engine
  say "== Expert cache =="
  if has_patch warm; then
    say "The warm patch is built: the hot list is saved when the server stops and loaded at the next start. Nothing to do now."
  else
    say "Generating a few hundred tokens on your kind of text so the card cache fills with the experts you use..."
    python3 -I "$HERE/chat.py" --port "$PORT" --tokens 400 ${file:+--file "$file"} \
      --once "Continue or summarise the following text in detail." >/dev/null
    say "Done."
  fi
  say "== Prompt cache =="
  say "Run ./run.sh agent --warm from the folder you work in. The agent's long opening prompt is then read once and kept."
  next "./run.sh agent --warm" "./run.sh chat"
}

cmd_agent() {
  local warm=0
  while (( $# )); do case "$1" in --warm) warm=1; shift ;; *) die "Unknown option $1" ;; esac; done
  local id="glm-5.3-flash"
  if server_alive; then
    local got; got="$(curl -s "http://$HOST:$PORT/v1/models" | python3 -I -c 'import sys,json
try: print(json.load(sys.stdin)["data"][0]["id"])
except Exception: pass' || true)"
    [[ -n "$got" ]] && id="$got"
  fi
  local out="$PWD/opencode.json"
  if [[ -f "$out" ]]; then out="$PWD/opencode.json.new"; say "opencode.json exists here; writing $out instead. Merge it yourself."; fi
  cat > "$out" <<JSON
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local llama.cpp",
      "options": { "baseURL": "http://$HOST:$PORT/v1", "chunkTimeout": false },
      "models": { "$id": { "name": "GLM-5.3-Flash (local)" } }
    }
  }
}
JSON
  say "Wrote $out"
  if command -v opencode >/dev/null; then say "The agent (opencode) is installed."
  else say "The agent (opencode) is not installed. The kit does not install it. One command (needs Node 22):  npm install -g opencode-ai"; fi
  say "Start it from this folder with:"
  say "  OPENCODE_DISABLE_AUTOUPDATE=1 OPENCODE_DISABLE_MODELS_FETCH=1 opencode --model local/$id"
  if (( warm )); then
    command -v opencode >/dev/null || die "The agent (opencode) is not installed, so it cannot send the warm-up request. Install it with: npm install -g opencode-ai (needs Node 22), then run ./run.sh agent --warm"
    server_alive || die "No server is running, so there is nothing to warm. Start it first: ./run.sh serve"
    say "Sending one throwaway request so the agent's opening prompt is read and cached..."
    OPENCODE_DISABLE_AUTOUPDATE=1 OPENCODE_DISABLE_MODELS_FETCH=1 opencode run --model "local/$id" 'hello' >/dev/null || true
    say "Done. Use the agent from this same folder, soon: the opening prompt must be the same."
  else
    say "Optional: ./run.sh agent --warm sends one throwaway request so the agent's long opening prompt is read once now, not at your first real question."
  fi
  next "./run.sh agent --warm" "./run.sh chat" "./run.sh stop"
}

server_up() { [[ "$(curl -s -o /dev/null -w '%{http_code}' "http://$HOST:$PORT/health" 2>/dev/null || true)" == 200 ]]; }
need_server() {
  server_up && return 0
  say "No server is running."
  say "Start it with: ./run.sh serve   (or ./run.sh to do everything)"
  exit 1
}

cmd_bench() {
  need_server
  say "The speed check takes a few minutes."
  python3 -I "$HERE/bench.py" --host "$HOST" --port "$PORT" --out "$HERE/result.json" --log "$LOGFILE"
  next "./run.sh chat" "./run.sh agent" "./run.sh stop"
}

cmd_chat() {
  need_server
  python3 -I "$HERE/chat.py" --port "$PORT"
  next "./run.sh agent" "./run.sh bench" "./run.sh stop"
}

cmd_status() {
  if server_alive; then
    local pid; pid="$(cat "$PIDFILE")"
    say "Server: running (pid $pid), $(cat "$STATE/preset" 2>/dev/null || echo "preset unknown")"
    say "Address: http://$HOST:$PORT  (API at /v1)"
    local used; used="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', *' -v p="$pid" '$1==p {print $2}' | head -1)"
    say "Card memory: this server holds ${used:-?} MiB; free on the card: $(card_field memory.free 2>/dev/null || echo ?) MiB"
    server_up || say "It is not answering yet: it may still be loading."
    next "./run.sh chat" "./run.sh agent" "./run.sh stop"
  else
    say "Server: not running."
    next "./run.sh serve" "./run.sh"
  fi
}

cmd_stop() {
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    local pid i; pid="$(cat "$PIDFILE")"
    kill "$pid" || true
    for (( i = 0; i < 120; i++ )); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" || true
    rm -f "$PIDFILE"
    if command -v nvidia-smi >/dev/null; then
      local before=0 now
      [[ -f "$STATE/used_before" ]] && before="$(cat "$STATE/used_before")"
      for (( i = 0; i < 180; i++ )); do
        now="$(card_field memory.used)"
        (( now <= before + 300 )) && break
        sleep 1
      done
      (( now <= before + 300 )) || say "The card still shows ${now} MiB used (${before} MiB before). Another program may be using it."
    fi
    say "Stopped server (pid $pid); card memory is free."
  else
    rm -f "$PIDFILE"; say "No server running."
  fi
}

cmd_stop_user() { cmd_stop; next "./run.sh serve"; }

# One command for the whole path.
cmd_up() {
  local yes=0 dir="" passthru=()
  while (( $# )); do
    case "$1" in
      -y) yes=1; shift ;;
      --dir) dir="${2:-}"; shift 2 ;;
      --mode|--resize) passthru+=("$1"); [[ "$1" == --mode ]] && { passthru+=("${2:-}"); shift; }; shift ;;
      *) die "Unknown option: $1. Run ./run.sh help." ;;
    esac
  done
  NEXT_QUIET=1
  say "Step 1 of 4: checking this machine"
  command -v nvidia-smi >/dev/null || die "No NVIDIA driver was found (nvidia-smi is missing), and the model needs the card. Install the NVIDIA driver, then run ./run.sh again."
  cuda_runtime_found || die "The CUDA 12 runtime libraries are missing (listed above). Without them the engine silently runs on the CPU. Install them, or set CUDA_LIB_DIR=/folder/with/them, then run ./run.sh again."
  local i; for (( i = 0; i < ${#passthru[@]}; i++ )); do [[ "${passthru[i]}" == --mode ]] && MODE_OVERRIDE="${passthru[i+1]}"; done
  pick_mode
  say "Card: $(card_field name), $(card_field memory.total) MiB. RAM: $(ram_gib) GiB. Mode: $MODE ($MODE_WHY)."
  (( $(ram_gib) >= 56 )) || die "This machine has $(ram_gib) GiB of RAM; the model needs at least about 64 GiB. It cannot run here."
  [[ "$MODE" == pinned ]] && [[ "$(sysctl_value)" != 0 ]] && say "Optional speed-up (run it yourself, the kit never uses sudo): sudo sysctl -w vm.compaction_proactiveness=0. Continuing without it."

  say "Step 2 of 4: the files"
  if [[ -z "$dir" ]]; then
    if [[ -f "$STATE/dir" ]]; then dir="$(cat "$STATE/dir")"; else dir="$HERE/glm-data"; say "No folder chosen: using $dir (change it with --dir /path)."; fi
  fi
  mkdir -p "$dir"; dir="$(cd "$dir" && pwd)"
  if [[ -f "$dir/.fetched-ok" ]]; then
    echo "$dir" > "$STATE/dir"; say "All files are already in $dir."
  else
    cmd_fetch "$dir" $( (( yes )) && echo -y ) 
  fi

  say "Step 3 of 4: starting the server"
  if server_alive; then say "A server is already running; keeping it."
  else cmd_serve ${passthru[@]+"${passthru[@]}"}; fi

  say "Step 4 of 4: a 10-second speed probe"
  python3 -I "$HERE/bench.py" --host "$HOST" --port "$PORT" --quick || true
  NEXT_QUIET=0
  say "All set."
  next_serve
}

usage() { cat <<'U'
./run.sh                 does everything: check, download (asks first), start, speed probe
./run.sh check [--mode pinned|mapped]
./run.sh fetch [DIR] [--source] [-y]
./run.sh build [--patches advise,routing,warm]      (optional)
./run.sh serve [exact|fast|jobs4|mapped] [--routing] [--experts N] [--cache-mib N] [--resize] [--ctx N] [--mode M] [--engine release|built]
./run.sh status | chat | bench | agent [--warm] | warm [--file FILE] | stop
U
}


cmd="${1:-up}"; [[ $# -gt 0 ]] && shift
[[ "$cmd" == -* && "$cmd" != -h && "$cmd" != --help ]] && { set -- "$cmd" "$@"; cmd=up; }
case "$cmd" in
  up) cmd_up "$@" ;;
  check) cmd_check "$@" ;;
  fetch) cmd_fetch "$@" ;;
  build) cmd_build "$@" ;;
  serve) cmd_serve "$@" ;;
  warm)  cmd_warm "$@" ;;
  bench) cmd_bench ;;
  chat)  cmd_chat ;;
  status) cmd_status ;;
  agent) cmd_agent "$@" ;;
  stop)  cmd_stop_user ;;
  help|-h|--help) usage ;;
  *) say "Unknown command: $cmd"; usage; exit 1 ;;
esac

#!/usr/bin/env bash
# Host-side Spark cluster watchdog: reachability, temps, memory, GPU, health.
# Logs to .scratch/watchdog/ (disk, never /tmp). Does not auto-relaunch.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${WATCHDOG_OUT_DIR:-${ROOT_DIR}/.scratch/watchdog}"
INTERVAL_SEC="${WATCHDOG_INTERVAL_SEC:-30}"
HOSTS="${WATCHDOG_HOSTS:-spark1 spark2}"
API_URL="${WATCHDOG_API_URL:-http://spark1:8000/health}"
SPARKRUN_BIN="${SPARKRUN_BIN:-/home/maci/bin/sparkrun}"
WOL_MAC_SPARK2="${WOL_MAC_SPARK2:-30:c5:99:be:49:81}"
SEND_WOL_ON_DOWN="${WATCHDOG_SEND_WOL:-1}"

mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/watchdog.log"
CSV="${OUT_DIR}/samples.csv"
EVENTS="${OUT_DIR}/events.log"
PIDFILE="${OUT_DIR}/watchdog.pid"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

log() {
  local line
  line="$(ts) $*"
  printf '%s\n' "${line}" | tee -a "${LOG}" >/dev/null
}

event() {
  local line
  line="$(ts) EVENT $*"
  printf '%s\n' "${line}" | tee -a "${EVENTS}" "${LOG}" >/dev/null
}

send_wol() {
  local mac_hex="$1"
  python3 - "$mac_hex" <<'PY' || true
import socket, sys
mac = bytes.fromhex(sys.argv[1].replace(":", ""))
pkt = b"\xff" * 6 + mac * 16
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
for b in ("192.168.0.255", "255.255.255.255"):
    s.sendto(pkt, (b, 9))
print("wol_sent", sys.argv[1])
PY
}

ssh_ok() {
  local host="$1"
  ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
    "${host}" 'true' >/dev/null 2>&1
}

remote_sample() {
  local host="$1"
  # Single SSH: return pipe-separated fields. Missing sensors -> n/a.
  ssh -o ConnectTimeout=5 -o BatchMode=yes "${host}" 'bash -s' <<'REMOTE' || echo "ssh_fail|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a"
set +e
gpu_t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
gpu_u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
pwr=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
mem_used=$(free -m | awk '/^Mem:/{print $3}')
mem_tot=$(free -m | awk '/^Mem:/{print $2}')
swap_used=$(free -m | awk '/^Swap:/{print $3}')
# hottest acpitz zone
soc_max=$(for z in /sys/class/thermal/thermal_zone*; do
  [ -f "$z/temp" ] || continue
  cat "$z/temp" 2>/dev/null
done | sort -nr | head -1 | awk '{printf "%.1f", $1/1000}')
load=$(cut -d' ' -f1-3 /proc/loadavg)
docker_n=$(docker ps -q --filter name=sparkrun 2>/dev/null | wc -l | tr -d ' ')
vllm_n=$(pgrep -af 'vllm serve|VLLM::Worker' 2>/dev/null | grep -vc 'grep' || true)
oom_recent=$(journalctl -k --since '5 min ago' --no-pager 2>/dev/null | grep -c 'NV_ERR_NO_MEMORY' || true)
xid_recent=$(journalctl -k --since '5 min ago' --no-pager 2>/dev/null | grep -ciE 'NVRM: Xid|gpu has fallen' || true)
printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
  "ok" "${gpu_t:-n/a}" "${gpu_u:-n/a}" "${pwr:-n/a}" \
  "${soc_max:-n/a}" "${mem_used:-n/a}" "${mem_tot:-n/a}" "${swap_used:-n/a}" \
  "${docker_n:-0}" "${vllm_n:-0}" "${oom_recent:-0}/${xid_recent:-0}"
REMOTE
}

# CSV header once
if [[ ! -f "${CSV}" ]]; then
  echo "ts,host,reachable,gpu_c,gpu_util,power_w,soc_max_c,mem_used_mb,mem_tot_mb,swap_used_mb,docker_n,vllm_n,oom_xid_5m,health,sparkrun" >"${CSV}"
fi

if [[ -f "${PIDFILE}" ]]; then
  old=$(cat "${PIDFILE}" 2>/dev/null || true)
  if [[ -n "${old}" ]] && kill -0 "${old}" 2>/dev/null; then
    echo "watchdog already running pid=${old} log=${LOG}" >&2
    exit 0
  fi
fi
echo $$ >"${PIDFILE}"
trap 'rm -f "${PIDFILE}"' EXIT

declare -A was_down
for h in ${HOSTS}; do was_down["$h"]=0; done

log "watchdog start interval=${INTERVAL_SEC}s hosts=${HOSTS} out=${OUT_DIR}"

while true; do
  now=$(ts)
  health=$(curl -sS -m 3 -o /dev/null -w '%{http_code}' "${API_URL}" 2>/dev/null || echo 000)
  sparkrun_line="n/a"
  if [[ -x "${SPARKRUN_BIN}" ]]; then
    sparkrun_line=$("${SPARKRUN_BIN}" status -H spark1,spark2 2>/dev/null | tr '\n' ';' | sed 's/;*$//' | cut -c1-200 || echo status_fail)
  fi

  for host in ${HOSTS}; do
    if ssh_ok "${host}"; then
      sample=$(remote_sample "${host}")
      IFS='|' read -r st gpu_t gpu_u pwr soc memu memt swapu dock vllm oomxid <<<"${sample}"
      echo "${now},${host},1,${gpu_t},${gpu_u},${pwr},${soc},${memu},${memt},${swapu},${dock},${vllm},${oomxid},${health},\"${sparkrun_line}\"" >>"${CSV}"

      # Alerts
      if [[ "${gpu_t}" != "n/a" ]] && awk "BEGIN{exit !(${gpu_t}+0 >= 85)}"; then
        event "${host} HIGH_GPU_TEMP ${gpu_t}C util=${gpu_u}% power=${pwr}W"
      fi
      if [[ "${soc}" != "n/a" ]] && awk "BEGIN{exit !(${soc}+0 >= 95)}"; then
        event "${host} HIGH_SOC_TEMP ${soc}C"
      fi
      if [[ "${oomxid}" != "n/a" && "${oomxid}" != "0/0" ]]; then
        event "${host} KERNEL_GPU_ERR oom_xid_5m=${oomxid}"
      fi
      if [[ "${was_down[$host]}" == "1" ]]; then
        event "${host} RECOVERED gpu=${gpu_t}C soc=${soc}C mem=${memu}/${memt}MB"
        was_down["$host"]=0
      fi
      # Soft note if vLLM gone but container remains
      if [[ "${dock}" != "0" && "${vllm}" == "0" ]]; then
        event "${host} CONTAINER_UP_VLLM_GONE docker=${dock}"
      fi
    else
      echo "${now},${host},0,n/a,n/a,n/a,n/a,n/a,n/a,n/a,0,0,n/a,${health},\"${sparkrun_line}\"" >>"${CSV}"
      if [[ "${was_down[$host]}" == "0" ]]; then
        event "${host} UNREACHABLE health=${health}"
        was_down["$host"]=1
        if [[ "${SEND_WOL_ON_DOWN}" == "1" && "${host}" == "spark2" ]]; then
          send_wol "${WOL_MAC_SPARK2}" | while read -r w; do event "WOL ${w}"; done
          # also send from spark1 if up
          if ssh_ok spark1; then
            ssh -o ConnectTimeout=5 -o BatchMode=yes spark1 \
              "python3 -c \"
import socket
mac=bytes.fromhex('${WOL_MAC_SPARK2//:/}')
pkt=b'\\\\xff'*6+mac*16
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET,socket.SO_BROADCAST,1)
for b in ('192.168.0.255','255.255.255.255'):
  s.sendto(pkt,(b,9))
print('spark1_wol_sent')
\"" 2>/dev/null | while read -r w; do event "WOL ${w}"; done || true
          fi
        fi
      fi
    fi
  done

  if [[ "${health}" != "200" ]]; then
    # only event once per sustained outage via simple flag file
    if [[ ! -f "${OUT_DIR}/.health_down" ]]; then
      event "API_DOWN health=${health} sparkrun=${sparkrun_line}"
      touch "${OUT_DIR}/.health_down"
    fi
  else
    if [[ -f "${OUT_DIR}/.health_down" ]]; then
      event "API_UP health=200"
      rm -f "${OUT_DIR}/.health_down"
    fi
  fi

  sleep "${INTERVAL_SEC}"
done

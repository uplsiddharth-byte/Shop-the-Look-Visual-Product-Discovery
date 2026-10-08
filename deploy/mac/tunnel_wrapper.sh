#!/bin/bash
# Runs the Cloudflare quick tunnel to the local app. When the tunnel announces its public address, waits until the app and the
# address both answer, saves it, and publishes it for the stable live.html link. launchd restarts this script if the tunnel exits.
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
LOGDIR="$HOME/Library/Logs/shop-the-look"; mkdir -p "$LOGDIR"
CF="$(command -v cloudflared || echo /opt/homebrew/bin/cloudflared)"
publish_when_ready() {
  local url="$1"
  for _ in $(seq 1 90); do curl -sf -m 5 localhost:8000/api/health >/dev/null && break; sleep 2; done          # the app has to be up first
  for _ in $(seq 1 60);  do curl -sf -m 8 "$url/api/health" >/dev/null && break; sleep 2; done                  # then the public address
  echo "$url" > "$ROOT/data/tunnel_url.txt"
  "$HERE/publish_live_url.sh" "$url" >> "$LOGDIR/publish.log" 2>&1 || echo "publish failed (see publish.log)" >> "$LOGDIR/tunnel.log"
}
SEEN=""
"$CF" tunnel --no-autoupdate --url http://localhost:8000 2>&1 | while IFS= read -r line; do
  echo "$line" >> "$LOGDIR/tunnel.log"
  if [ -z "$SEEN" ] && [[ "$line" =~ (https://[a-z0-9-]+\.trycloudflare\.com) ]]; then
    SEEN=1; publish_when_ready "${BASH_REMATCH[1]}" &
  fi
done

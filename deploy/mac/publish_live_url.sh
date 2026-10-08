#!/bin/bash
# Publish the current public address of the app to the GitHub Pages branch (live.json + live.html), so one stable link
# (https://<user>.github.io/<repo>/live.html) always leads to the running app. Usage: publish_live_url.sh https://xxxx.trycloudflare.com
set -euo pipefail
URL="$1"
REPO_URL="https://github.com/uplsiddharth-byte/Shop-the-Look-Visual-Product-Discovery.git"
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="$HOME/.shop-the-look/ghpages"
mkdir -p "$HOME/.shop-the-look"
[ -d "$DIR/.git" ] || git clone -q --branch gh-pages --single-branch "$REPO_URL" "$DIR"
cd "$DIR"
git pull -q --rebase origin gh-pages
printf '{"url":"%s","updated":"%s"}\n' "$URL" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > live.json
cp "$HERE/live.html" live.html
git add live.json live.html
git commit -q -m "Update live app address" || true
git push -q origin gh-pages
echo "published $URL"

"""Install (or remove) two macOS login jobs that keep the app and its public tunnel running:
  com.shopthelook.server  runs ./run.sh (kept awake with caffeinate), restarted if it crashes
  com.shopthelook.tunnel  runs the Cloudflare tunnel and publishes its address to the stable live.html link
Usage: python3 deploy/mac/install.py [--uninstall]. Logs: ~/Library/Logs/shop-the-look/. Stop everything: --uninstall."""
import os, plistlib, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
AGENTS = Path.home() / "Library/LaunchAgents"
LOGS = Path.home() / "Library/Logs/shop-the-look"
UID = os.getuid()
JOBS = {
    "com.shopthelook.server": ["/usr/bin/caffeinate", "-i", str(ROOT / "run.sh")],
    "com.shopthelook.tunnel": ["/bin/bash", str(ROOT / "deploy/mac/tunnel_wrapper.sh")],
}


def launchctl(*args):
    return subprocess.run(["launchctl", *args], capture_output=True, text=True)


for label in JOBS:
    plist = AGENTS / f"{label}.plist"
    launchctl("bootout", f"gui/{UID}/{label}")  # stop it if it is running (ignore "not loaded")
    for _ in range(30):  # unloading is asynchronous: wait until launchd has really forgotten the job, or the next bootstrap fails (error 5)
        if launchctl("print", f"gui/{UID}/{label}").returncode != 0:
            break
        time.sleep(1)
    if "--uninstall" in sys.argv:
        plist.unlink(missing_ok=True)
        print("removed", label)
        continue
    AGENTS.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
    with open(plist, "wb") as f:
        plistlib.dump({
            "Label": label,
            "ProgramArguments": JOBS[label],
            "WorkingDirectory": str(ROOT),
            "EnvironmentVariables": {"PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin", "HOME": str(Path.home())},
            "RunAtLoad": True,       # start at login
            "KeepAlive": True,       # restart if it exits or crashes
            "ThrottleInterval": 30,  # at most one restart every 30 s
            "StandardOutPath": str(LOGS / f"{label.split('.')[-1]}.out.log"),
            "StandardErrorPath": str(LOGS / f"{label.split('.')[-1]}.err.log"),
        }, f)
    for attempt in range(5):  # retry: bootstrap can still race with a job that is shutting down
        r = launchctl("bootstrap", f"gui/{UID}", str(plist))
        if r.returncode == 0:
            break
        time.sleep(2)
    print(("started " if r.returncode == 0 else "FAILED to start ") + label, r.stderr.strip())

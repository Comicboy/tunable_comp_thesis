#!/usr/bin/env bash
# Set up the Quest-over-USB-C link.
#
# Why this works without ngrok or HTTPS: `adb reverse` makes the headset's
# localhost:PORT resolve to this machine's localhost:PORT, and `http://localhost` is
# a potentially-trustworthy origin per the W3C secure-contexts spec. WebXR
# requires a secure context, and localhost satisfies it -- so navigator.xr is
# available over plain HTTP with no tunnel and no certificate.
#
# The forward does NOT survive unplugging the cable or the headset sleeping.
# Re-run this script whenever the headset reconnects.

set -euo pipefail

PORT="${1:-8080}"

if ! command -v adb >/dev/null 2>&1; then
  cat >&2 <<'EOF'
error: adb not found.

  brew install --cask android-platform-tools

Then enable Developer Mode for the headset in the Meta Horizon phone app
(Devices -> your headset -> Headset Settings -> Developer Mode).
EOF
  exit 1
fi

echo "==> Devices:"
adb devices -l

STATE="$(adb devices | awk 'NR>1 && NF {print $2; exit}')"

case "${STATE:-none}" in
  none|"")
    cat >&2 <<'EOF'

error: no device detected. Work through these in order:
  1. USB-C cable plugged into the headset and this machine (a data cable, not
     charge-only -- this is the most common cause).
  2. Developer Mode enabled in the Meta Horizon phone app.
  3. Headset powered on and worn, so it is not asleep.
  4. Accept the "Allow USB debugging?" prompt INSIDE the headset.
EOF
    exit 1
    ;;
  unauthorized)
    cat >&2 <<'EOF'

error: device is 'unauthorized'.
Put the headset on and accept the "Allow USB debugging?" prompt.
Tick "Always allow from this computer" so it does not ask again.
EOF
    exit 1
    ;;
  device) ;;
  *)
    echo "error: device in unexpected state '$STATE'." >&2
    exit 1
    ;;
esac

echo
echo "==> Forwarding headset localhost:$PORT -> this Mac's localhost:$PORT"
adb reverse --remove-all >/dev/null 2>&1 || true
adb reverse "tcp:$PORT" "tcp:$PORT"

echo "==> Active reverse forwards:"
adb reverse --list

cat <<EOF

Ready. Now:
  1. Start a teleop script on this machine (it serves the page on port $PORT).
  2. In the headset, open Meta Quest Browser and go to:

         http://localhost:$PORT

  3. Tap "Start Teleop" and allow the immersive session.

Tip: keep the headset awake while off your head with
     adb shell am broadcast -a com.oculus.vrpowermanager.prox_close
EOF

#!/usr/bin/env bash
# Install udev rules so the Quest and the Wuji Hand are usable without sudo.
#
# Run once with sudo. You must log out and back in afterwards for the plugdev
# group membership to take effect -- newgrp only affects the shell you run it
# in, which is a common source of "it still says no permissions".

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "error: run with sudo -- installing udev rules needs root." >&2
  echo "    sudo $0" >&2
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_USER="${SUDO_USER:-$USER}"

echo "==> Installing udev rules"
for rule in 49-oculus.rules 95-wujihand.rules; do
  install -m 0644 "$HERE/udev/$rule" "/etc/udev/rules.d/$rule"
  echo "    /etc/udev/rules.d/$rule"
done

echo "==> Ensuring the plugdev group exists and $TARGET_USER is in it"
getent group plugdev >/dev/null || groupadd plugdev
if id -nG "$TARGET_USER" | tr ' ' '\n' | grep -qx plugdev; then
  echo "    $TARGET_USER is already in plugdev"
  ALREADY_IN_GROUP=1
else
  usermod -aG plugdev "$TARGET_USER"
  echo "    added $TARGET_USER to plugdev"
  ALREADY_IN_GROUP=0
fi

echo "==> Reloading udev"
udevadm control --reload-rules
udevadm trigger

echo
echo "Installed. Now:"
if [[ "$ALREADY_IN_GROUP" -eq 0 ]]; then
  echo "  1. LOG OUT AND BACK IN -- the plugdev group is not active in existing"
  echo "     sessions, and this is the usual reason adb still reports"
  echo "     'no permissions' after running this script."
  echo "  2. Unplug and replug the Quest and the hand."
else
  echo "  1. Unplug and replug the Quest and the hand."
fi
echo "  3. Verify:  python scripts/troubleshooting/check_linux_stack.py"

#!/usr/bin/env bash
# One-time setup: allow non-root access to /dev/tcm via group "tcm".
# Run on the board: sudo ./setup-tcm-perms.sh [username]
set -euo pipefail

USER="${1:-${SUDO_USER:-orangepi}}"
RULE=/etc/udev/rules.d/99-spacemit-tcm.rules

if [[ "$(id -u)" -ne 0 ]]; then
    echo "Run with sudo: sudo $0 [$USER]" >&2
    exit 1
fi

if ! getent group tcm >/dev/null; then
    groupadd --system tcm
    echo "Created group tcm"
fi

if ! id -nG "$USER" | tr ' ' '\n' | grep -qx tcm; then
    usermod -aG tcm "$USER"
    echo "Added $USER to group tcm (log out/in or: newgrp tcm)"
fi

cat >"$RULE" <<'EOF'
# SpacemiT tightly-coupled memory (drivers/misc/tcm.c → /dev/tcm)
KERNEL=="tcm", MODE="0660", GROUP="tcm"
EOF
chmod 644 "$RULE"
echo "Wrote $RULE"

udevadm control --reload-rules
udevadm trigger --subsystem-match=misc --action=add 2>/dev/null || udevadm trigger

if [[ -e /dev/tcm ]]; then
    chgrp tcm /dev/tcm
    chmod 0660 /dev/tcm
    ls -la /dev/tcm
fi

echo "Done. Verify as $USER (new shell): test -r /dev/tcm && test -w /dev/tcm && echo OK"

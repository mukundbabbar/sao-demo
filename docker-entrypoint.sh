#!/bin/sh
set -e

if [ "${WAIT_FOR_POSTGRES:-1}" = "1" ]; then
  host="${POSTGRES_HOST:-postgres}"
  port="${POSTGRES_PORT:-5432}"
  echo "Waiting for PostgreSQL at ${host}:${port}..."
  until python -c "
import socket
s = socket.socket()
s.settimeout(2)
s.connect(('${host}', int('${port}')))
s.close()
" 2>/dev/null; do
    sleep 2
  done
  echo "PostgreSQL is up."
fi

exec "$@"

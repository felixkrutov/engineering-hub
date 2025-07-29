#!/bin/sh
set -e

hostport=$1
shift

until nc -z $(echo $hostport | cut -d : -f 1) $(echo $hostport | cut -d : -f 2); do
  >&2 echo "Database is unavailable - sleeping"
  sleep 1
done

>&2 echo "Database is up - executing command"
exec "$@"

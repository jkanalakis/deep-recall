#!/bin/bash

set -e

echo "Initializing PostgreSQL database with embeddings tables..."

# Check if POSTGRES_USER, POSTGRES_PASSWORD, and POSTGRES_DB are set
if [ -z "$POSTGRES_USER" ] || [ -z "$POSTGRES_PASSWORD" ] || [ -z "$POSTGRES_DB" ]; then
  echo "Error: Environment variables POSTGRES_USER, POSTGRES_PASSWORD, and POSTGRES_DB must be set."
  echo "Current values: POSTGRES_USER=$POSTGRES_USER, POSTGRES_DB=$POSTGRES_DB"
  exit 1
fi

# Wait for PostgreSQL to start - use socket connection instead of host
until pg_isready > /dev/null 2>&1; do
  echo "Waiting for PostgreSQL to start..."
  sleep 1
done

echo "PostgreSQL started, initializing embeddings tables..."

# Run the initialization script - use socket connection
psql -U $POSTGRES_USER -d $POSTGRES_DB -f /docker-entrypoint-initdb.d/init_embeddings_tables.sql

echo "Database initialization complete!" 
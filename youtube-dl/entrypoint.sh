#!/bin/bash
service cron start
exec /app/.venv/bin/python src/main.py

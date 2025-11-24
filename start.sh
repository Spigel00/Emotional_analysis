#!/usr/bin/env bash
# Start script for Render deployment with gunicorn configuration

gunicorn --config gunicorn.conf.py app:app

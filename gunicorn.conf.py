# Gunicorn configuration for Render deployment
# Optimized for SSE (Server-Sent Events) streams

import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"
backlog = 2048

# Worker processes - optimized for Render's resources
workers = int(os.getenv('WEB_CONCURRENCY', '4'))  # Default 4 workers, override with env var
worker_class = 'gevent'  # Use gevent for async/SSE support
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
threads = 2  # 2 threads per worker

# Timeout settings - CRITICAL for SSE
timeout = 300  # 5 minutes - allows long-running SSE connections
graceful_timeout = 300  # Give workers time to finish requests
keepalive = 5

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = 'emotional_analysis'

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (handled by Render)
keyfile = None
certfile = None

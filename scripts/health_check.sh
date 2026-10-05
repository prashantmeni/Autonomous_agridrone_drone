#!/usr/bin/env bash
curl -sf http://localhost:8000/api/health && echo "API OK" || echo "API DOWN"

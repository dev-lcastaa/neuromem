#!/usr/bin/env bash
# Verify k-NN plugin is initialized and Lucene engine is available.
set -euo pipefail

: "${OPENSEARCH_URL:=http://192.168.1.208:9200}"

auth_args=()
if [[ -n "${OPENSEARCH_USER:-}" && -n "${OPENSEARCH_PASSWORD:-}" ]]; then
    auth_args=(-u "${OPENSEARCH_USER}:${OPENSEARCH_PASSWORD}")
fi

echo "=== _plugins/_knn/stats (per-node engine init) ==="
curl -fsS "${auth_args[@]}" "${OPENSEARCH_URL}/_plugins/_knn/stats?pretty" \
    | grep -E '"(lucene_initialized|faiss_initialized|nmslib_initialized|knn_query_requests|graph_memory_usage)"' \
    || echo "(k-NN plugin not installed?)"

echo
echo "=== cluster version ==="
curl -fsS "${auth_args[@]}" "${OPENSEARCH_URL}/" | grep -E '"(number|distribution)"'

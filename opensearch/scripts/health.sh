#!/usr/bin/env bash
# Curl cluster + index health for a NeuroMem OpenSearch cluster.
set -euo pipefail

: "${OPENSEARCH_URL:=http://192.168.1.208:9200}"

auth_args=()
if [[ -n "${OPENSEARCH_USER:-}" && -n "${OPENSEARCH_PASSWORD:-}" ]]; then
    auth_args=(-u "${OPENSEARCH_USER}:${OPENSEARCH_PASSWORD}")
fi

echo "=== cluster health @ ${OPENSEARCH_URL} ==="
curl -fsS "${auth_args[@]}" "${OPENSEARCH_URL}/_cluster/health?pretty"

echo
echo "=== neuromem-* indices ==="
curl -fsS "${auth_args[@]}" "${OPENSEARCH_URL}/_cat/indices/neuromem-*?v&h=health,status,index,docs.count,store.size" || \
    echo "(no neuromem-* indexes yet)"

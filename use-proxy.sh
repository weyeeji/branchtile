#!/usr/bin/env bash
# 使用 SSH 反向隧道上的本地代理。
# 隧道：本机 127.0.0.1:17893  ->  发起 SSH 的机器 127.0.0.1:7897
#
# 当前 shell 启用（必须 source）：
#   source ./use-proxy.sh
#   source ./use-proxy.sh off
#
# 只让一条命令走代理：
#   ./use-proxy.sh curl -I https://example.com
#   ./use-proxy.sh test
#   ./use-proxy.sh status

PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
PROXY_PORT="${PROXY_PORT:-17893}"
HTTP_PROXY_URL="http://${PROXY_HOST}:${PROXY_PORT}"
SOCKS_PROXY_URL="socks5h://${PROXY_HOST}:${PROXY_PORT}"
NO_PROXY_LIST="${NO_PROXY_LIST:-localhost,127.0.0.1,::1}"

_proxy_exports() {
  cat <<EOF
export http_proxy=${HTTP_PROXY_URL}
export https_proxy=${HTTP_PROXY_URL}
export HTTP_PROXY=${HTTP_PROXY_URL}
export HTTPS_PROXY=${HTTP_PROXY_URL}
export all_proxy=${SOCKS_PROXY_URL}
export ALL_PROXY=${SOCKS_PROXY_URL}
export no_proxy=${NO_PROXY_LIST}
export NO_PROXY=${NO_PROXY_LIST}
EOF
}

_proxy_unsets() {
  cat <<'EOF'
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
EOF
}

_proxy_apply() {
  export http_proxy="${HTTP_PROXY_URL}"
  export https_proxy="${HTTP_PROXY_URL}"
  export HTTP_PROXY="${HTTP_PROXY_URL}"
  export HTTPS_PROXY="${HTTP_PROXY_URL}"
  export all_proxy="${SOCKS_PROXY_URL}"
  export ALL_PROXY="${SOCKS_PROXY_URL}"
  export no_proxy="${NO_PROXY_LIST}"
  export NO_PROXY="${NO_PROXY_LIST}"
}

_proxy_clear() {
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
}

_proxy_status() {
  echo "代理地址: ${HTTP_PROXY_URL}"
  echo "SOCKS:    ${SOCKS_PROXY_URL}"
  if [[ -n "${http_proxy:-}${https_proxy:-}${all_proxy:-}" ]]; then
    echo "当前 shell: 已启用"
    echo "  http_proxy=${http_proxy:-}"
    echo "  https_proxy=${https_proxy:-}"
    echo "  all_proxy=${all_proxy:-}"
  else
    echo "当前 shell: 未启用"
  fi
  if timeout 1 bash -c "echo >/dev/tcp/${PROXY_HOST}/${PROXY_PORT}" 2>/dev/null; then
    echo "端口 ${PROXY_HOST}:${PROXY_PORT}: 可连接"
  else
    echo "端口 ${PROXY_HOST}:${PROXY_PORT}: 不可连接（先确认 SSH 反向隧道还在）"
    return 1
  fi
}

_proxy_test() {
  local code
  if ! timeout 1 bash -c "echo >/dev/tcp/${PROXY_HOST}/${PROXY_PORT}" 2>/dev/null; then
    echo "端口 ${PROXY_HOST}:${PROXY_PORT} 不可连接。请先保持这条 SSH 隧道：" >&2
    echo "  ssh -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -R 127.0.0.1:${PROXY_PORT}:127.0.0.1:7897 <user>@<host>" >&2
    return 1
  fi
  code="$(curl -sS -m 15 -o /dev/null -w '%{http_code}' -x "${HTTP_PROXY_URL}" https://www.gstatic.com/generate_204)" || {
    echo "HTTP 代理请求失败" >&2
    return 1
  }
  echo "HTTP 代理可用，探测状态码: ${code}"
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  case "${1:-on}" in
    off|unset|disable)
      _proxy_clear
      echo "已关闭当前 shell 的代理"
      ;;
    status)
      _proxy_status
      ;;
    test)
      _proxy_apply
      _proxy_test
      ;;
    on|enable|"")
      _proxy_apply
      echo "已启用代理: ${HTTP_PROXY_URL}"
      ;;
    *)
      echo "source 用法: source $0 [on|off|status|test]" >&2
      return 1
      ;;
  esac
  return 0
fi

set -euo pipefail

case "${1:-}" in
  ""|help|-h|--help)
    sed -n '2,12p' "$0"
    ;;
  on|enable)
    _proxy_exports
    ;;
  off|unset|disable)
    _proxy_unsets
    ;;
  status)
    _proxy_status
    ;;
  test)
    _proxy_test
    ;;
  *)
    _proxy_apply
    exec "$@"
    ;;
esac

# GPU Docker GitHub proxy runbook

This note records the verified recovery used from the GPU Docker workspace on
2026-09-15. It intentionally contains no proxy address, user name, or password.

## Verified outcome

The authenticated enterprise proxy already provisioned through
`/workspace/hys/.bashrc` successfully fetched
`test/ddp-long-convergence-v2` through commit `1dc4154`. The precision exporter
then completed with 9/9 first-step topology results and 8/8 long-run topology
results passing.

## Root cause and durable lesson

The working proxy was not an unauthenticated `http://IP:port` value. Its
credentials and endpoint were carried by `ftp_proxy` after sourcing the Docker
workspace Bash configuration. Hard-coding a remembered IP lost authentication
and also allowed several Git configuration files to disagree.

At the time of diagnosis, proxy values existed in all of these scopes:

- the Docker workspace Git configuration;
- the login user's Git configuration;
- the repository-local Git configuration;
- shell environment variables.

A successful local `git switch` is not evidence of network access: Git can
switch to a stale remote-tracking ref even when `fetch` failed.

## One-shot authenticated recovery

Run the block from an interactive Bash shell. It obtains credentials from the
existing protected environment and does not print them.

```bash
source /workspace/hys/.bashrc >/dev/null 2>&1 || true

if [ -z "${ftp_proxy:-}" ]; then
  echo "ERROR: ftp_proxy is not provisioned"
else
  export http_proxy="http://${ftp_proxy#ftp://}"
  export https_proxy="$http_proxy"
  export all_proxy="$http_proxy"
  export HTTP_PROXY="$http_proxy"
  export HTTPS_PROXY="$https_proxy"
  export ALL_PROXY="$all_proxy"

  cd /workspace/hys/torchtitan-test

  git -c http.proxy="$https_proxy" \
    -c remote.origin.proxy="$https_proxy" \
    -c http.sslVerify=false \
    fetch origin test/ddp-long-convergence-v2
fi
```

`http.sslVerify=false` reproduces the verified command but weakens TLS
verification. Keep it command-scoped, never make it a persistent Git setting,
and remove it after the enterprise proxy CA is trusted by the container.

## Safe diagnostics

Show configuration origins before changing anything:

```bash
git config --show-origin --show-scope --get-regexp proxy
```

Show the proxy endpoint without exposing credentials:

```bash
printf '%s\n' "$https_proxy" | sed -E 's#(https?://)[^@/]+@#\1***@#'
```

Interpret common failures as follows:

- `Failed to connect to <proxy> port <port>`: Git never reached GitHub; the
  proxy endpoint or route is unavailable.
- HTTP 407: the selected proxy is reachable but authentication is absent or
  rejected.
- `git switch` succeeds after `fetch` fails: the branch came from a cached ref;
  verify the fetched commit before running new code.

## Security rules

- Never paste or commit the expanded values of `ftp_proxy`, `http_proxy`, or
  `https_proxy` when they contain credentials.
- Prefer environment-provisioned credentials over duplicating secrets in a new
  script.
- Mask credentials in diagnostic output and rotate any credential that was
  accidentally pasted into logs or chat.

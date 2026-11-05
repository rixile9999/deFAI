# Resilient peer learning over Java UDT

This mode runs independent learners and Java UDP inboxes without a parameter
server. It is a **research implementation** of static-graph, equal-peer binary
logistic learning. The HTTP reference is not used and is not a fallback path.

## Start a complete local job

Install Python 3.11/3.12, JDK 17+ on PATH, then:

```sh
pip install ".[transport]"
defai p2p udt-demo --peers 3 --rounds 3 --topology ring --output results/udt-job-001
```

The command builds the pinned Java adapter, creates reproducible synthetic local
data, a private group key and a common graph, then starts independent learners.
The launcher never computes an aggregate. Each peer trains on its own data and
mixes same-round neighbor models with Metropolis weights. `full` and `star` are
also supported. Existing standalone UDT and HTTP commands remain available.

`demo.json` summarizes the completed job. Per-peer files include:

| File/directory | Purpose |
|---|---|
| `checkpoint.json` | Signed, atomically replaced completed-round state |
| `outbox/` | Immutable encrypted messages and verified delivery receipts |
| `inbox/` | Java UDT partial/committed payloads, independently resumable |
| `result.json` | Status, model/history, and counters for this process |
| `receiver.log` | Java service lifecycle/session errors |
| `process.json` | Actual Python worker PID for local process supervision |
| `worker.lock` | OS-owned exclusion lock, released automatically on process death |

The model has a bias term and one full-batch local gradient step per round.
Objectives are equally weighted per peer, not per observation. Local losses and
model agreement are functional diagnostics, not clinical outcome validation.

## Run participants separately and recover

The generated job contains `peers.json`, `transport.jar`, `group.key` and one
local numeric NPZ dataset per peer. For example:

```sh
defai p2p udt-peer peers.json --peer peer-0 --data peer-0.npz \
  --output peer-0-state --jar transport.jar --key-file group.key \
  --round-timeout 120 --attempt-timeout 5

# Restart the same participant with the same job, dataset, key, and workspace:
defai p2p udt-peer peers.json --peer peer-0 --data peer-0.npz \
  --output peer-0-state --jar transport.jar --key-file group.key \
  --round-timeout 120 --attempt-timeout 5 --resume
```

A finished round is checkpointed before the next local step. A completed local
step is saved as a pending model before transmission and reused after restart.
The exact encrypted message is saved before sending. A crash after delivery but before saving its
receipt causes an idempotent resend; it does not repeat a committed model update.
Partial files resume at verified chunk boundaries. Resume rejects changed
datasets, job/graph/model settings, group keys, invalid checkpoints and models.
An existing workspace requires `--resume`; it is never silently reset.

Outgoing edges retry independently with bounded backoff. A disconnected neighbor
does not starve delivery to other neighbors. If a required neighbor remains
missing beyond the round budget, the process reports `paused` and exits nonzero;
restart all paused participants with `--resume` after restoring connectivity.
No missing neighbor is silently excluded, reweighted or replaced by an old model.
This preserves the stated learning algorithm instead of claiming arbitrary
permanent dropout tolerance.

The Java inbox accepts concurrent connections. Its supervisor restarts a failed
Java process. A per-process heartbeat lease lets orphaned Java inboxes stop after
about five seconds when their Python owner dies, so a resumed peer can rebind.
OS process cleanup and JVM startup add a small allowance beyond the round budget.

Choose attempt/round budgets for the expected link. Attempt timeout is at least
3 seconds and round timeout at least attempt timeout + 2 seconds. A larger budget
is required for high latency, long disconnections or large file transfers.
There is no throughput superiority claim over modern TCP/QUIC.

After completion, the inbox lingers for twice the attempt timeout by default to
answer neighbors retrying final receipts. `--linger SECONDS` changes that interval.
The completed model/checkpoint is durable before this interval. Final receipt
loss can leave a neighbor paused even if a model completed; relaunch that peer
and any completed peer it needs with `--resume` to serve the persisted state.

## Directly reachable LAN participants

Use the same graph/configuration and Java artifact on every participant, give
each only its own local data, and distribute the private group key out of band.
Change endpoints **before starting a new job**, for example:

```json
{
  "peer-0": {"endpoint":"udt://10.20.0.11:19000", "bind":"0.0.0.0", "bind_port":19000, "neighbors":["peer-1"]},
  "peer-1": {"endpoint":"udt://10.20.0.12:19000", "bind":"0.0.0.0", "bind_port":19000, "neighbors":["peer-0"]}
}
```

This is the `peers` field, not a complete job manifest. Preserve the generated
format, feature/round settings, key fingerprint and Java artifact fingerprint.
Use `--allow-network` on each trusted-lab worker. An endpoint is the advertised
destination; `bind`/`bind_port` are the local listener, permitting configured UDP
port forwarding. Only numeric addresses are accepted. Participants must be
mutually reachable by UDP. No automatic discovery, relay, carrier-NAT traversal
or live address-book migration is included. Endpoint changes alter the job
identity; a paused job must retain its original configuration.

## Message protection and identity

Model/completion messages use AES-256-GCM through `cryptography`, with an
independently derived encryption key, fresh random 96-bit nonce per **new**
message, and authenticated context containing job, sender, recipient, kind and
round. Retries reuse the saved ciphertext; they do not re-encrypt an existing
transfer ID. Checkpoints and delivery receipts use HMAC-SHA256. The configured
key fingerprint and Java JAR hash are verified before starting a worker.

Decryption, context, HMAC, vector dimensions, finite values and model SHA-256
are checked before mixing. Unencrypted messages are rejected. All learning
payloads remain encrypted in Java inbox/outbox files. Local datasets and model
checkpoints remain local plaintext plus authentication; protect the workspace.

This uses a pre-shared **group** key, not individual participant certificates.
It provides no forward secrecy or defense against a malicious group member.
Raw UDT control headers are not encrypted/authenticated by this application
layer. This is not a denial-of-service-resistant public Internet service,
secure aggregation or differential privacy implementation. No GV-HE is used.

## Verification scope

`pytest tests/test_p2p_udt.py -q` exercises actual Python/Java processes and UDP:

- Ring/full/star peers and an independently computed gradient/mixing reference.
- Periodic data-packet loss, delay/jitter, reordering, duplicates, lost handshake
  replies and an eight-second total UDP blackout followed by recovery.
- High latency with 150..250 ms of configured delay per relay direction.
- Actual Python worker termination after a committed round and checkpoint resume.
- Java inbox termination and automatic receiver restart.
- Missing peer beyond the round deadline, paused state and later safe resume.
- Corrupted cipher/nonces, wrong keys/contexts, stale/misaddressed models, and
  changed resume identity/state rejected without resetting a checkpoint.

These tests use small synthetic datasets, two to four peers and static graphs.
They do not establish behavior for every loss pattern, resource limit, mobile
carrier, arbitrary IP migration, model family or institutional deployment.
Android DEX compilation is a Java build check. Android/iOS device execution,
background scheduling and battery measurements are separate validation work.

See [standalone transport/reference peer details](peer_transport_research.md)
and [AES-GCM primitive documentation](https://cryptography.io/en/latest/hazmat/primitives/aead/).

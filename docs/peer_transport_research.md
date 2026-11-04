# Separate UDT transport and decentralized learning backends

Both features are **research implementations, not production deployments**.
UDT and P2P are independently usable. Neither calls the other in this version.
No GV-HE code or unpublished cryptographic work is included.

## Java UDT transport

The pinned [UDT-Java implementation](https://github.com/JohannesBuchner/udt-java)
supplies actual UDP reliability/congestion control. We add durable application
transfer semantics, not a replacement UDP protocol. Original licenses, commit
identity, original file hashes and local modifications are retained under
`java/udt`. Local patches fix handshake retry identity/session reuse, interrupted
writes and sender shutdown. The RTT accessor supports bounded close draining.

Install deFAI and put JDK 17+ `java`/`javac` on PATH:

```sh
defai udt build --output defai-udt.jar
java -jar defai-udt.jar receive received 127.0.0.1 19000 60000
# A second terminal:
java -jar defai-udt.jar send model.npz experiment-001 round-3 127.0.0.1 19000 60000
```

The receiver handles one connection and exits. Start it again at the same store
and repeat the sender command to resume after a disconnect or process restart.
`READY PORT` is printed before accepting; `COMMITTED` is printed after the final
receipt. `resumed_bytes` distinguishes new, partial and already-completed work.
Choose a new transfer ID when model content or model version changes.

The application contract is `DFU1` (big-endian magic `0x44465531`), followed by
Java DataOutput UTF transfer ID/model version/SHA-256 strings, signed 64-bit
payload size and a 32-bit chunk size. Chunk size is 65,536 bytes; maximum file
size is 256 MiB. IDs use 1..64 ASCII letters/digits/underscore/hyphen; versions
use 1..128 such characters plus dot, starting with an alphanumeric character.
`DataOutput.writeUTF` uses Java modified UTF-8; allowed header values are ASCII.

The receiver replies with a signed 64-bit durable offset and a prefix SHA-256.
Each frame carries signed 64-bit offset, 32-bit length, raw 32-byte SHA-256 and
bytes. Only a verified chunk is fsynced and acknowledged with the next offset.
A signed 64-bit `-1` ends the stream; full-file verification precedes publication
as `payload.bin`, followed by the UTF `COMMITTED` receipt. Metadata is published
before payload writes; whole chunks survive process termination. A damaged
prefix, conflicting model, oversized/malformed chunk or damaged completed file
is rejected. Failed sessions retain incomplete work for retry. A missing receipt
is an unknown outcome: retry the same ID; do not assume the payload was absent.

Stream flush enqueues bytes; application receipts establish durable delivery.
A bounded drain after success keeps transport acknowledgements active. This
initial implementation acknowledges each chunk individually; it does not claim
optimal high-latency throughput. Deadline applies to connect, accept and transfer;
cancelled sessions stop their sender/receiver/UDP socket. OS/DNS resolution is
not a validated mobile connectivity manager. Use numeric lab addresses.

The adapter core uses Java 8 bytecode and standard Java I/O/network APIs. Its
`Transfer` stream methods and `Session` can be called from an Android worker
thread. Android dex compilation is a separate build check, not proof of device
runtime behavior. No Android foreground service, network-switch monitor or
background scheduler is implemented. iOS, power loss durability and battery
efficiency have not been tested.

UDT here provides **no authentication or encryption**. Default CLI use is
loopback. Non-loopback lab addresses require `--allow-insecure-network` at both
ends and must only carry approved test data in a trusted lab. No Internet/NAT
traversal or secure clinical model transport is established by these tests.

## Decentralized peer learning

```sh
defai p2p demo --peers 4 --rounds 20 --topology ring --output peer-demo
# Other supported topologies: full, star
```

The demo creates synthetic local datasets, a static graph and independent peer
processes. It starts no parameter/aggregation server. Each peer is its own HTTP
server and learner, reads only its own data, publishes an immutable post-local
model and fetches the same-round neighbor models. The demo launcher creates test
inputs and reads final reports; it never mixes training models.

The first model is binary logistic regression with a bias term and one full-batch
local gradient step per round. For connected reciprocal graphs, edge weight is
`1/(1+max(degree_i,degree_j))`; the self weight fills the row to one. This symmetric
Metropolis matrix preserves the network mean in pure mixing. It optimizes the
equal-peer average of local objectives, **not** sample-count-weighted FedAvg.
No universal non-IID convergence or superiority is claimed.

To run peers separately, use the generated `peers.json` as a template:

```sh
defai p2p peer peers.json --peer peer-0 --data peer-0.npz --output peer-0-run --timeout 30
```

Local NPZ files must be trusted numeric files with finite `x[N,features]` and
binary `y[N]`. Each peer independently validates the static configuration.
The current HTTP transport deliberately accepts loopback endpoints only.
Raw observations are never served. Snapshot messages contain format
`defai-peer-model-v1`, canonical configuration SHA-256, sender, exact round,
float64 parameter vector including bias and its little-endian byte SHA-256.
Wrong job, sender, round, dimensions, nonfinite values and changed hashes fail
closed. Hashes establish integrity, not authentication or privacy.

Delayed startup/round publication is retried within a round deadline. All listed
neighbors must respond; there is no silent reweighting or reused stale model.
Loss of a peer aborts the round and records failed status and only completed
history. Peers retain immutable snapshots until their neighbors finish and read
their completion markers. Dynamic membership, offline training resume,
asynchronous stale updates, Byzantine defenses and secure aggregation are not
implemented. Do not equate decentralization with model confidentiality.

`application_payload_bytes_received` measures successfully fetched model JSON,
not Ethernet/IP/HTTP headers, all retries or actual wire traffic. Synthetic local
loss and model spread are functional diagnostics, not clinical evaluations.

## Future combination boundary

UDT transfers opaque model bytes with durable transfer ID/model version. P2P
validates peer/job/round/model identity independently of how bytes arrive. Future
integration can serialize the immutable peer snapshot as UTF-8 and derive a
64-character transfer ID from job+sender+recipient+round; a transport adapter
would supply bytes to the existing snapshot validation/mixing path. No such
adapter is enabled now. The completed-result interchange format, if installed,
remains distinct from these live peer snapshots.

## Verification and references

`python -m pytest tests/test_udt.py tests/test_p2p.py -q` builds the real Java
implementation and runs real loopback UDP/HTTP processes. UDT tests include
periodic data-packet loss, delayed forwarding, a lost handshake reply, duplicate
delivery, process-kill/resume and deadline cleanup. Additional stream cases test
malformed frames and integrity failures. P2P tests cover independent matrix
invariants, an independent gradient reference, invalid/stale messages, peer
failure and multi-process ring/full/star learning.

- [UDT protocol draft](https://datatracker.ietf.org/doc/html/draft-gg-udt-03)
- [Decentralized parallel stochastic gradient descent, Lian et al. 2017](https://proceedings.neurips.cc/paper/2017/hash/f75526659f31040afeb61cb7133e4e6d-Abstract.html)
- [Time-varying Metropolis weights, Xiao/Boyd/Lall](https://stanford.edu/~boyd/papers/avg_metropolis.html)

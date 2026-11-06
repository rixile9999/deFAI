import hashlib
import hmac

from flwr.app import ConfigRecord

from fl_core.records import digest, to_arrays


class KeyStore:
    def __init__(self, secret="defai-sejong-security"):
        self.master = secret.encode()

    def _derive(self, label):
        return hmac.new(self.master, label.encode(), hashlib.sha256).digest()

    def mac_key(self, principal):
        return self._derive(f"mac:{principal}")

    def device_secret(self, principal):
        return self._derive(f"device:{principal}")

    def certificate(self, principal):
        return hmac.new(self._derive("ca"), principal.encode(), hashlib.sha256).hexdigest()

    def device_id(self, principal):
        return hashlib.sha256(self.device_secret(principal)).hexdigest()[:16]


def otp(secret, rnd):
    code = hmac.new(secret, str(int(rnd)).encode(), hashlib.sha256).hexdigest()
    return str(int(code[:8], 16) % 1_000_000).zfill(6)


def payload_digest(arrays_record, rnd):
    return f"{digest(to_arrays(arrays_record))}:{int(rnd)}"


def node_principal(node_id):
    return f"node{int(node_id)}"


def reported_pid(reply):
    auth = reply.content.get("auth") if reply.has_content() else None
    if auth is not None and "pid" in auth:
        return int(auth["pid"])
    return int(reply.metadata.src_node_id) - 1


def sign(content, principal, rnd, keys, mac_key=None, device_secret=None, device_id=None, pid=None):
    mac = hmac.new(mac_key or keys.mac_key(principal), payload_digest(content["arrays"], rnd).encode(),
                   hashlib.sha256).hexdigest()
    record = {
        "principal": principal,
        "cert": keys.certificate(principal),
        "mac": mac,
        "otp": otp(device_secret or keys.device_secret(principal), rnd),
        "device": device_id or keys.device_id(principal),
    }
    if pid is not None:
        record["pid"] = int(pid)
    content["auth"] = ConfigRecord(record)
    return content


def verify(content, principal, rnd, keys, checks):
    auth = content.get("auth")
    problems = []
    if auth is None:
        return ["missing credentials"] if checks else []
    if "cert" in checks and str(auth.get("cert")) != keys.certificate(principal):
        problems.append("invalid certificate")
    if "hmac" in checks:
        expected = hmac.new(keys.mac_key(principal), payload_digest(content["arrays"], rnd).encode(),
                            hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, str(auth.get("mac"))):
            problems.append("HMAC-SHA256 mismatch (payload modified in transit)")
    if "mfa" in checks and str(auth.get("otp")) != otp(keys.device_secret(principal), rnd):
        problems.append("second factor rejected")
    return problems

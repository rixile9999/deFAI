import struct

from flwr.common.serde import message_from_proto, message_to_proto
from flwr.proto.message_pb2 import Message as ProtoMessage

CONTENT_TYPE = "application/x-flower-messages"


def dumps(message):
    return message_to_proto(message).SerializeToString()


def loads(data):
    proto = ProtoMessage()
    proto.ParseFromString(data)
    return message_from_proto(proto)


def pack(messages):
    out = bytearray()
    for m in messages:
        b = dumps(m)
        out += struct.pack(">Q", len(b))
        out += b
    return bytes(out)


def unpack(data):
    out, pos = [], 0
    while pos < len(data):
        (n,) = struct.unpack(">Q", data[pos:pos + 8])
        pos += 8
        out.append(loads(data[pos:pos + n]))
        pos += n
    return out

import itertools
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from flwr.app import Error, Message, MessageType, RecordDict
from flwr.common.constant import ErrorCode
from flwr.serverapp import Grid


class BaseGrid(Grid):
    def __init__(self):
        self._run = None
        self._pending = {}

    def set_run(self, run):
        self._run = run

    @property
    def run(self):
        return self._run

    def create_message(self, content, message_type, dst_node_id, group_id, ttl=None):
        return Message(content=content, dst_node_id=dst_node_id, message_type=message_type, group_id=group_id,
                       ttl=ttl)

    def push_messages(self, messages):
        ids = []
        for msg in messages:
            mid = uuid.uuid4().hex
            self._pending[mid] = msg
            ids.append(mid)
        return ids

    def pull_messages(self, message_ids):
        msgs = [self._pending.pop(m) for m in message_ids if m in self._pending]
        return self.send_and_receive(msgs)

    def send_and_receive(self, messages, *, timeout=None):
        raise NotImplementedError


def error_reply(msg, reason, code=ErrorCode.NODE_UNAVAILABLE):
    return Message(Error(code=code, reason=reason), reply_to=msg)


class InProcessGrid(BaseGrid):
    def __init__(self, nodes, workers=1, middleware=()):
        super().__init__()
        self.nodes = dict(nodes)
        self.workers = workers
        self.middleware = list(middleware)
        self.active = None
        self.log = []

    def get_node_ids(self):
        ids = sorted(self.nodes)
        if self.active is not None:
            ids = [n for n in ids if n in self.active]
        return ids

    def _dispatch(self, msg):
        node = self.nodes.get(msg.metadata.dst_node_id)
        if node is None:
            return error_reply(msg, "unknown node")
        start = time.perf_counter()
        try:
            for mw in self.middleware:
                early = mw.before(msg, node)
                if early is not None:
                    return early
            handler = node.train if msg.metadata.message_type == MessageType.TRAIN else node.evaluate
            reply = handler(msg, node.context)
            for mw in reversed(self.middleware):
                reply = mw.after(msg, reply, node)
        except Exception as ex:
            reply = error_reply(msg, f"{type(ex).__name__}: {ex}", ErrorCode.CLIENT_APP_RAISED_EXCEPTION)
        self.log.append({"node": msg.metadata.dst_node_id, "type": msg.metadata.message_type,
                         "sec": time.perf_counter() - start, "error": reply.has_error()})
        return reply

    def send_and_receive(self, messages, *, timeout=None):
        messages = list(messages)
        if self.workers <= 1 or len(messages) <= 1:
            return [self._dispatch(m) for m in messages]
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(self._dispatch, messages))


class Node:
    _ids = itertools.count(1)

    def __init__(self, node_id, context, train, evaluate):
        self.node_id = node_id
        self.context = context
        self.train = train
        self.evaluate = evaluate


def empty_content():
    return RecordDict()

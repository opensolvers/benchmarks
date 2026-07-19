#!/usr/bin/env python3
"""Dependency-free ONNX patcher: add accuracy_level=4 to every MatMulNBits node.

No numpy / onnx / protobuf needed. Implements just enough protobuf wire-format
parsing to walk ModelProto -> GraphProto -> repeated NodeProto, and rewrite each
MatMulNBits NodeProto by appending an AttributeProto {name:"accuracy_level",
type:INT, i:4}. All enclosing length prefixes are recomputed by rebuilding the
message tree bottom-up, so nested lengths stay consistent.

ONNX field numbers used:
  ModelProto.graph        = 7  (message)
  GraphProto.node         = 1  (repeated message NodeProto)
  NodeProto.op_type       = 4  (string)
  NodeProto.attribute     = 5  (repeated message AttributeProto)
  AttributeProto.name     = 1  (string)
  AttributeProto.i        = 3  (int64)
  AttributeProto.type     = 20 (enum; INT = 2)
"""
import sys

# ---- varint ----
def read_varint(buf, pos):
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos

def write_varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)

def tag(field, wire):
    return write_varint((field << 3) | wire)

def ld(field, payload):
    """length-delimited field encoding (wire type 2)."""
    return tag(field, 2) + write_varint(len(payload)) + payload

# ---- generic field walker: returns list of (field_no, wire_type, value_bytes_or_int, raw_start, raw_end) ----
def iter_fields(buf):
    pos = 0
    n = len(buf)
    while pos < n:
        start = pos
        key, pos = read_varint(buf, pos)
        field = key >> 3
        wire = key & 7
        if wire == 0:      # varint
            val, pos = read_varint(buf, pos)
            yield field, wire, val, start, pos
        elif wire == 2:    # length-delimited
            ln, pos = read_varint(buf, pos)
            val = buf[pos:pos+ln]
            pos += ln
            yield field, wire, val, start, pos
        elif wire == 5:    # 32-bit
            val = buf[pos:pos+4]; pos += 4
            yield field, wire, val, start, pos
        elif wire == 1:    # 64-bit
            val = buf[pos:pos+8]; pos += 8
            yield field, wire, val, start, pos
        else:
            raise ValueError(f"unsupported wire type {wire} at {start}")

def build_accuracy_level_attr():
    # AttributeProto { name="accuracy_level"(1,str), type=INT=2 (20,varint), i=4 (3,varint) }
    body = bytearray()
    body += ld(1, b"accuracy_level")
    body += tag(3, 0) + write_varint(4)    # i = 4
    body += tag(20, 0) + write_varint(2)   # type = INT (2)
    return bytes(body)

def patch_node(node_bytes):
    """If op_type == MatMulNBits and no accuracy_level attr, append one. Return new node bytes + flag."""
    op_type = None
    has_attr = False
    for field, wire, val, s, e in iter_fields(node_bytes):
        if field == 4 and wire == 2:
            op_type = bytes(val)
        elif field == 5 and wire == 2:
            # AttributeProto; check its name field(1)
            for f2, w2, v2, s2, e2 in iter_fields(val):
                if f2 == 1 and w2 == 2 and bytes(v2) == b"accuracy_level":
                    has_attr = True
    if op_type != b"MatMulNBits":
        return node_bytes, False
    if has_attr:
        return node_bytes, False
    # append a new attribute(5) submessage. Repeated fields concatenate; appending
    # at end of the NodeProto is valid protobuf.
    attr = build_accuracy_level_attr()
    return bytes(node_bytes) + ld(5, attr), True

def patch_graph(graph_bytes):
    """Rebuild GraphProto, patching each node(1) submessage."""
    out = bytearray()
    patched = 0
    for field, wire, val, s, e in iter_fields(graph_bytes):
        if field == 1 and wire == 2:
            new_node, did = patch_node(val)
            if did:
                patched += 1
            out += ld(1, new_node)
        else:
            # re-emit original raw bytes for this field verbatim
            out += graph_bytes[s:e]
    return bytes(out), patched

def patch_model(model_bytes):
    out = bytearray()
    total_patched = 0
    found_graph = False
    for field, wire, val, s, e in iter_fields(model_bytes):
        if field == 7 and wire == 2:   # graph
            found_graph = True
            new_graph, p = patch_graph(val)
            total_patched += p
            out += ld(7, new_graph)
        else:
            out += model_bytes[s:e]
    assert found_graph, "no graph field(7) found in ModelProto"
    return bytes(out), total_patched

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    data = open(src, "rb").read()
    print(f"input {len(data)} bytes, MatMulNBits count(raw)={data.count(b'MatMulNBits')}, "
          f"accuracy_level(raw)={data.count(b'accuracy_level')}")
    new, patched = patch_model(data)
    open(dst, "wb").write(new)
    verify = open(dst, "rb").read()
    print(f"patched {patched} nodes")
    print(f"output {len(verify)} bytes, MatMulNBits count(raw)={verify.count(b'MatMulNBits')}, "
          f"accuracy_level(raw)={verify.count(b'accuracy_level')}")
    # structural re-parse sanity: walk model->graph->nodes and count MatMulNBits with attr
    ok_nodes = 0
    for f, w, v, s, e in iter_fields(verify):
        if f == 7 and w == 2:
            for f1, w1, gval, gs, ge in iter_fields(v):
                if f1 == 1 and w1 == 2:
                    ot = None; hasa = False
                    for f2, w2, nval, ns, ne in iter_fields(gval):
                        if f2 == 4 and w2 == 2 and bytes(nval) == b"MatMulNBits":
                            ot = True
                        if f2 == 5 and w2 == 2:
                            for f3, w3, av, as_, ae in iter_fields(nval):
                                if f3 == 1 and w3 == 2 and bytes(av) == b"accuracy_level":
                                    hasa = True
                    if ot and hasa:
                        ok_nodes += 1
    print(f"structural re-parse: MatMulNBits nodes WITH accuracy_level = {ok_nodes}")

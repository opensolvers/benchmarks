#!/usr/bin/env python3
"""Patch MatMulNBits nodes to accuracy_level=4 (CompInt8 IME path).

Handles nodes with missing accuracy_level OR accuracy_level=0 (AMD Qwen export).
"""
import sys

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
    return tag(field, 2) + write_varint(len(payload)) + payload

def iter_fields(buf):
    pos = 0
    n = len(buf)
    while pos < n:
        start = pos
        key, pos = read_varint(buf, pos)
        field = key >> 3
        wire = key & 7
        if wire == 0:
            val, pos = read_varint(buf, pos)
            yield field, wire, val, start, pos
        elif wire == 2:
            ln, pos = read_varint(buf, pos)
            val = buf[pos:pos+ln]
            pos += ln
            yield field, wire, val, start, pos
        elif wire == 5:
            val = buf[pos:pos+4]
            pos += 4
            yield field, wire, val, start, pos
        elif wire == 1:
            val = buf[pos:pos+8]
            pos += 8
            yield field, wire, val, start, pos
        else:
            raise ValueError(f"unsupported wire type {wire} at {start}")

def build_accuracy_level_attr(value=4):
    body = bytearray()
    body += ld(1, b"accuracy_level")
    body += tag(3, 0) + write_varint(value)
    body += tag(20, 0) + write_varint(2)
    return bytes(body)

def patch_attr_to_acc4(attr_bytes):
    """Return (new_attr_bytes, changed) setting i=4 for accuracy_level."""
    name = None
    i_val = None
    fields = list(iter_fields(attr_bytes))
    for f, w, v, s, e in fields:
        if f == 1 and w == 2:
            name = bytes(v)
        if f == 3 and w == 0:
            i_val = v
    if name != b"accuracy_level":
        return attr_bytes, False
    if i_val == 4:
        return attr_bytes, False
    out = bytearray()
    out += ld(1, b"accuracy_level")
    out += tag(3, 0) + write_varint(4)
    out += tag(20, 0) + write_varint(2)
    return bytes(out), True

def patch_node(node_bytes):
    op_type = None
    acc_val = None
    for field, wire, val, s, e in iter_fields(node_bytes):
        if field == 4 and wire == 2:
            op_type = bytes(val)
        elif field == 5 and wire == 2:
            for f2, w2, v2, s2, e2 in iter_fields(val):
                if f2 == 1 and w2 == 2 and bytes(v2) == b"accuracy_level":
                    for f3, w3, v3, s3, e3 in iter_fields(val):
                        if f3 == 3 and w3 == 0:
                            acc_val = v3
    if op_type != b"MatMulNBits":
        return node_bytes, False
    if acc_val == 4:
        return node_bytes, False

    out = bytearray()
    changed = False
    for field, wire, val, s, e in iter_fields(node_bytes):
        if field == 5 and wire == 2:
            new_attr, did = patch_attr_to_acc4(val)
            if did:
                out += ld(5, new_attr)
                changed = True
            else:
                # check if this is accuracy_level with wrong val handled above
                is_acc = any(
                    f2 == 1 and w2 == 2 and bytes(v2) == b"accuracy_level"
                    for f2, w2, v2, s2, e2 in iter_fields(val)
                )
                if is_acc and acc_val is not None and acc_val != 4:
                    out += ld(5, build_accuracy_level_attr(4))
                    changed = True
                else:
                    out += node_bytes[s:e]
        else:
            out += node_bytes[s:e]

    if not changed and acc_val is None:
        out = bytearray(node_bytes)
        out += ld(5, build_accuracy_level_attr(4))
        changed = True
    return bytes(out), changed

def patch_graph(graph_bytes):
    out = bytearray()
    patched = 0
    for field, wire, val, s, e in iter_fields(graph_bytes):
        if field == 1 and wire == 2:
            new_node, did = patch_node(val)
            if did:
                patched += 1
            out += ld(1, new_node)
        else:
            out += graph_bytes[s:e]
    return bytes(out), patched

def patch_model(model_bytes):
    out = bytearray()
    total_patched = 0
    found_graph = False
    for field, wire, val, s, e in iter_fields(model_bytes):
        if field == 7 and wire == 2:
            found_graph = True
            new_graph, p = patch_graph(val)
            total_patched += p
            out += ld(7, new_graph)
        else:
            out += model_bytes[s:e]
    assert found_graph, "no graph field(7) found in ModelProto"
    return bytes(out), total_patched

def count_acc_vals(model_bytes):
    vals = {}
    for f, w, v, s, e in iter_fields(model_bytes):
        if f == 7 and w == 2:
            for f1, w1, gval, gs, ge in iter_fields(v):
                if f1 == 1 and w1 == 2:
                    ot = False
                    for f2, w2, nval, ns, ne in iter_fields(gval):
                        if f2 == 4 and w2 == 2 and bytes(nval) == b"MatMulNBits":
                            ot = True
                        if f2 == 5 and w2 == 2 and ot:
                            for f3, w3, av, as_, ae in iter_fields(nval):
                                if f3 == 1 and w3 == 2 and bytes(av) == b"accuracy_level":
                                    for f4, w4, iv, is_, ie in iter_fields(nval):
                                        if f4 == 3 and w4 == 0:
                                            vals[iv] = vals.get(iv, 0) + 1
    return vals

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    data = open(src, "rb").read()
    print(f"input {len(data)} bytes, MatMulNBits(raw)={data.count(b'MatMulNBits')}")
    print(f"accuracy_level before: {count_acc_vals(data)}")
    new, patched = patch_model(data)
    open(dst, "wb").write(new)
    verify = open(dst, "rb").read()
    print(f"patched {patched} nodes -> {dst}")
    print(f"accuracy_level after: {count_acc_vals(verify)}")

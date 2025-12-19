#!/usr/bin/env python3
"""Convert a Cerebus .cmp map file to a Trellis-style .map file.

Lightweight implementation without external dependencies.
"""
from pathlib import Path
import sys

BANK_OFFSETS = {
    'A': 0,
    'B': 32,
    'C': 64,
    'D': 96,
}

PORT_FE_BY_BANK = {
    'A': ('A', '1'),
    'B': ('A', '2'),
    'C': ('A', '3'),
    'D': ('B', '1'),
}


# array name by bank
# Sulley uses A/D for M1 and B/C for PMd
# ARRAY_BY_BANK = {
#     'A': 'M1',
#     'B': 'PMd',
#     'C': 'PMd',
#     'D': 'M1',
# }
# Prez uses A/D for PMd and B/C for M1
ARRAY_BY_BANK = {
    'A': 'PMd',
    'B': 'M1',
    'C': 'M1',
    'D': 'PMd',
}


def parse_cmp(path: Path):
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('//') or s.startswith('#'):
                continue
            rows.append(s)

    if not rows:
        return []

    # skip first non-comment description line
    data_lines = rows[1:]
    result = []
    for ln in data_lines:
        parts = ln.split()
        if len(parts) < 4:
            continue
        col = int(parts[0])
        row = int(parts[1])
        bank = parts[2]
        elec = int(parts[3])
        label = parts[4] if len(parts) >= 5 else ''
        result.append({'col': col, 'row': row, 'bank': bank, 'elec': elec, 'label': label})

    return result


def make_map_rows(records):
    out_rows = []
    for r in records:
        col = int(r['col'])
        row = int(r['row'])
        bank = r['bank']
        elec = int(r['elec'])

        if bank not in BANK_OFFSETS:
            raise ValueError(f"Unknown bank: {bank}")
        # electrode label uses array name and absolute channel numbering with bank offsets
        chan_abs = elec + BANK_OFFSETS[bank]
        array_name = ARRAY_BY_BANK.get(bank, 'elec2')
        elec_label = f"{array_name}.chan{chan_abs:03d}"

        # hardware address uses per-FE channel number (1-32)
        port, fe_slot = PORT_FE_BY_BANK[bank]
        hw_addr = f"1.{port}.{fe_slot}.{elec:03d}"

        # view coords: transpose and 1-index
        x = row + 1
        y = col + 1
        view = f"{x}.{y}"

        out_rows.append((hw_addr, elec_label, view))

    return out_rows


def write_map(out_path: Path, rows):
    with out_path.open('w', encoding='utf-8') as f:
        for hw, label, view in rows:
            f.write(f"{hw}; {label}; {view}\n")


def main(argv):
    if len(argv) < 3:
        print("Usage: cmp_to_map.py input.cmp output.map")
        return 2

    in_path = Path(argv[1])
    out_path = Path(argv[2])

    recs = parse_cmp(in_path)
    rows = make_map_rows(recs)
    write_map(out_path, rows)
    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))

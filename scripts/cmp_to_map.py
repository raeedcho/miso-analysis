#!/usr/bin/env python3
"""Convert a Cerebus .cmp map file to a Trellis-style .map file.

Usage: python scripts/cmp_to_map.py input.cmp output.map

Rules implemented:
- CMP file columns are 0-indexed; MAP file coordinates are 1-indexed.
- Transpose col/row into x.y where x = row+1 and y = col+1 (so CMP col 0,row 17 -> MAP 18.1)
- Banks A,B,C map to port A with FE slots 1,2,3 respectively. Bank D maps to port B FE slot 1.
- Electrode label output: elec2.chan<N> where N = elec + bank_offset (A:0, B:+32, C:+64, D:+96)
- Hardware address format: <port>.<fe_slot>.1.<channel_padded>
  (We use FEslot.channel with channel 1-32 and zero-padded to 3 digits as in example: 1.A.1.001)

This script is written to be small and dependency-light (uses pandas for convenience).
"""
import sys
from pathlib import Path
import pandas as pd

BANK_OFFSETS = {
    'A': 0,
    'B': 32,
    'C': 64,
    'D': 96,
}

# Mapping banks to (port, fe_slot)
BANK_PORT_FE = {
    'A': ('1', 'A', '1'),  # (unused NIP id placeholder, port, FE slot)
}

# As described: banks A-C -> port A, FE slots 1-3 respectively; D -> port B FE slot 1.
PORT_FE_BY_BANK = {
    'A': ('A', '1'),
    'B': ('A', '2'),
    'C': ('A', '3'),
    'D': ('B', '1'),
}


def parse_cmp(path: Path) -> pd.DataFrame:
    # Read CMP file ignoring comment lines starting with //
    lines = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('//') or s.startswith('#'):
                continue
            lines.append(s)

    # The first non-comment line is a description; skip it
    if not lines:
        return pd.DataFrame()
    header = lines[0]
    data_lines = lines[1:]

    # CMP columns are: col\trow\tbank\telec\tlabel
    rows = []
    for ln in data_lines:
        parts = ln.split()  # whitespace separated
        if len(parts) < 4:
            continue
        col = int(parts[0])
        row = int(parts[1])
        bank = parts[2]
        elec = int(parts[3])
        label = parts[4] if len(parts) >= 5 else ''
        rows.append({'col': col, 'row': row, 'bank': bank, 'elec': elec, 'label': label})

    return pd.DataFrame(rows)


def make_map_rows(df: pd.DataFrame):
    out_rows = []
    for _, r in df.iterrows():
        col = int(r['col'])
        row = int(r['row'])
        bank = r['bank']
        elec = int(r['elec'])

        # compute chan number offset
        if bank not in BANK_OFFSETS:
            raise ValueError(f"Unknown bank: {bank}")
        chan_num = elec + BANK_OFFSETS[bank]

        # electrode label like elec2.chan<N>
        elec_label = f"elec2.chan{chan_num}"

        # map to port and FE slot
        port, fe_slot = PORT_FE_BY_BANK[bank]

        # hardware address format: <NIP?>.<port>.<FEslot>.<channel_padded>
        # We'll omit NIP id; use port.FEslot.channel with channel zero-padded to 3 digits
        hw_channel = chan_num  # channel number 1-128 maybe; but example expects 1-32 per FE
        hw_addr = f"1.{port}.{fe_slot}.{hw_channel:03d}"

        # transpose and convert to 1-indexed view coords x.y
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

    df = parse_cmp(in_path)
    rows = make_map_rows(df)
    write_map(out_path, rows)
    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))

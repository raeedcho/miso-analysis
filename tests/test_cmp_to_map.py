import tempfile
from pathlib import Path
from src.cmp_to_map import parse_cmp, make_map_rows, write_map


def test_conversion_simple():
    sample = """
Cerebus mapping example
0	17	C	5	elec2-126
0	0	A	1	elec1-57
7	0	D	30	elec1-1
"""

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "sample.cmp"
        p.write_text(sample)

        df = parse_cmp(p)
        rows = make_map_rows(df)

        out = Path(td) / "out.map"
        write_map(out, rows)

        content = out.read_text().strip().splitlines()
        # Expect 3 lines
        assert len(content) == 3

    # Check a known mapping: CMP col0,row17,bankC,elec5 -> absolute chan = 5+64=69
    # hardware address uses per-FE channel number (elec), so expected hw addr: 1.A.3.005
    assert any('1.A.3.005; elec2.chan69; 18.1' in line for line in content)

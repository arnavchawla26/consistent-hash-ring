import subprocess
import sys

import pytest

from chring.cli import build_parser, main


def run(args):
    return subprocess.run(
        [sys.executable, "-m", "chring.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_parser_requires_subcommand():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_movement_via_main_returns_zero(capsys):
    rc = main(["movement", "--nodes", "5", "--keys", "1000", "--event", "add"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "consistent" in out
    assert "naive-mod-n" in out
    assert "Topology change" in out


def test_movement_remove_event(capsys):
    rc = main(["movement", "--nodes", "5", "--keys", "1000", "--event", "remove"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "remove a node" in out


def test_distribution_via_main(capsys):
    rc = main(["distribution", "--nodes", "4", "--keys", "2000", "--replicas", "50"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "coefficient_of_variation" in out
    assert "node-0" in out


def test_lookup_via_main(capsys):
    rc = main(["lookup", "mykey", "--node", "a", "--node", "b", "--replicate", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "mykey" in out
    assert ("'a'" in out) or ("'b'" in out)


def test_lookup_replicate_two_returns_two_distinct_nodes(capsys):
    rc = main(
        ["lookup", "mykey", "--node", "a", "--node", "b", "--node", "c", "--replicate", "2"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    # e.g. "key='mykey' -> ['b', 'c']"
    bracket = out[out.index("[") : out.index("]") + 1]
    items = eval(bracket)  # safe: our own controlled CLI output
    assert len(items) == 2
    assert len(set(items)) == 2


def test_cli_end_to_end_subprocess_smoke():
    result = run(["movement", "--nodes", "6", "--keys", "500"])
    assert result.returncode == 0
    assert "algorithm" in result.stdout


def test_cli_help_exits_zero():
    result = run(["--help"])
    assert result.returncode == 0
    assert "chring" in result.stdout

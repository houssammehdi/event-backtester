from __future__ import annotations

from pathlib import Path

import pytest

from backtester.cli import main


def test_run_prints_a_report(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "run",
            "--strategy",
            "tsmom",
            "--symbols",
            "3",
            "--years",
            "2",
            "--seed",
            "1",
            "-p",
            "lookback=63",
            "--trades",
            "3",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    for text in ("Strategy: tsmom", "lookback=63", "Sharpe ratio", "Buy & hold", "entry_time"):
        assert text in out


def test_walkforward_prints_windows_and_warning(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "walkforward",
            "--strategy",
            "sma",
            "--symbols",
            "2",
            "--years",
            "3",
            "--seed",
            "2",
            "--train-bars",
            "252",
            "--test-bars",
            "126",
            "--warmup",
            "100",
            "-g",
            "fast=5,10",
            "-g",
            "slow=40",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Stitched out-of-sample" in out
    assert "deflated Sharpe" in out
    assert "fast=" in out
    assert "IS PBO" in out
    assert "Out-of-sample uncertainty" in out
    assert "Min. track record" in out


@pytest.mark.parametrize("benchmark", ["cash", "buyhold"])
def test_validate_reports_overfitting_and_snooping(
    benchmark: str, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "validate",
            "--strategy",
            "sma",
            "--symbols",
            "3",
            "--years",
            "4",
            "--seed",
            "5",
            "-g",
            "fast=10,20",
            "-g",
            "slow=50,100",
            "--samples",
            "200",
            "--pbo-splits",
            "8",
            "--benchmark",
            benchmark,
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    for needle in (
        "4 configurations",
        "fast=10, slow=50",
        "Deflated Sharpe (4 trials)",
        "Probability of backtest overfitting",
        "Hansen SPA (consistent)",
        "Combinatorial purged CV",
        "within 100 bars",
    ):
        assert needle in out
    assert ("buy & hold" in out) == (benchmark == "buyhold")


def test_generate_then_run_from_csv(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    csv = tmp_path / "data.csv"
    assert main(["generate", "--out", str(csv), "--symbols", "2", "--years", "1"]) == 0
    assert csv.exists()
    assert main(["run", "--strategy", "bollinger", "--csv", str(csv)]) == 0
    assert "SYN" not in capsys.readouterr().err


def test_strategies_listing(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["strategies"]) == 0
    out = capsys.readouterr().out
    for name in ("sma", "tsmom", "xsmom", "bollinger"):
        assert name in out


def test_errors_exit_with_code_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--years", "1", "-p", "nonsense=1"]) == 2
    assert "unknown parameter" in capsys.readouterr().err
    assert main(["run", "--years", "1", "-p", "novalue"]) == 2
    assert main(["walkforward", "--years", "1", "-g", "bad=1"]) == 2
    assert main(["validate", "--years", "1", "-g", "bad=1"]) == 2


def test_plot_option_writes_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    out = tmp_path / "equity.png"
    assert main(["run", "--strategy", "sma", "--years", "2", "--plot", str(out)]) == 0
    assert out.stat().st_size > 10_000
    wf = tmp_path / "wf.png"
    args = [
        "walkforward",
        "-s",
        "sma",
        "--years",
        "3",
        "--train-bars",
        "252",
        "--test-bars",
        "252",
        "--warmup",
        "50",
        "-g",
        "fast=10",
        "-g",
        "slow=40",
        "--plot",
        str(wf),
    ]
    assert main(args) == 0
    assert wf.exists()


def test_run_allocation_strategy(capsys: pytest.CaptureFixture[str]) -> None:
    args = ["run", "-s", "allocation", "-p", "method=erc", "-p", "lookback=126", "--years", "2"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "Strategy: allocation" in out
    assert "method=erc" in out


def test_multi_asset_universe(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(["run", "-s", "allocation", "-p", "method=iv", "--universe", "multi", "--years", "2"])
        == 0
    )
    assert "14 symbols" in capsys.readouterr().out
    csv = tmp_path / "multi.csv"
    assert main(["generate", "--universe", "multi", "--years", "1", "--out", str(csv)]) == 0
    assert "for 14 symbols" in capsys.readouterr().out


def test_closed_stdout_exits_quietly() -> None:
    """Regression: `backtest ... | head` printed a BrokenPipeError traceback."""
    import subprocess
    import sys

    cmd = [sys.executable, "-m", "backtester.cli", "strategies"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    assert proc.stderr is not None
    proc.stdout.close()  # the reader goes away before the first write
    with proc.stderr:
        stderr = proc.stderr.read().decode()
    assert proc.wait() == 141  # 128 + SIGPIPE, like other command-line tools
    assert "Traceback" not in stderr

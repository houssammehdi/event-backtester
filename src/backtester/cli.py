"""Command-line interface: ``backtest run | walkforward | validate | generate | strategies``."""

from __future__ import annotations

import argparse
import ast
import inspect
import os
import signal
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from backtester.analytics.benchmark import buy_and_hold
from backtester.analytics.metrics import cagr, max_drawdown, sharpe_ratio, total_return
from backtester.analytics.report import num, pct, table
from backtester.config import BacktestConfig
from backtester.data.feed import DataFeed
from backtester.data.loaders import load_csv, save_csv
from backtester.data.synthetic import TRADING_DAYS, generate_multi_asset, generate_ohlcv
from backtester.errors import BacktesterError, ConfigError
from backtester.execution.commission import BpsCommission, CommissionModel, PerShareCommission
from backtester.execution.slippage import FixedBpsSlippage, SlippageModel, SquareRootImpactSlippage
from backtester.research.grid import config_label, grid_search
from backtester.research.parallel import resolve_jobs
from backtester.research.walkforward import walk_forward, walk_forward_windows
from backtester.risk import RiskLimits
from backtester.strategies import STRATEGIES
from backtester.strategy.rebalancing import TargetWeightStrategy
from backtester.validation.cv import CombinatorialPurgedCV
from backtester.validation.sharpe import sample_moments

DEFAULT_GRIDS: dict[str, dict[str, list[Any]]] = {
    "sma": {"fast": [20, 50], "slow": [100, 200]},
    "tsmom": {"lookback": [63, 126, 252], "target_vol": [0.10, 0.20]},
    "xsmom": {"lookback": [63, 126, 252], "top_k": [1, 2]},
    "bollinger": {"window": [10, 20, 40], "n_std": [1.5, 2.0, 2.5]},
    "allocation": {"method": ["ew", "iv", "minvar", "erc", "hrp"], "lookback": [126, 252]},
}


def _literal(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def _parse_params(items: Sequence[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ConfigError(f"expected key=value, got {item!r}")
        params[key.strip()] = _literal(value.strip())
    return params


def _parse_grid(items: Sequence[str]) -> dict[str, list[Any]]:
    grid: dict[str, list[Any]] = {}
    for key, value in _parse_params([i.replace(" ", "") for i in items]).items():
        values = value if isinstance(value, tuple | list) else [value]
        grid[key] = list(values)
    return grid


def _strategy_params(name: str) -> set[str]:
    sig = inspect.signature(STRATEGIES[name].__init__)
    return {p for p in sig.parameters if p != "self"}


def _make_strategy(name: str, params: dict[str, Any]) -> TargetWeightStrategy:
    unknown = set(params) - _strategy_params(name)
    if unknown:
        raise ConfigError(f"unknown parameter(s) for {name}: {sorted(unknown)}")
    return STRATEGIES[name](**params)


def _add_data_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("data")
    g.add_argument("--csv", type=Path, help="CSV file (long format) or directory of CSVs")
    g.add_argument(
        "--universe",
        choices=("single", "multi"),
        default="single",
        help="synthetic market: one market factor (default), or 6 equities, 4 bonds and "
        "4 commodities with independent factors (ignores --symbols)",
    )
    g.add_argument("--symbols", type=int, default=5, help="synthetic symbols (default 5)")
    g.add_argument("--years", type=float, default=10.0, help="synthetic years (default 10)")
    g.add_argument("--seed", type=int, default=42, help="synthetic data seed (default 42)")
    g.add_argument(
        "--missing-prob", type=float, default=0.0, help="drop synthetic bars at this rate"
    )


def _add_exec_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("execution & risk")
    g.add_argument("--cash", type=float, default=1_000_000.0, help="initial capital")
    g.add_argument("--slippage-bps", type=float, default=2.0, help="fixed slippage (bps)")
    g.add_argument(
        "--impact",
        type=float,
        metavar="ETA",
        help="use square-root impact with this coefficient instead of fixed slippage",
    )
    g.add_argument("--commission-bps", type=float, default=1.0, help="commission (bps)")
    g.add_argument(
        "--per-share",
        type=float,
        metavar="RATE",
        help="per-share commission (minimum 1.0) instead of bps",
    )
    g.add_argument(
        "--participation", type=float, default=0.1, help="max share of bar volume filled"
    )
    g.add_argument("--max-gross", type=float, default=2.0, help="max gross leverage")
    g.add_argument("--max-weight", type=float, help="max absolute position weight")
    g.add_argument("--max-drawdown", type=float, help="kill-switch drawdown (e.g. 0.3)")
    g.add_argument("--borrow-rate", type=float, default=0.0, help="annual short borrow fee")


def _strategy_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--strategy", "-s", choices=sorted(STRATEGIES), default="tsmom")


def _grid_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--grid",
        "-g",
        action="append",
        default=[],
        help="parameter grid key=v1,v2,... (repeatable; defaults per strategy)",
    )
    p.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="worker processes for the backtests (-1: one per CPU; default 1)",
    )


def _add_bootstrap_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("statistics")
    g.add_argument("--samples", type=int, default=2000, help="bootstrap resamples (default 2000)")
    g.add_argument(
        "--boot-seed", type=int, default=0, help="seed of the bootstrap resampling (default 0)"
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="backtest",
        description="Event-driven backtester. Runs offline on seeded synthetic data by default.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="backtest one strategy and print a report")
    _strategy_arg(run)
    run.add_argument(
        "--param", "-p", action="append", default=[], help="strategy parameter key=value"
    )
    run.add_argument("--plot", type=Path, help="save an equity/drawdown PNG here")
    run.add_argument("--report", type=Path, help="write a self-contained HTML tear sheet here")
    run.add_argument("--trades", type=int, default=0, help="also print the last N trades")
    _add_data_args(run)
    _add_exec_args(run)
    _add_bootstrap_args(run)

    wf = sub.add_parser("walkforward", help="walk-forward optimisation with stitched OOS")
    _strategy_arg(wf)
    _grid_arg(wf)
    wf.add_argument("--train-bars", type=int, default=3 * TRADING_DAYS, help="in-sample bars")
    wf.add_argument("--test-bars", type=int, default=TRADING_DAYS, help="out-of-sample bars")
    wf.add_argument("--warmup", type=int, default=TRADING_DAYS, help="bars before first fit")
    wf.add_argument("--gap", type=int, default=0, help="embargo bars between IS and OOS")
    wf.add_argument("--anchored", action="store_true", help="expanding in-sample window")
    wf.add_argument("--plot", type=Path, help="save the stitched OOS equity PNG here")
    _add_data_args(wf)
    _add_exec_args(wf)
    _add_bootstrap_args(wf)

    val = sub.add_parser(
        "validate",
        help="overfitting and data-snooping statistics of a parameter search",
        description=(
            "Backtest every configuration of a grid over the whole sample, then report "
            "the selected configuration's bootstrap intervals, PSR, MinTRL and deflated "
            "Sharpe; the probability of backtest overfitting (CSCV); White's Reality "
            "Check and Hansen's SPA against a benchmark; and the out-of-sample Sharpe "
            "ratios of combinatorial purged CV backtest paths."
        ),
    )
    _strategy_arg(val)
    _grid_arg(val)
    val.add_argument(
        "--benchmark",
        choices=("cash", "buyhold"),
        default="cash",
        help="SPA/RC benchmark: zero returns or equal-weight buy-and-hold (default cash)",
    )
    val.add_argument("--pbo-splits", type=int, default=16, help="CSCV blocks (even, default 16)")
    val.add_argument("--cpcv-groups", type=int, default=6, help="CPCV groups (default 6)")
    val.add_argument("--cpcv-test", type=int, default=2, help="CPCV test groups (default 2)")
    val.add_argument(
        "--embargo", type=int, default=0, help="CPCV embargo bars after each test block"
    )
    val.add_argument(
        "--report",
        type=Path,
        help="write the HTML tear sheet of the selected configuration (with its DSR) here",
    )
    _add_data_args(val)
    _add_exec_args(val)
    _add_bootstrap_args(val)

    gen = sub.add_parser("generate", help="write synthetic OHLCV data to CSV")
    gen.add_argument("--out", type=Path, required=True, help="output CSV path")
    _add_data_args(gen)

    sub.add_parser("strategies", help="list built-in strategies and their parameters")
    return parser


def _synthetic_frames(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    if args.universe == "multi":
        return generate_multi_asset(args.years, args.seed, missing_prob=args.missing_prob)
    return generate_ohlcv(args.symbols, args.years, args.seed, missing_prob=args.missing_prob)


def _load_feed(args: argparse.Namespace) -> DataFeed:
    if args.csv is not None:
        return DataFeed(load_csv(args.csv))
    return DataFeed(_synthetic_frames(args))


def _config(args: argparse.Namespace) -> BacktestConfig:
    slippage: SlippageModel = (
        SquareRootImpactSlippage(eta=args.impact, spread_bps=args.slippage_bps)
        if args.impact is not None
        else FixedBpsSlippage(args.slippage_bps)
    )
    commission: CommissionModel = (
        PerShareCommission(rate=args.per_share)
        if args.per_share is not None
        else BpsCommission(args.commission_bps)
    )
    return BacktestConfig(
        initial_cash=args.cash,
        slippage=slippage,
        commission=commission,
        max_participation=args.participation,
        limits=RiskLimits(
            max_position_weight=args.max_weight,
            max_gross_leverage=args.max_gross,
            max_drawdown=args.max_drawdown,
        ),
        borrow_rate=args.borrow_rate,
    )


def _report_notes(args: argparse.Namespace, cfg: BacktestConfig) -> list[tuple[str, str]]:
    """Header lines of a tear sheet: where the data came from and what trading cost."""
    if args.csv is not None:
        data = f"CSV: {args.csv}"
    elif args.universe == "multi":
        data = f"Synthetic multi-asset universe, {args.years:g} years, seed {args.seed}"
    else:
        data = f"Synthetic: {args.symbols} symbols, {args.years:g} years, seed {args.seed}"
    costs = [
        _describe_model(cfg.slippage, "slippage"),
        _describe_model(cfg.commission, "commission"),
    ]
    if cfg.max_participation is not None:
        costs.append(f"fills capped at {cfg.max_participation:.0%} of each bar's volume")
    return [("Data", data), ("Costs", "; ".join(costs))]


def _describe_model(model: SlippageModel | CommissionModel, kind: str) -> str:
    """Short description of a cost model for a report header."""
    if isinstance(model, FixedBpsSlippage | BpsCommission):
        return f"{model.bps:g} bps {kind}"
    if isinstance(model, SquareRootImpactSlippage):
        return f"square-root impact (eta {model.eta:g}) plus {model.spread_bps:g} bps {kind}"
    if isinstance(model, PerShareCommission):
        return f"{model.rate:g} per share {kind} (minimum {model.minimum:g})"
    return f"{kind}: {model}"


def _describe_costs(cfg: BacktestConfig) -> str:
    return (
        f"Execution: next-bar open / intrabar, slippage={cfg.slippage}, "
        f"commission={cfg.commission}, max participation={cfg.max_participation}"
    )


def _headless_plotting() -> None:
    """Select the non-interactive backend: the CLI only ever writes PNG files."""
    try:
        import matplotlib
    except ImportError as exc:
        raise ConfigError("--plot needs matplotlib: pip install 'event-backtester[plot]'") from exc
    matplotlib.use("Agg")


def _cmd_run(args: argparse.Namespace) -> int:
    if args.plot is not None:
        _headless_plotting()
    feed = _load_feed(args)
    cfg = _config(args)
    strategy = _make_strategy(args.strategy, _parse_params(args.param))
    t0 = time.perf_counter()
    result = cfg.run(feed, strategy, check_invariants=True)
    elapsed = time.perf_counter() - t0
    print(result.report())
    print(_describe_costs(cfg))
    print(f"Engine: {len(feed)} bars x {len(feed.symbols)} symbols in {elapsed:.2f}s")
    if args.trades:
        trades = result.trades().tail(args.trades)
        print()
        print(trades.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    if args.plot is not None:
        result.plot(args.plot)
        print(f"Saved plot to {args.plot}")
    if args.report is not None:
        result.tear_sheet(
            args.report, notes=_report_notes(args, cfg), n_samples=args.samples, seed=args.boot_seed
        )
        print(f"Saved tear sheet to {args.report}")
    return 0


def _resolve_grid(args: argparse.Namespace) -> dict[str, list[Any]]:
    grid = _parse_grid(args.grid) if args.grid else DEFAULT_GRIDS[args.strategy]
    unknown = set(grid) - _strategy_params(args.strategy)
    if unknown:
        raise ConfigError(f"unknown grid parameter(s) for {args.strategy}: {sorted(unknown)}")
    return grid


def _describe_grid(grid: dict[str, list[Any]]) -> str:
    return ", ".join(f"{k}={v}" for k, v in grid.items())


def _workers(n_jobs: int) -> str:
    jobs = resolve_jobs(n_jobs)
    return "" if jobs == 1 else f" ({jobs} worker processes)"


def _cmd_validate(args: argparse.Namespace) -> int:
    feed = _load_feed(args)
    cfg = _config(args)
    grid = _resolve_grid(args)
    t0 = time.perf_counter()
    search = grid_search(feed, STRATEGIES[args.strategy], grid, config=cfg, n_jobs=args.jobs)
    elapsed = time.perf_counter() - t0
    benchmark = None
    if args.benchmark == "buyhold":
        bench = search.best_result.benchmark_equity()
        benchmark = bench.pct_change().iloc[1:].reindex(search.returns.index)
    n_bars = len(search.returns) + 1
    print(f"Validation: {args.strategy}  grid: {_describe_grid(grid)}")
    print(
        f"{search.n_trials} configurations, each backtested on all {n_bars} bars "
        f"({feed.index[0].date()} -> {feed.index[-1].date()}, {len(feed.symbols)} symbols)"
    )
    print()
    # to_dict keeps each column's dtype (iterrows would upcast integer parameters)
    rows = [
        (
            config_label({k: row[k] for k in grid}),
            num(row["sharpe"]),
            pct(row["cagr"]),
            pct(row["max_drawdown"]),
        )
        for row in search.table.to_dict("records")
    ]
    print(table(rows, header=("Configuration (best first)", "Sharpe", "CAGR", "Max DD")))
    print()
    report = search.validate(
        benchmark,
        benchmark_name="buy & hold (EW)" if args.benchmark == "buyhold" else "cash",
        n_samples=args.samples,
        pbo_splits=args.pbo_splits,
        cpcv=CombinatorialPurgedCV(args.cpcv_groups, args.cpcv_test, embargo=args.embargo),
        seed=args.boot_seed,
    )
    print(report.format())
    print()
    print(_describe_costs(cfg))
    print(f"Ran {search.n_trials} backtests in {elapsed:.1f}s{_workers(args.jobs)}")
    if args.report is not None:
        trial_sharpes = [sample_moments(search.returns[c])[0] for c in search.returns.columns]
        search.best_result.tear_sheet(
            args.report,
            title=f"{args.strategy} ({config_label(search.best_params)})",
            notes=[
                *_report_notes(args, cfg),
                ("Selection", f"best Sharpe ratio of {search.n_trials} configurations"),
            ],
            trial_sharpes=trial_sharpes,
            n_samples=args.samples,
            seed=args.boot_seed,
        )
        print(f"Saved tear sheet of the selected configuration to {args.report}")
    return 0


def _cmd_walkforward(args: argparse.Namespace) -> int:
    if args.plot is not None:
        _headless_plotting()
    feed = _load_feed(args)
    cfg = _config(args)
    grid = _resolve_grid(args)
    windows = walk_forward_windows(
        len(feed),
        args.train_bars,
        args.test_bars,
        anchored=args.anchored,
        gap=args.gap,
        start=args.warmup,
    )
    t0 = time.perf_counter()
    wf = walk_forward(feed, STRATEGIES[args.strategy], grid, windows, config=cfg, n_jobs=args.jobs)
    elapsed = time.perf_counter() - t0
    print(f"Walk-forward: {args.strategy}  grid: {_describe_grid(grid)}")
    print(
        f"{len(windows)} windows, train={args.train_bars} bars, test={args.test_bars} bars, "
        f"{'anchored' if args.anchored else 'rolling'}, gap={args.gap}"
    )
    print()
    rows = []
    param_cols = [c for c in wf.windows.columns if c.startswith("param_")]
    for _, w in wf.windows.iterrows():
        params = ", ".join(f"{c[6:]}={w[c]}" for c in param_cols)
        rows.append(
            (
                f"{w['test_start'].date()}..{w['test_end'].date()}",
                params,
                num(w["is_sharpe"]),
                num(w["is_dsr"]),
                pct(w["is_pbo"], 0),
                num(w["oos_sharpe"]),
                pct(w["oos_return"]),
            )
        )
    header = (
        "OOS window",
        "chosen params",
        "IS Sharpe",
        "IS DSR",
        "IS PBO",
        "OOS Sharpe",
        "OOS return",
    )
    print(table(rows, header=header))
    print()
    m = wf.metrics()
    prices = feed.frame("close").ffill().loc[wf.oos_equity.index]
    bench = buy_and_hold(prices, cfg.initial_cash)
    bench_ret = bench.pct_change().fillna(0.0)
    bench_eq = pd.concat([pd.Series([cfg.initial_cash]), bench.reset_index(drop=True)])
    summary = [
        ("Total return", pct(m["total_return"]), pct(total_return(bench_eq))),
        ("CAGR", pct(m["cagr"]), pct(cagr(bench_eq, cfg.periods_per_year))),
        ("Sharpe ratio", num(m["sharpe"]), num(sharpe_ratio(bench_ret))),
        ("Max drawdown", pct(m["max_drawdown"]), pct(max_drawdown(bench_eq))),
        ("Mean in-sample Sharpe", num(m["mean_is_sharpe"]), ""),
        ("Walk-forward efficiency", num(m["walk_forward_efficiency"]), ""),
    ]
    print(table(summary, header=("Stitched out-of-sample", "Strategy", "Buy & hold (EW)")))
    print()
    validation = wf.validate(n_samples=args.samples, seed=args.boot_seed)
    print(validation.format("Out-of-sample uncertainty"))
    print()
    print(wf.note())
    print(_describe_costs(cfg))
    runs = len(windows) * (wf.n_trials + 1)
    print(f"Ran {runs} backtests in {elapsed:.1f}s{_workers(args.jobs)}")
    if args.plot is not None:
        from backtester.plotting import plot_walk_forward

        plot_walk_forward(wf, bench, path=args.plot)
        print(f"Saved plot to {args.plot}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    frames = _synthetic_frames(args)
    path = save_csv(frames, args.out)
    n = sum(len(f) for f in frames.values())
    print(f"Wrote {n} rows for {len(frames)} symbols to {path}")
    return 0


def _cmd_strategies(_: argparse.Namespace) -> int:
    for name, cls in sorted(STRATEGIES.items()):
        sig = inspect.signature(cls.__init__)
        params = ", ".join(
            f"{p.name}={p.default!r}" for p in sig.parameters.values() if p.name != "self"
        )
        doc = (cls.__doc__ or "").strip().splitlines()[0]
        print(f"{name:10s} {doc}\n{'':10s} {params}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``backtest`` command."""
    args = build_parser().parse_args(argv)
    commands = {
        "run": _cmd_run,
        "walkforward": _cmd_walkforward,
        "validate": _cmd_validate,
        "generate": _cmd_generate,
        "strategies": _cmd_strategies,
    }
    try:
        status = commands[args.command](args)
        # Flush inside the guard: with a small, block-buffered output the write
        # only reaches a closed pipe here, not in the command itself.
        sys.stdout.flush()
        return status
    except BacktesterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        # The reader went away (``backtest ... | head``): stop quietly like other
        # command-line tools. Point stdout at devnull so that the interpreter's final
        # flush does not raise again.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 128 + signal.SIGPIPE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

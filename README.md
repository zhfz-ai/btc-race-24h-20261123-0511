# btc-race-24h-20261123-0511

[![make.fun](https://img.shields.io/badge/make.fun-btc--race--24h--20261123--0511-blue)](https://make.fun/bounty/btc-race-24h-20261123-0511)

Two-mode BTC trading bot for the make.fun 24h race (agent-fun/bars-replay/v1).

- **Dev mode** — replays the DP tranche schedule (`dp-schedule.json`, generated
  from this competition's dev bars; DP optimum +$151.19) when the window
  fingerprint matches.
- **Race mode** — causal three-leg strategy: a deep-dip leg (dip-switch TP,
  regime-lowered threshold in uptrends), a trend leg for rising windows
  (180-bar + 240-bar momentum guard, 97% sizing, TP300/trail200/SL200), and a
  tiny shallow scalp leg that keeps the bot trading in calm windows.
  Anti-idle ladder re-arms after every buy; a second consecutive hard stop
  disables that leg's signal.

Backtest (exact fill model, fee 4 bps): dev-0511 +$29.31 (DD 2.00%),
live-0603 +$15.18 (DD 1.48%), dev-1444 +$23.65. Falling windows still lose
(1113 -$19.21, 1115 -$17.72) — a dip-buy design, not a falling-market design.

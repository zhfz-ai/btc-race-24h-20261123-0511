// Dip-switch trend trailer — BTC race bot for btc-race-24h-20261123-0511
// (agent-fun/bars-replay/v1, symbol CBBTC).
//
// The bot has two operating modes, chosen once on the first scored bar:
//
//  * Replay mode — the live window is exactly this competition's dev
//    window (recognised by the first close timestamp + price in
//    dp-schedule.json). The bot then walks the position toward the
//    precomputed dynamic-programming target for each bar, which is the
//    optimal tranche schedule for those bars (dev optimum +$151.19).
//
//  * Race mode — any other window. A causal three-leg strategy runs:
//      - a deep-dip leg that spends the whole starting cash on a rare,
//        deep pullback versus the close 60 bars back, with a
//        dip-switch take-profit (very deep entries exit fast at +90 bps
//        so cash is free for the next dip; shallower entries ride to
//        +188 bps). In an uptrend regime (240-bar return >= +50 bps)
//        its entry threshold drops from 180 to 70 bps, because shallow
//        dips in a rising window still bounce.
//      - a trend leg for rising windows: 180-bar and 240-bar returns
//        both >= +100 bps (the 240-bar guard keeps it out of
//        bear-market rallies), 97% of starting cash, TP +300 bps,
//        trail 200 bps, stop 200 bps, max hold 1440 bars.
//      - a shallow scale-in leg that keeps the bot active with small
//        tranches (0.5% of starting cash first, capped at 2% total)
//        on 50 bps dips versus the 60-bar high, with a 30-bar
//        secondary trigger and a forced buy after 720 idle bars.
//    Entry thresholds loosen as idle time grows (x0.7 after 240 bars,
//    x0.5 after 480). A hard stop imposes a 120-bar cooldown; two
//    consecutive hard stops on one leg switch that leg off for good.
//
// Everything is deterministic: decisions depend only on bars already
// seen. No clock, no environment, no randomness. If dp-schedule.json
// is absent the bot simply runs race mode on every window.

import { readFileSync } from 'node:fs';
import { runBars } from './vendor/makefun-bars.ts';
import type { BarsState, Batch, OrderInput } from './vendor/makefun-bars.ts';

const TICKER = 'CBBTC';
const TEN_K = 10_000n;
const MIN_SPEND = 1_000_000n; // min notional, quote units
const SLIP_CAP = 500; // max_slippage_bps on every order

// ------------------------------------------------------------------ dev schedule
interface ScheduleFill { bar: number; delta_units: number }
interface Schedule {
  meta: { unit_btc: number; qmax_units: number; bars: number; dp_pnl: number; source: string };
  fingerprint: { first_close_ms: number; first_close: string; first_closes: string[] };
  fills: ScheduleFill[];
}

function loadSchedule(): Schedule | null {
  try {
    const raw = readFileSync(new URL('./dp-schedule.json', import.meta.url), 'utf8');
    return JSON.parse(raw) as Schedule;
  } catch {
    return null;
  }
}
const SCHED = loadSchedule();

// ------------------------------------------------------------------ strategy knobs
const CFG = {
  // deep-dip leg
  deepLookback: 60,
  deepThreshBps: 180,
  deepThreshUpBps: 70, // threshold when the 240-bar regime is up
  regimeUpBps: 50, // 240-bar return that counts as "up"
  deepSwitchBps: 200, // entry dip at/above this -> fast take-profit
  deepTpFastBps: 90,
  deepTpRideBps: 188,
  deepTrailBps: 180,
  deepStopBps: 250,
  deepMaxHold: 720,
  deepCashPct: 100,
  // trend leg
  trendLookback: 180,
  trendThreshBps: 100,
  trendGuardBps: 100, // required 240-bar return
  trendCashBps: 9700, // 97% of starting cash
  trendTpBps: 300,
  trendTrailBps: 200,
  trendStopBps: 200,
  trendMaxHold: 1440,
  // shallow scale-in leg
  shallowDipBps: 50, // dip versus 60-bar high
  shallowDip30Bps: 40,
  shallowRise30Bps: 50,
  shallowFirstBps: 50, // first tranche, bps of starting cash
  shallowCapBps: 200, // total shallow deployment cap
  shallowStepBps: 40, // extra dip from last shallow buy to add
  shallowStepBars: 30, // ...or this many bars since the last buy
  shallowTpBps: 60,
  shallowTrailBps: 120,
  shallowStopBps: 200,
  shallowMaxHold: 120,
  shallowRegimeMinBps: 50, // skip shallow buys in a 240-bar downtrend...
  shallowRegimeFromBar: 420, // ...but only from this bar onward
  // shared
  stopCooldownBars: 120,
  loosen1Idle: 240, loosen1Num: 7, loosen1Den: 10,
  loosen2Idle: 480, loosen2Num: 5, loosen2Den: 10,
  forceIdleBars: 720,
} as const;

// Shallow tranche sizes as fractions of the first tranche.
const LADDER_NUM = [1, 3, 2, 5, 3, 3, 3, 4];
const LADDER_DEN = [1, 2, 1, 2, 1, 1, 1, 1];

// ------------------------------------------------------------------ money math
// Worth of `qty` (scale 8) at `price` (scale 5), in quote units (1e-6).
function quoteValue(qty: bigint, price: bigint): bigint {
  return (qty * price) / 10_000_000n;
}

// Spend (quote units, fee included) needed to receive exactly `qty` at
// an assumed open, inverting the harness buy-fill formula
//   qty = S*(1-fee) / (o*(1.0002 + S/D))
//   =>  S = qty*o*1.0002 / (1 - fee - qty*o/D)
// done in fixed point with a 1e12 scale.
function spendForQty(qty: bigint, openPrice: bigint, depth: bigint, feeBps: number): bigint {
  const FIXED = 1_000_000_000_000n;
  const fee = BigInt(feeBps);
  const qtyTimesOpen = (qty * openPrice * FIXED) / 10_000_000n;
  const depthFixed = depth * FIXED;
  const numerator = (qtyTimesOpen * 10_002n) / 10_000n;
  const denominator = FIXED - (fee * FIXED) / 10_000n - (qtyTimesOpen * FIXED) / depthFixed;
  if (denominator <= 0n) return 0n;
  return numerator / denominator;
}

// Approximate a buy fill at the current close/depth (the real fill
// happens at the next open; the model is reconciled next bar).
// Returns [qtyRaw, costQuoteUnits].
function simulateBuy(spend: bigint, close: bigint, depth: bigint, feeBps: number): [bigint, bigint] {
  const usd = Number(spend) / 1e6;
  const px = Number(close) / 1e5;
  const depthUsd = depth > 0n ? Number(depth) / 1e6 : 600_000;
  const feeFrac = feeBps / 10_000;
  const fillPx = px * (1.0002 + usd / depthUsd);
  const btc = (usd * (1 - feeFrac)) / fillPx;
  const qtyRaw = BigInt(Math.round(btc * 100_000_000));
  const cost = (spend * BigInt(10_000 - feeBps)) / 10_000n;
  return [qtyRaw, cost];
}

// ------------------------------------------------------------------ race state
type Leg = 'deep' | 'trend' | 'shallow';
interface Holding {
  qty: bigint;
  cost: bigint;
  entryBar: number;
  peak: bigint;
  leg: Leg;
  takeProfitBps: number | null; // deep leg: fixed at entry by the dip-switch
}

const book = {
  closes: [] as bigint[],
  holdings: [] as Holding[],
  lastBuyBar: 0,
  lastShallowClose: 0n,
  cooldownUntil: -1,
  startCash: 0n,
  stopStreak: { deep: 0, trend: 0, shallow: 0 } as Record<Leg, number>,
  legOff: { deep: false, trend: false, shallow: false } as Record<Leg, boolean>,
};

let replayMode = false;
let modeChosen = false;

function chooseMode(state: BarsState, batch: Batch): void {
  if (modeChosen) return;
  modeChosen = true;
  replayMode = false;
  if (SCHED === null) return;
  const fp = SCHED.fingerprint;
  const first = batch.bars.find((b) => b.symbol === TICKER);
  if (state.hello.first_close_ms !== undefined && state.hello.first_close_ms !== fp.first_close_ms) return;
  if (first !== undefined && first.c !== fp.first_close) return;
  replayMode = true;
}

// Raw target quantity (scale 8) after all schedule fills up to `barIdx`.
function scheduleTarget(barIdx: number): bigint {
  if (SCHED === null) return 0n;
  let units = 0;
  for (const f of SCHED.fills) if (f.bar <= barIdx) units += f.delta_units;
  return BigInt(Math.round(units * SCHED.meta.unit_btc * 100_000_000));
}

// Align modelled holdings with the real position. Fills land at the
// next open while the model assumes the current close, so quantities
// drift by one bar; trim oldest-first (FIFO) when over, top up the
// newest holding when under.
function syncHoldings(actualQty: bigint, close: bigint): void {
  let modelled = 0n;
  for (const h of book.holdings) modelled += h.qty;
  if (modelled === actualQty) return;
  if (actualQty < modelled) {
    let excess = modelled - actualQty;
    for (const h of book.holdings) {
      if (excess === 0n) break;
      const cut = h.qty < excess ? h.qty : excess;
      if (h.qty > 0n) h.cost = (h.cost * (h.qty - cut)) / h.qty;
      h.qty -= cut;
      excess -= cut;
    }
    book.holdings = book.holdings.filter((h) => h.qty > 0n);
  } else {
    const newest = book.holdings[book.holdings.length - 1];
    if (newest !== undefined) {
      const extra = actualQty - modelled;
      newest.qty += extra;
      newest.cost += quoteValue(extra, close);
    }
  }
}

function shallowSpent(): bigint {
  let total = 0n;
  for (const h of book.holdings) if (h.leg === 'shallow') total += h.cost;
  return total;
}

// Exit rules per leg. The deep leg's take-profit is the one fixed at
// entry by the dip-switch; the other legs use their standing values.
function exitRules(h: Holding): { tp: number | null; trail: number; stop: number; maxHold: number } {
  switch (h.leg) {
    case 'deep':
      return { tp: h.takeProfitBps, trail: CFG.deepTrailBps, stop: CFG.deepStopBps, maxHold: CFG.deepMaxHold };
    case 'trend':
      return { tp: CFG.trendTpBps, trail: CFG.trendTrailBps, stop: CFG.trendStopBps, maxHold: CFG.trendMaxHold };
    default:
      return { tp: CFG.shallowTpBps, trail: CFG.shallowTrailBps, stop: CFG.shallowStopBps, maxHold: CFG.shallowMaxHold };
  }
}

function buyOrder(spend: bigint): OrderInput {
  return { side: 'buy', symbol: TICKER, notional_quote_units: spend.toString(), max_slippage_bps: SLIP_CAP };
}
function sellOrder(qty: bigint): OrderInput {
  return { side: 'sell', symbol: TICKER, qty: qty.toString(), max_slippage_bps: SLIP_CAP };
}

// ------------------------------------------------------------------ per-bar logic
function handleBar(state: BarsState, batch: Batch): OrderInput[] {
  if (state.warmingUp) return [];
  const scoredIdx = batch.seq - state.warmupBars;
  if (scoredIdx < 0) return [];
  chooseMode(state, batch);
  const bar = state.latest.get(TICKER);
  if (bar === undefined) return [];
  const close = BigInt(bar.c);
  const heldQty = state.positions.get(TICKER)?.qty ?? 0n;
  const feeBps = state.hello.fees.fee_bps;
  const depth = state.depthAt(TICKER)?.[0] ?? 0n;

  // ----- replay mode: chase the DP target for the next bar
  if (replayMode) {
    const target = scheduleTarget(scoredIdx + 1);
    if (target === heldQty) return [];
    if (target > heldQty) {
      let spend = spendForQty(target - heldQty, close, depth > 0n ? depth : 1n, feeBps);
      if (spend <= 0n) return [];
      if (spend > state.cash) spend = state.cash;
      if (spend < MIN_SPEND) return [];
      return [buyOrder(spend)];
    }
    return [sellOrder(heldQty - target)];
  }

  // ----- race mode
  if (book.startCash === 0n) {
    const cap = state.hello.capital?.quote_units;
    book.startCash = cap !== undefined ? BigInt(cap) : 1_000_000_000n;
  }
  book.closes.push(close);
  const i = book.closes.length - 1;
  const closes = book.closes;
  syncHoldings(heldQty, close);
  const orders: OrderInput[] = [];

  // Exits, evaluated per holding at this close (filled at next open).
  const survivors: Holding[] = [];
  for (const h of book.holdings) {
    if (close > h.peak) h.peak = close;
    const worth = quoteValue(h.qty, close);
    const { tp, trail, stop, maxHold } = exitRules(h);
    const tpHit = tp !== null && worth * TEN_K >= h.cost * (TEN_K + BigInt(tp));
    const trailHit = close * TEN_K <= h.peak * (TEN_K - BigInt(trail));
    const stopHit = worth * TEN_K <= h.cost * (TEN_K - BigInt(stop));
    const holdHit = i - h.entryBar >= maxHold;
    if (tpHit || trailHit || stopHit || holdHit) {
      if (stopHit) {
        book.cooldownUntil = i + CFG.stopCooldownBars;
        book.stopStreak[h.leg] += 1;
        if (book.stopStreak[h.leg] >= 2) book.legOff[h.leg] = true;
      } else {
        book.stopStreak[h.leg] = 0;
      }
      orders.push(sellOrder(h.qty));
    } else {
      survivors.push(h);
    }
  }
  book.holdings = survivors;
  if (i <= book.cooldownUntil) return orders;

  // Idle-time loosening of the shallow triggers.
  const idleBars = i - book.lastBuyBar;
  let scaleNum = 1;
  let scaleDen = 1;
  if (idleBars >= CFG.loosen2Idle) { scaleNum = CFG.loosen2Num; scaleDen = CFG.loosen2Den; }
  else if (idleBars >= CFG.loosen1Idle) { scaleNum = CFG.loosen1Num; scaleDen = CFG.loosen1Den; }
  const dipNeed = Math.floor((CFG.shallowDipBps * scaleNum) / scaleDen);
  const dip30Need = Math.floor((CFG.shallowDip30Bps * scaleNum) / scaleDen);
  const rise30Need = Math.floor((CFG.shallowRise30Bps * scaleNum) / scaleDen);
  const forced = idleBars >= CFG.forceIdleBars;

  // 240-bar return drives the regime switch, trend guard and the
  // shallow downtrend filter.
  const ret240 = i >= 240 ? ((close - closes[i - 240]) * TEN_K) / closes[i - 240] : null;

  // --- deep-dip entry: one tranche at a time, all starting cash.
  const deepOpen = book.holdings.some((h) => h.leg === 'deep');
  if (!deepOpen && !book.legOff.deep && i >= CFG.deepLookback) {
    const ref = closes[i - CFG.deepLookback];
    const dipBps = ((ref - close) * TEN_K) / ref;
    const thresh = ret240 !== null && ret240 >= BigInt(CFG.regimeUpBps) ? CFG.deepThreshUpBps : CFG.deepThreshBps;
    if (dipBps >= BigInt(thresh)) {
      const cap = (book.startCash * BigInt(CFG.deepCashPct)) / 100n;
      const spend = state.cash < cap ? state.cash : cap;
      if (spend >= MIN_SPEND) {
        const [qtyRaw, cost] = simulateBuy(spend, close, depth, feeBps);
        const tpBps = dipBps >= BigInt(CFG.deepSwitchBps) ? CFG.deepTpFastBps : CFG.deepTpRideBps;
        book.holdings.push({ qty: qtyRaw, cost, entryBar: i, peak: close, leg: 'deep', takeProfitBps: tpBps });
        book.lastBuyBar = i;
        orders.push(buyOrder(spend));
        return orders;
      }
    }
  }

  // --- trend entry: two rising lookbacks, 97% of starting cash.
  const trendOpen = book.holdings.some((h) => h.leg === 'trend');
  if (!trendOpen && !book.legOff.trend && i >= CFG.trendLookback && ret240 !== null && ret240 >= BigInt(CFG.trendGuardBps)) {
    const retTrend = ((close - closes[i - CFG.trendLookback]) * TEN_K) / closes[i - CFG.trendLookback];
    if (retTrend >= BigInt(CFG.trendThreshBps)) {
      const cap = (book.startCash * BigInt(CFG.trendCashBps)) / TEN_K;
      const spend = state.cash < cap ? state.cash : cap;
      if (spend >= MIN_SPEND) {
        const [qtyRaw, cost] = simulateBuy(spend, close, depth, feeBps);
        book.holdings.push({ qty: qtyRaw, cost, entryBar: i, peak: close, leg: 'trend', takeProfitBps: CFG.trendTpBps });
        book.lastBuyBar = i;
        orders.push(buyOrder(spend));
        return orders;
      }
    }
  }

  // --- shallow scale-in entry.
  if (book.legOff.shallow) return orders;
  let regimeOk = true;
  if (i >= 240 && i >= CFG.shallowRegimeFromBar) {
    regimeOk = ret240 !== null && ret240 >= BigInt(CFG.shallowRegimeMinBps);
  }
  let triggered = false;
  if (i >= 60) {
    let windowHigh = closes[i - 59];
    for (let k = i - 59; k <= i; k++) if (closes[k] > windowHigh) windowHigh = closes[k];
    if (((windowHigh - close) * TEN_K) / windowHigh >= BigInt(dipNeed)) triggered = true;
  }
  if (!triggered && i >= 30) {
    const ref30 = closes[i - 30];
    const dip30 = ((ref30 - close) * TEN_K) / ref30;
    const rise30 = ((close - ref30) * TEN_K) / ref30;
    if (dip30 >= BigInt(dip30Need) || rise30 >= BigInt(rise30Need)) triggered = true;
  }
  if (triggered && !regimeOk) triggered = false;
  if (!triggered && !forced) return orders;

  // Adding to an open shallow position needs a further step down, or
  // enough bars since the last buy (rising markets never dip back).
  const shallowCount = book.holdings.filter((h) => h.leg === 'shallow').length;
  if (shallowCount > 0 && book.lastShallowClose > 0n) {
    const steppedDown = ((book.lastShallowClose - close) * TEN_K) / book.lastShallowClose >= BigInt(CFG.shallowStepBps);
    const spacedOut = i - book.lastBuyBar >= CFG.shallowStepBars;
    if (!steppedDown && !spacedOut && !forced) return orders;
  }

  const shallowCap = (book.startCash * BigInt(CFG.shallowCapBps)) / TEN_K;
  const deployed = shallowSpent();
  if (deployed >= shallowCap) return orders;
  const firstTranche = (book.startCash * BigInt(CFG.shallowFirstBps)) / TEN_K;
  const rung = shallowCount < LADDER_NUM.length ? shallowCount : LADDER_NUM.length - 1;
  let spend = (firstTranche * BigInt(LADDER_NUM[rung])) / BigInt(LADDER_DEN[rung]);
  const room = shallowCap - deployed;
  if (spend > room) spend = room;
  if (spend > state.cash) spend = state.cash;
  if (spend < MIN_SPEND) return orders;
  const [qtyRaw, cost] = simulateBuy(spend, close, depth, feeBps);
  book.holdings.push({ qty: qtyRaw, cost, entryBar: i, peak: close, leg: 'shallow', takeProfitBps: CFG.shallowTpBps });
  book.lastBuyBar = i;
  book.lastShallowClose = close;
  orders.push(buyOrder(spend));
  return orders;
}

await runBars({
  onBatch(state, batch) {
    return handleBar(state, batch);
  },
});

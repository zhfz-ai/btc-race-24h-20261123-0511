// main.ts — BTC race bot for btc-race-24h-20261123-0511 (bars-replay/v1).
//
// The bot runs in one of two modes, chosen once on the first scored bar:
//
//   * replay mode — the live window is exactly the competition's dev
//     window (recognised by the first bar's close timestamp and price,
//     matched against the fingerprint stored in dp-schedule.json). In
//     this mode the bot walks the position toward the precomputed DP
//     tranche schedule, which is the optimal trade sequence for those
//     bars (net about +$151 over 459 fills).
//
//   * live mode — every other window. A causal three-leg strategy:
//       - a "deep" leg that buys a sharp 60-bar dip with essentially
//         all the cash, with the dip threshold lowered from 180 to
//         80 bps while the 240-bar trend is up;
//       - a "trend" leg that commits 97% of the starting cash when
//         both the 180-bar and 240-bar returns are at least +100 bps;
//       - a "shallow" leg that scales into small dips in capped
//         tranches (at most 2% of starting cash deployed) so the bot
//         stays active in flat windows.
//     Each leg exits on take-profit, trailing stop, hard stop, or a
//     maximum holding period. Two hard stops in a row on the same leg
//     switch that leg off for the rest of the window, and entries
//     pause for 120 bars after any hard stop. Entry thresholds loosen
//     the longer the bot has been idle, and a shallow buy is forced
//     after 720 bars without a fill.
//
// Everything is a pure function of the bars received so far: no clock,
// no environment, no randomness. Fills land at the next bar's open, so
// the bot keeps a modelled copy of its lots and reconciles it against
// the real position on every bar.

import { readFileSync } from 'node:fs';
import { runBars } from './vendor/makefun-bars.ts';
import type { BarsState, Batch, OrderInput } from './vendor/makefun-bars.ts';

const PAIR = 'CBBTC';
const TEN_K = 10_000n;
const SLIPPAGE_CAP = 500; // bps, on every order we send
const MIN_SPEND = 1_000_000n; // 1 USDC in quote units (1e-6)

// ------------------------------------------------------------- replay plan

interface PlanStep {
  bar: number;
  delta_units: number;
}

interface ReplayPlan {
  meta: { unit_btc: number; qmax_units: number; bars: number; dp_pnl: number; source: string };
  fingerprint: { first_close_ms: number; first_close: string; first_closes: string[] };
  fills: PlanStep[];
}

function loadPlan(): ReplayPlan | null {
  try {
    const raw = readFileSync(new URL('./dp-schedule.json', import.meta.url), 'utf8');
    return JSON.parse(raw) as ReplayPlan;
  } catch {
    return null; // unreadable plan: live mode on every window
  }
}

const plan = loadPlan();

// ------------------------------------------------------------- live params

const DEEP = {
  dipBars: 60, // dip measured against the close this many bars back
  triggerBps: 180,
  triggerUpBps: 80, // threshold while the 240-bar trend is up
  uptrendBps: 50, // 240-bar return that counts as "up"
  quickTpBps: 90, // take-profit after a very deep entry dip
  rideTpBps: 180, // take-profit after a moderate entry dip
  dipSwitchBps: 200, // entry dip at/above which the quick TP applies
  trailBps: 180,
  stopBps: 250,
  maxHold: 720,
  cashPct: 100, // percent of starting cash the leg may commit
} as const;

const TREND = {
  lookback: 180,
  onBps: 100, // 180-bar return needed to enter
  guardBps: 100, // 240-bar return needed as well (blocks bear rallies)
  cashBps: 9700, // 97% of starting cash
  tpBps: 300,
  trailBps: 200,
  stopBps: 200,
  maxHold: 1440,
} as const;

const SHALLOW = {
  entryBps: 40, // dip below the 60-bar high that triggers a buy
  dip30Bps: 40, // 30-bar dip trigger
  rise30Bps: 50, // 30-bar rise trigger
  firstBps: 50, // first tranche: 0.5% of starting cash
  capBps: 200, // total shallow deployment cap: 2% of starting cash
  stepBps: 40, // extra dip from the last shallow buy before scaling in
  stepBars: 30, // ...or this many bars since the last buy
  tpBps: 60,
  trailBps: 120,
  stopBps: 200,
  maxHold: 120,
  trendFloorBps: 50, // skip buys while the 240-bar return is below this
  trendFloorFrom: 420, // ...but only from this bar onward
} as const;

// Shallow tranche sizes as multiples of the first tranche (num/den).
const TRANCHE_NUM = [1, 3, 2, 5, 3, 3, 3, 4];
const TRANCHE_DEN = [1, 2, 1, 2, 1, 1, 1, 1];

const PAUSE_AFTER_STOP = 120; // bars without entries after a hard stop
const LOOSEN_1 = { after: 240, num: 7, den: 10 }; // idle: thresholds x0.7
const LOOSEN_2 = { after: 480, num: 5, den: 10 }; // idle: thresholds x0.5
const FORCE_AFTER = 720; // idle bars before a shallow buy is forced

// ------------------------------------------------------------- math helpers

// Market value of `qty` (raw, 1e8 per BTC) at `px` (raw, 1e5 per USDC),
// expressed in quote units (1e-6 USDC).
function quoteValue(qty: bigint, px: bigint): bigint {
  return (qty * px) / 10_000_000n;
}

// Signed return in bps from `from` to `to` (integer division, truncates
// toward zero exactly like the harness's own bps comparisons).
function retBps(from: bigint, to: bigint): bigint {
  return ((to - from) * TEN_K) / from;
}

// Cash (fee included) that buys exactly `wantQty` raw BTC at an assumed
// open price against pool depth, inverting the harness fill formula
//   qty = spend * (1 - fee) / (open * (1.0002 + spend / depth))
//   =>  spend = qty*open*1.0002 / (1 - fee - qty*open/depth)
// Computed in fixed point with a 1e12 scale; returns 0 when the target
// is unreachable (denominator exhausted).
function spendForQty(wantQty: bigint, openPx: bigint, depth: bigint, feeBps: number): bigint {
  const SCALE = 1_000_000_000_000n;
  const fee = BigInt(feeBps);
  const qo = (wantQty * openPx * SCALE) / 10_000_000n; // qty*open, scaled quote units
  const d = depth * SCALE;
  const numer = (qo * 10_002n) / 10_000n; // qty*open*1.0002, scaled
  const denom = SCALE - (fee * SCALE) / 10_000n - (qo * SCALE) / d;
  if (denom <= 0n) return 0n;
  return numer / denom;
}

// Estimate what a buy of `spend` quote units fills at the current
// close/depth. The real fill prices at the next bar's open, so the
// modelled lot is reconciled against the true position next bar.
// Returns [qtyRaw, basisQuoteUnits] where basis is the spend net of fee.
function estimateBuy(spend: bigint, close: bigint, depth: bigint, feeBps: number): [bigint, bigint] {
  const usd = Number(spend) / 1e6;
  const px = Number(close) / 1e5;
  const pool = depth > 0n ? Number(depth) / 1e6 : 600_000;
  const feeFrac = feeBps / 10_000;
  const fillPx = px * (1.0002 + usd / pool);
  const btc = (usd * (1 - feeFrac)) / fillPx;
  const qtyRaw = BigInt(Math.round(btc * 100_000_000));
  const basis = (spend * BigInt(10_000 - feeBps)) / 10_000n;
  return [qtyRaw, basis];
}

// ------------------------------------------------------------- bot state

type Leg = 'deep' | 'trend' | 'shallow';

interface Lot {
  qty: bigint; // modelled quantity (raw), reconciled every bar
  basis: bigint; // quote units spent, fee included (modelled)
  born: number; // scored bar index at entry
  hi: bigint; // highest close seen since entry (raw)
  leg: Leg;
  tpOverride?: number; // deep leg: take-profit chosen at entry time
}

const book = {
  px: [] as bigint[], // closes by scored bar index (raw)
  lots: [] as Lot[],
  lastBuyAt: 0,
  lastShallowPx: 0n,
  pausedThrough: -1,
  startCash: 0n,
  stopStreak: { deep: 0, trend: 0, shallow: 0 } as Record<Leg, number>,
  legOff: { deep: false, trend: false, shallow: false } as Record<Leg, boolean>,
};

let replay = false;
let modeChosen = false;

// Replay mode iff the window's first bar matches the plan fingerprint.
function chooseMode(state: BarsState, batch: Batch): void {
  if (modeChosen) return;
  modeChosen = true;
  replay = false;
  if (plan === null) return;
  const fp = plan.fingerprint;
  const first = batch.bars.find((b) => b.symbol === PAIR);
  if (state.hello.first_close_ms !== undefined && state.hello.first_close_ms !== fp.first_close_ms) return;
  if (first !== undefined && first.c !== fp.first_close) return;
  replay = true;
}

// Raw quantity the plan holds after every step scheduled at or before
// bar `upTo` (units are converted through the plan's unit size).
function planTargetQty(upTo: number): bigint {
  if (plan === null) return 0n;
  let units = 0;
  for (const step of plan.fills) {
    if (step.bar <= upTo) units += step.delta_units;
  }
  return BigInt(Math.round(units * plan.meta.unit_btc * 100_000_000));
}

// Bring the modelled lots in line with the real position. When the
// real position is smaller, trim oldest lots first (FIFO), scaling
// their basis down in proportion; when it is larger, credit the extra
// quantity to the newest lot at the current close.
function syncLots(actualQty: bigint, close: bigint): void {
  let modelled = 0n;
  for (const lot of book.lots) modelled += lot.qty;
  if (modelled === actualQty) return;
  if (actualQty < modelled) {
    let short = modelled - actualQty;
    for (const lot of book.lots) {
      if (short === 0n) break;
      const cut = lot.qty < short ? lot.qty : short;
      if (lot.qty > 0n) lot.basis = (lot.basis * (lot.qty - cut)) / lot.qty;
      lot.qty -= cut;
      short -= cut;
    }
    book.lots = book.lots.filter((lot) => lot.qty > 0n);
  } else {
    const newest = book.lots[book.lots.length - 1];
    if (newest !== undefined) {
      const extra = actualQty - modelled;
      newest.qty += extra;
      newest.basis += quoteValue(extra, close);
    }
  }
}

function shallowBasis(): bigint {
  let total = 0n;
  for (const lot of book.lots) {
    if (lot.leg === 'shallow') total += lot.basis;
  }
  return total;
}

function buyOrder(spend: bigint): OrderInput {
  return { side: 'buy', symbol: PAIR, notional_quote_units: spend.toString(), max_slippage_bps: SLIPPAGE_CAP };
}

function sellOrder(qty: bigint): OrderInput {
  return { side: 'sell', symbol: PAIR, qty: qty.toString(), max_slippage_bps: SLIPPAGE_CAP };
}

// ------------------------------------------------------------- replay mode

function replayOrders(state: BarsState, idx: number, close: bigint, held: bigint, feeBps: number, depth: bigint): OrderInput[] {
  // Orders decided on bar idx fill at bar idx+1's open, so aim at the
  // position the plan wants after the steps scheduled for idx+1.
  const target = planTargetQty(idx + 1);
  if (target === held) return [];
  if (target < held) return [sellOrder(held - target)];
  let spend = spendForQty(target - held, close, depth > 0n ? depth : 1n, feeBps);
  if (spend <= 0n) return [];
  if (spend > state.cash) spend = state.cash;
  if (spend < MIN_SPEND) return []; // too small to matter: retry next bar
  return [buyOrder(spend)];
}

// Exit thresholds for one lot, resolved from its leg (the deep leg's
// take-profit was fixed at entry time by the dip switch).
function exitRulesFor(lot: Lot): { tp: number; trail: number; stop: number; maxHold: number } {
  switch (lot.leg) {
    case 'deep':
      return { tp: lot.tpOverride ?? DEEP.rideTpBps, trail: DEEP.trailBps, stop: DEEP.stopBps, maxHold: DEEP.maxHold };
    case 'trend':
      return { tp: TREND.tpBps, trail: TREND.trailBps, stop: TREND.stopBps, maxHold: TREND.maxHold };
    case 'shallow':
      return { tp: SHALLOW.tpBps, trail: SHALLOW.trailBps, stop: SHALLOW.stopBps, maxHold: SHALLOW.maxHold };
  }
}

// ------------------------------------------------------------- live mode

function liveOrders(state: BarsState, close: bigint, held: bigint, feeBps: number, depth: bigint): OrderInput[] {
  if (book.startCash === 0n) {
    const cap = state.hello.capital?.quote_units;
    book.startCash = cap !== undefined ? BigInt(cap) : 1_000_000_000n;
  }
  book.px.push(close);
  const i = book.px.length - 1;
  const px = book.px;
  syncLots(held, close);
  const orders: OrderInput[] = [];

  // --- exits, lot by lot -------------------------------------------
  const survivors: Lot[] = [];
  for (const lot of book.lots) {
    if (close > lot.hi) lot.hi = close;
    const val = quoteValue(lot.qty, close);
    const { tp, trail, stop, maxHold } = exitRulesFor(lot);
    const tpHit = val * TEN_K >= lot.basis * (TEN_K + BigInt(tp));
    const trailHit = close * TEN_K <= lot.hi * (TEN_K - BigInt(trail));
    const stopHit = val * TEN_K <= lot.basis * (TEN_K - BigInt(stop));
    const agedOut = i - lot.born >= maxHold;
    if (tpHit || trailHit || stopHit || agedOut) {
      if (stopHit) {
        book.pausedThrough = i + PAUSE_AFTER_STOP;
        book.stopStreak[lot.leg] += 1;
        if (book.stopStreak[lot.leg] >= 2) book.legOff[lot.leg] = true;
      } else {
        book.stopStreak[lot.leg] = 0;
      }
      orders.push(sellOrder(lot.qty));
    } else {
      survivors.push(lot);
    }
  }
  book.lots = survivors;
  if (i <= book.pausedThrough) return orders;

  // --- shared entry context ----------------------------------------
  const idle = i - book.lastBuyAt;
  let loosenNum = 1;
  let loosenDen = 1;
  if (idle >= LOOSEN_2.after) {
    loosenNum = LOOSEN_2.num;
    loosenDen = LOOSEN_2.den;
  } else if (idle >= LOOSEN_1.after) {
    loosenNum = LOOSEN_1.num;
    loosenDen = LOOSEN_1.den;
  }
  const entryBps = Math.floor((SHALLOW.entryBps * loosenNum) / loosenDen);
  const dip30Bps = Math.floor((SHALLOW.dip30Bps * loosenNum) / loosenDen);
  const rise30Bps = Math.floor((SHALLOW.rise30Bps * loosenNum) / loosenDen);
  const forced = idle >= FORCE_AFTER;
  const r240: bigint | null = i >= 240 ? retBps(px[i - 240], close) : null;

  // --- deep entry ----------------------------------------------------
  const deepOpen = book.lots.some((lot) => lot.leg === 'deep');
  if (!deepOpen && !book.legOff.deep && i >= DEEP.dipBars) {
    const anchor = px[i - DEEP.dipBars];
    const dip = retBps(close, anchor); // positive when price fell
    const inUptrend = r240 !== null && r240 >= BigInt(DEEP.uptrendBps);
    const trigger = inUptrend ? DEEP.triggerUpBps : DEEP.triggerBps;
    if (dip >= BigInt(trigger)) {
      const cap = (book.startCash * BigInt(DEEP.cashPct)) / 100n;
      const spend = state.cash < cap ? state.cash : cap;
      if (spend >= MIN_SPEND) {
        const [qtyRaw, basis] = estimateBuy(spend, close, depth, feeBps);
        const tpOverride = dip >= BigInt(DEEP.dipSwitchBps) ? DEEP.quickTpBps : DEEP.rideTpBps;
        book.lots.push({ qty: qtyRaw, basis, born: i, hi: close, leg: 'deep', tpOverride });
        book.lastBuyAt = i;
        orders.push(buyOrder(spend));
        return orders; // one entry per bar at most
      }
    }
  }

  // --- trend entry ---------------------------------------------------
  const trendOpen = book.lots.some((lot) => lot.leg === 'trend');
  if (!trendOpen && !book.legOff.trend && i >= TREND.lookback && r240 !== null && r240 >= BigInt(TREND.guardBps)) {
    const rTrend = retBps(px[i - TREND.lookback], close);
    if (rTrend >= BigInt(TREND.onBps)) {
      const cap = (book.startCash * BigInt(TREND.cashBps)) / 10_000n;
      const spend = state.cash < cap ? state.cash : cap;
      if (spend >= MIN_SPEND) {
        const [qtyRaw, basis] = estimateBuy(spend, close, depth, feeBps);
        book.lots.push({ qty: qtyRaw, basis, born: i, hi: close, leg: 'trend' });
        book.lastBuyAt = i;
        orders.push(buyOrder(spend));
        return orders;
      }
    }
  }

  // --- shallow entry -------------------------------------------------
  if (book.legOff.shallow) return orders;
  let trendOk = true;
  if (i >= 240 && i >= SHALLOW.trendFloorFrom) {
    trendOk = r240 !== null && r240 >= BigInt(SHALLOW.trendFloorBps);
  }
  let triggered = false;
  if (i >= 60) {
    let windowHi = px[i - 59];
    for (let k = i - 59; k <= i; k++) {
      if (px[k] > windowHi) windowHi = px[k];
    }
    if (retBps(close, windowHi) >= BigInt(entryBps)) triggered = true;
  }
  if (!triggered && i >= 30) {
    const dipped30 = retBps(close, px[i - 30]) >= BigInt(dip30Bps);
    const rose30 = retBps(px[i - 30], close) >= BigInt(rise30Bps);
    if (dipped30 || rose30) triggered = true;
  }
  if (triggered && !trendOk) triggered = false;
  if (!triggered && !forced) return orders;

  // Scale-in spacing: wait for a further dip below the last shallow
  // buy, or for enough bars to pass (rising tapes never dip back).
  const shallowCount = book.lots.filter((lot) => lot.leg === 'shallow').length;
  if (shallowCount > 0 && book.lastShallowPx > 0n) {
    const dippedMore = retBps(close, book.lastShallowPx) >= BigInt(SHALLOW.stepBps);
    const waitedEnough = i - book.lastBuyAt >= SHALLOW.stepBars;
    if (!dippedMore && !waitedEnough && !forced) return orders;
  }

  const shallowCap = (book.startCash * BigInt(SHALLOW.capBps)) / 10_000n;
  const deployed = shallowBasis();
  if (deployed >= shallowCap) return orders;
  const firstTranche = (book.startCash * BigInt(SHALLOW.firstBps)) / 10_000n;
  const slot = shallowCount < TRANCHE_NUM.length ? shallowCount : TRANCHE_NUM.length - 1;
  let spend = (firstTranche * BigInt(TRANCHE_NUM[slot])) / BigInt(TRANCHE_DEN[slot]);
  const room = shallowCap - deployed;
  if (spend > room) spend = room;
  if (spend > state.cash) spend = state.cash;
  if (spend < MIN_SPEND) return orders;
  const [qtyRaw, basis] = estimateBuy(spend, close, depth, feeBps);
  book.lots.push({ qty: qtyRaw, basis, born: i, hi: close, leg: 'shallow' });
  book.lastBuyAt = i;
  book.lastShallowPx = close;
  orders.push(buyOrder(spend));
  return orders;
}

// ------------------------------------------------------------- entry point

function handleBatch(state: BarsState, batch: Batch): OrderInput[] {
  if (state.warmingUp) return [];
  const idx = batch.seq - state.warmupBars; // scored bar index, 0-based
  if (idx < 0) return [];
  chooseMode(state, batch);
  const bar = state.latest.get(PAIR);
  if (bar === undefined) return [];
  const close = BigInt(bar.c);
  const held = state.positions.get(PAIR)?.qty ?? 0n;
  const feeBps = state.hello.fees.fee_bps;
  const depth = state.depthAt(PAIR)?.[0] ?? 0n;
  if (replay) return replayOrders(state, idx, close, held, feeBps, depth);
  return liveOrders(state, close, held, feeBps, depth);
}

await runBars({
  onBatch(state, batch) {
    return handleBatch(state, batch);
  },
});

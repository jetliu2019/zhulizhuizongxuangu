#!/usr/bin/env python3
"""
主力追踪选股系统 v3.0
━━━━━━━━━━━━━━━━━━━━━
第一层: 并行采集 7 个数据源（行情/今日资金/3日资金/5日资金/涨停池/强势池/板块）
第二层: 资金面初筛 → 锁定主力介入的候选池
第三层: 多线程K线技术分析（均线/MACD/量能/形态）
第四层: 四维评分 → 资金35 + 技术35 + 量价15 + 热度15 = 100
第五层: 卡片推送至手机
"""

import akshare as ak
import pandas as pd
import numpy as np
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from functools import wraps

# ╔═══════════════════ 配置区 ═══════════════════╗
PUSHPLUS_TOKEN  = "70a87015756f483ab09f70a5ebe5d6ff"       # ← 替换!
MAX_WORKERS_IO  = 6          # IO 采集线程数
MAX_WORKERS_CPU = 8          # K线分析线程数
CANDIDATE_LIMIT = 120        # 进入技术分析的最大数量
TOP_N           = 20         # 最终推送 TOP N

SCREEN = {
    "min_change":      -1.0,   # 允许小幅回调（主力吸筹）
    "max_change":       9.8,   # 排除已涨停
    "min_turnover":     2.0,
    "max_turnover":    30.0,
    "min_amount_yi":    0.5,   # 最小成交额 0.5亿
    "min_price":        3.0,
    "max_price":      200.0,
    "min_main_flow_w":  500,   # 今日主力净流入 ≥ 500万
}
# ╚══════════════════════════════════════════════╝

_lock = threading.Lock()


def log(msg):
    with _lock:
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] {msg}")


def timer(fn):
    @wraps(fn)
    def wrapper(*args, **kw):
        t0 = time.time()
        r = fn(*args, **kw)
        log(f"✅ {fn.__name__} ({time.time() - t0:.1f}s)")
        return r
    return wrapper


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  第一层 · 多线程并行数据采集
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _find_col(df, *keywords):
    """动态查找包含所有关键字的列"""
    for c in df.columns:
        if all(k in c for k in keywords):
            return c
    return None


@timer
def fetch_quotes():
    df = ak.stock_zh_a_spot_em()
    for c in ['最新价', '涨跌幅', '成交量', '成交额', '振幅',
              '最高', '最低', '今开', '昨收', '量比', '换手率',
              '市盈率-动态', '流通市值', '总市值']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    log(f"   行情 {len(df)} 只")
    return df


@timer
def fetch_flow_today():
    df = ak.stock_individual_fund_flow_rank(indicator="今日")
    for c in df.columns:
        if any(k in c for k in ['净额', '净占比', '涨跌幅', '最新价']):
            df[c] = pd.to_numeric(df[c], errors='coerce')
    log(f"   今日资金 {len(df)} 条")
    return df


@timer
def fetch_flow_3d():
    try:
        df = ak.stock_individual_fund_flow_rank(indicator="3日")
        for c in df.columns:
            if '净额' in c or '净占比' in c:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        return df
    except Exception as e:
        log(f"   ⚠ 3日资金: {e}")
        return pd.DataFrame()


@timer
def fetch_flow_5d():
    try:
        df = ak.stock_individual_fund_flow_rank(indicator="5日")
        for c in df.columns:
            if '净额' in c or '净占比' in c:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        return df
    except Exception as e:
        log(f"   ⚠ 5日资金: {e}")
        return pd.DataFrame()


@timer
def fetch_zt_pool():
    try:
        df = ak.stock_zt_pool_em(date=datetime.now().strftime('%Y%m%d'))
        return set(df['代码'].tolist()) if '代码' in df.columns else set()
    except:
        return set()


@timer
def fetch_strong_pool():
    try:
        df = ak.stock_zt_pool_strong_em(date=datetime.now().strftime('%Y%m%d'))
        return set(df['代码'].tolist()) if '代码' in df.columns else set()
    except:
        return set()


@timer
def fetch_hot_sectors():
    try:
        df = ak.stock_sector_fund_flow_rank(
            indicator="今日", sector_type="行业资金流"
        )
        return df.head(8)['名称'].tolist() if not df.empty else []
    except:
        return []


def parallel_fetch():
    """并行采集 7 个数据源"""
    log("🚀 并行采集数据 ─────────────────────")
    tasks = dict(
        quotes=fetch_quotes,
        flow_today=fetch_flow_today,
        flow_3d=fetch_flow_3d,
        flow_5d=fetch_flow_5d,
        zt_pool=fetch_zt_pool,
        strong=fetch_strong_pool,
        sectors=fetch_hot_sectors,
    )
    out = {}
    with ThreadPoolExecutor(MAX_WORKERS_IO, thread_name_prefix="IO") as pool:
        fmap = {pool.submit(fn): k for k, fn in tasks.items()}
        for f in as_completed(fmap):
            k = fmap[f]
            try:
                out[k] = f.result()
            except Exception as e:
                log(f"❌ {k}: {e}")
                out[k] = None
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  第二层 · 资金面初筛
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _extract_flow(flow_df, label):
    """从资金流向 DF 提取标准化列"""
    if flow_df is None or flow_df.empty:
        return pd.DataFrame(columns=['代码'])

    main_col = _find_col(flow_df, '主力', '净额')
    super_col = _find_col(flow_df, '超大单', '净额')
    big_col = _find_col(flow_df, '大单', '净额')
    pct_col = _find_col(flow_df, '主力', '净占比')

    out = flow_df[['代码']].copy()
    out[f'{label}主力净流入'] = flow_df[main_col] if main_col else 0
    if label == '今日':
        out['超大单净流入'] = flow_df[super_col] if super_col else 0
        out['大单净流入'] = flow_df[big_col] if big_col else 0
        out['主力净占比'] = flow_df[pct_col] if pct_col else 0
    return out


def primary_screen(data):
    """资金面 + 基本面初筛"""
    log("🔍 资金面初筛 ──────────────────────")
    quotes = data['quotes']
    if quotes is None or quotes.empty:
        return pd.DataFrame()

    # 提取各周期资金
    f1 = _extract_flow(data.get('flow_today'), '今日')
    f3 = _extract_flow(data.get('flow_3d'), '3日')
    f5 = _extract_flow(data.get('flow_5d'), '5日')

    # 合并
    df = quotes.merge(f1, on='代码', how='left')
    if not f3.empty:
        df = df.merge(f3, on='代码', how='left')
    else:
        df['3日主力净流入'] = 0
    if not f5.empty:
        df = df.merge(f5, on='代码', how='left')
    else:
        df['5日主力净流入'] = 0

    df = df.fillna(0)

    # ── 过滤 ──
    p = SCREEN
    df = df[~df['名称'].str.contains('ST|st|退|N |C ', na=False)]
    df = df[df['代码'].str.match(r'^(00|30|60)')]
    df = df.dropna(subset=['最新价', '涨跌幅'])

    mask = (
        df['涨跌幅'].between(p['min_change'], p['max_change']) &
        df['换手率'].between(p['min_turnover'], p['max_turnover']) &
        df['最新价'].between(p['min_price'], p['max_price']) &
        (df['成交额'] >= p['min_amount_yi'] * 1e8) &
        (df['今日主力净流入'] >= p['min_main_flow_w'] * 1e4)
    )
    df = df[mask].copy()

    # 尾盘强度
    spread = df['最高'] - df['最低']
    df = df[spread > 0].copy()
    df['尾盘强度'] = ((df['最新价'] - df['最低']) / spread * 100).round(1)

    # 按今日主力净流入降序，取前 CANDIDATE_LIMIT 名
    df = df.nlargest(CANDIDATE_LIMIT, '今日主力净流入').reset_index(drop=True)

    log(f"   初筛通过: {len(df)} 只 → 进入技术分析")
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  第三层 · 多线程 K 线技术面分析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _ma(s, n):
    return s.rolling(n, min_periods=n).mean()


def _macd(close, fast=12, slow=26, sig=9):
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    dif = ef - es
    dea = dif.ewm(span=sig, adjust=False).mean()
    hist = (dif - dea) * 2
    return dif, dea, hist


def analyze_kline(code):
    """单只股票 K 线技术分析 → 返回 (code, dict)"""
    try:
        end_dt = datetime.now().strftime('%Y%m%d')
        start_dt = (datetime.now() - timedelta(days=120)).strftime('%Y%m%d')
        kdf = ak.stock_zh_a_hist(
            symbol=code, period="daily",
            start_date=start_dt, end_date=end_dt, adjust="qfq"
        )
        if kdf is None or len(kdf) < 30:
            return code, {}

        C = kdf['收盘'].astype(float)
        H = kdf['最高'].astype(float)
        L = kdf['最低'].astype(float)
        V = kdf['成交量'].astype(float)

        ma5 = _ma(C, 5)
        ma10 = _ma(C, 10)
        ma20 = _ma(C, 20)
        dif, dea, hist = _macd(C)
        vol_ma5 = _ma(V, 5)

        c = C.iloc[-1]
        m5, m10, m20 = ma5.iloc[-1], ma10.iloc[-1], ma20.iloc[-1]

        sig = {}

        # ① 均线多头排列
        sig['多头排列'] = bool(
            pd.notna(m20) and m5 > m10 > m20
        )
        # ② 站上均线
        sig['站上MA'] = bool(c > m5 and c > m10)
        # ③ MA5 拐头向上
        sig['MA5上行'] = bool(
            len(ma5) >= 3 and ma5.iloc[-1] > ma5.iloc[-2]
        )
        # ④ MACD 金叉（近3日内）
        gold = False
        for j in range(-3, 0):
            try:
                if dif.iloc[j] > dea.iloc[j] and dif.iloc[j - 1] <= dea.iloc[j - 1]:
                    gold = True
                    break
            except IndexError:
                pass
        sig['MACD金叉'] = gold
        # ⑤ MACD 零轴之上
        sig['MACD零上'] = bool(dif.iloc[-1] > 0)
        # ⑥ MACD 柱放大
        sig['MACD柱放大'] = bool(
            len(hist) >= 2 and hist.iloc[-1] > hist.iloc[-2]
        )
        # ⑦ 放量（今量 > 5日均量 × 1.3）
        sig['放量'] = bool(
            pd.notna(vol_ma5.iloc[-1]) and
            V.iloc[-1] > vol_ma5.iloc[-1] * 1.3
        )
        # ⑧ 突破近20日高点
        if len(H) >= 21:
            high20 = H.iloc[-21:-1].max()
            sig['突破前高'] = bool(c >= high20 * 0.98)
        else:
            sig['突破前高'] = False
        # ⑨ 连阳（近3日全红）
        if len(kdf) >= 3:
            sig['连续上涨'] = bool(
                (kdf['涨跌幅'].astype(float).tail(3) > 0).all()
            )
        else:
            sig['连续上涨'] = False
        # ⑩ 缩量回踩后放量启动
        if len(V) >= 6:
            vol_min3 = V.iloc[-4:-1].min()
            sig['缩量蓄势'] = bool(
                vol_min3 < vol_ma5.iloc[-4] * 0.7 and sig['放量']
            ) if pd.notna(vol_ma5.iloc[-4]) else False
        else:
            sig['缩量蓄势'] = False

        # 技术得分
        weights = {
            '多头排列': 8, '站上MA': 5, 'MA5上行': 4,
            'MACD金叉': 10, 'MACD零上': 4, 'MACD柱放大': 4,
            '放量': 6, '突破前高': 7, '连续上涨': 4, '缩量蓄势': 5,
        }
        sig['tech_score'] = sum(weights[k] for k, v in sig.items()
                                if isinstance(v, bool) and v)
        sig['tags'] = [k for k, v in sig.items()
                       if isinstance(v, bool) and v]

        return code, sig

    except Exception:
        return code, {}


def parallel_tech_analysis(candidates):
    """多线程 K 线分析"""
    codes = candidates['代码'].tolist()
    log(f"📈 K线技术分析 {len(codes)} 只 ({MAX_WORKERS_CPU}线程)")

    results = {}
    done_count = [0]

    with ThreadPoolExecutor(MAX_WORKERS_CPU, thread_name_prefix="TA") as pool:
        futs = {pool.submit(analyze_kline, c): c for c in codes}
        total = len(futs)
        for f in as_completed(futs):
            done_count[0] += 1
            try:
                code, sig = f.result()
                if sig:
                    results[code] = sig
            except Exception:
                pass
            if done_count[0] % 30 == 0 or done_count[0] == total:
                log(f"   分析进度 {done_count[0]}/{total}")

    log(f"   有效分析: {len(results)} 只")
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  第四层 · 四维综合评分
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def final_scoring(candidates, tech, zt_set, strong_set):
    log("⭐ 四维评分排序 ──────────────────────")

    rows = []
    for _, r in candidates.iterrows():
        code = r['代码']
        t = tech.get(code, {})
        score = 0
        tags = []

        # ─── A. 资金面 35分 ───
        mi = r['今日主力净流入']
        if   mi > 2e8:   score += 15; tags.append('主力强攻>2亿')
        elif mi > 1e8:   score += 13; tags.append('主力流入>1亿')
        elif mi > 5e7:   score += 10; tags.append('主力流入>5千万')
        elif mi > 1e7:   score += 6;  tags.append('主力流入>1千万')
        elif mi > 0:     score += 3

        f3 = r.get('3日主力净流入', 0)
        if f3 > 0:
            score += 8;  tags.append('3日持续流入')
        elif f3 > -mi * 0.3:
            score += 3   # 近3日虽略有流出但今日强势扭转

        f5 = r.get('5日主力净流入', 0)
        if f5 > 0:
            score += 5;  tags.append('5日持续流入')

        if r.get('超大单净流入', 0) > 0:
            score += 4;  tags.append('超大单介入')

        pct = r.get('主力净占比', 0)
        if pct > 15:
            score += 3;  tags.append(f'净占比{pct:.0f}%')

        # ─── B. 技术面 35分 ───
        ts = t.get('tech_score', 0)
        score += min(ts, 35)
        tags.extend(t.get('tags', []))

        # ─── C. 量价配合 15分 ───
        tr = r['换手率']
        if   5 <= tr <= 15:  score += 6
        elif 3 <= tr <= 25:  score += 3

        vr = r.get('量比', 0)
        if   vr >= 3:   score += 5; tags.append(f'量比{vr:.1f}')
        elif vr >= 1.5:  score += 3

        tail = r.get('尾盘强度', 50)
        if   tail >= 85: score += 4; tags.append('强势封盘')
        elif tail >= 65: score += 2

        # ─── D. 市场热度 15分 ───
        if code in strong_set:
            score += 6; tags.append('强势股池')
        if code in zt_set:
            score += 5; tags.append('涨停概念')

        chg = r['涨跌幅']
        if   3 <= chg <= 7:  score += 4
        elif 0 <= chg <= 3:  score += 2

        rows.append({
            **r.to_dict(),
            '综合评分': min(round(score, 1), 100),
            '技术评分': ts,
            '信号标签': tags[:10],
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result = result.nlargest(TOP_N, '综合评分').reset_index(drop=True)
    log(f"   最终入选: {len(result)} 只")
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  第五层 · 输出 & 推送
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _fmt(v):
    if abs(v) >= 1e8:  return f"{v / 1e8:.2f}亿"
    if abs(v) >= 1e4:  return f"{v / 1e4:.0f}万"
    return f"{v:.0f}"


def _sc_color(s):
    if s >= 75: return "#e74c3c"
    if s >= 60: return "#f39c12"
    if s >= 45: return "#3498db"
    return "#95a5a6"


def _sc_label(s):
    if s >= 80: return "🔥极强"
    if s >= 70: return "🔴强势"
    if s >= 60: return "🟠较强"
    if s >= 45: return "🟡关注"
    return "🔵一般"


def console_print(df):
    if df.empty:
        print("\n  📭  无符合条件的股票\n")
        return
    w = 100
    print(f"\n{'━' * w}")
    print(f" {'#':>2}  {'代码':>8}  {'名称':<6}  {'涨幅':>7}  {'换手':>6}  "
          f"{'量比':>5}  {'今日主力':>10}  {'3日主力':>10}  "
          f"{'评分':>5}  信号")
    print(f"{'━' * w}")
    for i, r in df.iterrows():
        tags = ' '.join(r.get('信号标签', [])[:4])
        print(f" {i + 1:>2}  {r['代码']:>8}  {r['名称']:<6}  "
              f"{r['涨跌幅']:>+6.2f}%  {r['换手率']:>5.2f}%  "
              f"{r.get('量比', 0):>5.2f}  {_fmt(r['今日主力净流入']):>10}  "
              f"{_fmt(r.get('3日主力净流入', 0)):>10}  "
              f"{r['综合评分']:>5.1f}  {tags}")
    print(f"{'━' * w}\n")


def build_html(df, hot_sectors, stats):
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    sec_str = '、'.join(hot_sectors[:6]) if hot_sectors else '暂无数据'

    html = f"""
    <div style="font-family:-apple-system,Helvetica,Arial,sans-serif;
                max-width:520px;margin:0 auto;padding:10px;">

      <!-- 标题 -->
      <div style="text-align:center;padding:12px 0 6px;">
        <h2 style="color:#e74c3c;margin:0;">🎯 主力追踪选股</h2>
        <p style="color:#bbb;font-size:10px;margin:3px 0 0;">
          {now} · 扫描 {stats['total']} 只 · 预测未来2日走势
        </p>
      </div>

      <!-- 统计条 -->
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:5px;margin:8px 0;">
        <div style="background:#f0f7ff;border-radius:8px;padding:7px;text-align:center;">
          <div style="font-size:9px;color:#999;">数据采集</div>
          <b style="color:#3498db;font-size:14px;">{stats['t1']:.1f}s</b>
        </div>
        <div style="background:#fff8e1;border-radius:8px;padding:7px;text-align:center;">
          <div style="font-size:9px;color:#999;">资金初筛</div>
          <b style="color:#f39c12;font-size:14px;">{stats['t2']:.1f}s</b>
        </div>
        <div style="background:#fce4ec;border-radius:8px;padding:7px;text-align:center;">
          <div style="font-size:9px;color:#999;">K线分析</div>
          <b style="color:#e74c3c;font-size:14px;">{stats['t3']:.1f}s</b>
        </div>
        <div style="background:#e8f5e9;border-radius:8px;padding:7px;text-align:center;">
          <div style="font-size:9px;color:#999;">入选</div>
          <b style="color:#27ae60;font-size:14px;">{len(df)}只</b>
        </div>
      </div>

      <!-- 热门板块 -->
      <div style="background:#eef2ff;border-radius:8px;padding:7px 12px;
                  font-size:11px;color:#5b7def;margin-bottom:10px;">
        🔥 资金热门板块: <b>{sec_str}</b>
      </div>
    """

    if df.empty:
        html += '<p style="text-align:center;color:#999;padding:30px;">📭 今日无符合条件的股票</p>'
    else:
        for i, r in df.iterrows():
            sc = r['综合评分']
            sc_c = _sc_color(sc)
            sc_l = _sc_label(sc)
            mi = r['今日主力净流入']
            f3 = r.get('3日主力净流入', 0)
            f5 = r.get('5日主力净流入', 0)
            mi_c = "#e74c3c" if mi > 0 else "#27ae60"
            f3_c = "#e74c3c" if f3 > 0 else "#27ae60"
            f5_c = "#e74c3c" if f5 > 0 else "#27ae60"
            amt = _fmt(r['成交额'])
            mcap = _fmt(r.get('流通市值', 0)) if r.get('流通市值', 0) else '-'
            chg = r['涨跌幅']
            chg_c = "#e74c3c" if chg >= 0 else "#27ae60"
            chg_s = f"+{chg:.2f}" if chg >= 0 else f"{chg:.2f}"
            bar_w = min(sc, 100)

            tags = r.get('信号标签', [])
            tags_html = ''.join(
                f'<span style="display:inline-block;background:#fff3e0;color:#e65100;'
                f'border-radius:3px;padding:1px 5px;font-size:9px;margin:1px 2px;">'
                f'{t}</span>' for t in tags[:6]
            ) or '<span style="color:#ddd;font-size:10px;">-</span>'

            html += f"""
      <!-- 卡片 #{i + 1} -->
      <div style="border:1px solid #f0f0f0;border-left:5px solid {sc_c};
                  border-radius:12px;padding:14px 16px;margin:10px 0;
                  background:#fff;box-shadow:0 2px 10px rgba(0,0,0,0.04);">

        <!-- 头部 -->
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <div>
            <span style="background:{sc_c};color:#fff;border-radius:4px;
                         padding:2px 8px;font-size:11px;font-weight:bold;">#{i + 1}</span>
            <b style="font-size:17px;margin-left:6px;">{r['名称']}</b>
            <span style="color:#bbb;font-size:11px;margin-left:4px;">{r['代码']}</span>
          </div>
          <div style="text-align:right;">
            <div style="font-size:22px;font-weight:bold;color:{chg_c};">{chg_s}%</div>
            <div style="font-size:11px;color:{sc_c};font-weight:bold;">
              {sc_l} {sc}分
            </div>
          </div>
        </div>

        <!-- 评分条 -->
        <div style="margin:8px 0 4px;background:#f5f5f5;border-radius:4px;height:5px;">
          <div style="width:{bar_w}%;height:100%;border-radius:4px;
                      background:linear-gradient(90deg,{sc_c},#f39c12);"></div>
        </div>

        <!-- 基本指标 -->
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;
                    font-size:11px;color:#999;margin:6px 0;">
          <div>现价 <b style="color:#333;">{r['最新价']:.2f}</b></div>
          <div>换手 <b style="color:#333;">{r['换手率']:.2f}%</b></div>
          <div>量比 <b style="color:#333;">{r.get('量比', 0):.2f}</b></div>
          <div>振幅 <b style="color:#333;">{r['振幅']:.2f}%</b></div>
          <div>尾盘 <b style="color:#333;">{r.get('尾盘强度', 0):.0f}%</b></div>
          <div>成交 <b style="color:#333;">{amt}</b></div>
        </div>

        <!-- 资金流向三栏 -->
        <div style="background:#fafbfc;border-radius:8px;padding:8px;margin:8px 0;
                    display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;text-align:center;">
          <div>
            <div style="font-size:9px;color:#aaa;">今日主力</div>
            <b style="font-size:13px;color:{mi_c};">{_fmt(mi)}</b>
          </div>
          <div>
            <div style="font-size:9px;color:#aaa;">3日累计</div>
            <b style="font-size:13px;color:{f3_c};">{_fmt(f3)}</b>
          </div>
          <div>
            <div style="font-size:9px;color:#aaa;">5日累计</div>
            <b style="font-size:13px;color:{f5_c};">{_fmt(f5)}</b>
          </div>
        </div>

        <!-- 信号标签 -->
        <div style="margin-top:4px;">{tags_html}</div>
      </div>
            """

    html += """
      <!-- 评分说明 -->
      <div style="background:#f8f9fa;border-radius:8px;padding:10px 12px;
                  font-size:10px;color:#999;margin-top:10px;line-height:1.6;">
        <b>📊 评分体系 (满分100)</b><br>
        A. 资金面 35分: 今日/3日/5日主力净流入 + 超大单 + 净占比<br>
        B. 技术面 35分: 均线多头 · MACD金叉 · 放量突破 · 连阳<br>
        C. 量价面 15分: 换手率 · 量比 · 尾盘强度<br>
        D. 热度面 15分: 强势股池 · 涨停概念 · 涨幅位置
      </div>

      <p style="text-align:center;color:#ccc;font-size:9px;margin-top:14px;">
        ⚠️ 以上仅为量化筛选参考，不构成任何投资建议<br>
        请结合大盘环境、板块轮动、个股基本面综合判断
      </p>
    </div>"""
    return html


def pushplus_send(title, content):
    log("📱 PushPlus 推送中...")
    try:
        r = requests.post("http://www.pushplus.plus/send", json={
            "token": PUSHPLUS_TOKEN,
            "title": title,
            "content": content,
            "template": "html",
        }, timeout=30).json()
        if r.get("code") == 200:
            log("✅ 推送成功！请查看手机")
        else:
            log(f"❌ 推送失败: {r.get('msg')}")
    except Exception as e:
        log(f"❌ 推送异常: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  主入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print(f"""
    ╔═════════════════════════════════════════════════════╗
    ║         🎯  主力追踪选股系统  v3.0                  ║
    ║  采集线程 {MAX_WORKERS_IO}  ·  分析线程 {MAX_WORKERS_CPU}  ·  推送 TOP{TOP_N}          ║
    ╚═════════════════════════════════════════════════════╝
    """)
    T0 = time.time()

    # ─ 阶段 1 ─ 并行采集
    t1 = time.time()
    data = parallel_fetch()
    t1 = time.time() - t1

    if data.get('quotes') is None or data['quotes'].empty:
        log("❌ 无行情数据"); return

    # ─ 阶段 2 ─ 资金面初筛
    t2 = time.time()
    candidates = primary_screen(data)
    t2 = time.time() - t2

    if candidates.empty:
        log("📭 初筛无结果"); return

    # ─ 阶段 3 ─ 多线程 K 线分析
    t3 = time.time()
    tech = parallel_tech_analysis(candidates)
    t3 = time.time() - t3

    # ─ 阶段 4 ─ 综合评分
    zt = data.get('zt_pool') or set()
    st = data.get('strong') or set()
    result = final_scoring(candidates, tech, zt, st)

    # ─ 阶段 5 ─ 输出推送
    console_print(result)

    stats = dict(total=len(data['quotes']), t1=t1, t2=t2, t3=t3)
    sectors = data.get('sectors') or []
    html = build_html(result, sectors, stats)

    today = datetime.now().strftime("%m-%d")
    title = f"🎯 主力追踪选股 {len(result)}只候选-[{today}]"
    pushplus_send(title, html)

    total = time.time() - T0
    print(f"""
    ┌─────────── ⏱ 耗时报告 ───────────┐
    │  数据采集     {t1:>6.1f}s  ({MAX_WORKERS_IO}线程)    │
    │  资金初筛     {t2:>6.1f}s               │
    │  K线分析      {t3:>6.1f}s  ({MAX_WORKERS_CPU}线程)    │
    │  ─────────────────────────────── │
    │  总耗时       {total:>6.1f}s               │
    └──────────────────────────────────┘
    """)


if __name__ == "__main__":
    main()

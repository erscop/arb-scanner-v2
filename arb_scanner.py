import requests, json, re, hashlib, os, csv
from datetime import datetime, timezone
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── CONFIGURAZIONE ───────────────────────────────────────────────────────────
NTFY_TOPIC           = 'arb_erscop_83041'

MIN_ARB_EDGE         = 0.10
MAX_ARB_EDGE         = 0.30
MIN_EV_EDGE          = 0.12
SIMILARITY_THRESHOLD = 0.90

POLY_FEE             = 0.02
PREDICTIT_FEE        = 0.10
KALSHI_FEE           = 0.07
KALSHI_MIN_VOLUME    = 2000

SEEN_FILE            = 'seen_alerts.json'
COOLDOWN_HOURS       = 4
LOG_FILE             = 'signals_log.csv'
LOG_HEADERS          = [
    'timestamp_utc', 'signal_type', 'edge_pct', 'similarity',
    'platform_a', 'platform_b', 'category_a', 'category_b',
    'title_a', 'title_b', 'url_a', 'url_b',
    'profit_per_1', 'sent_ntfy', 'skip_reason'
]

KALSHI_CATEGORY_MAP = {
    'kxmlb'     : 'baseball',
    'kxnba'     : 'basketball',
    'kxnfl'     : 'football',
    'kxnhl'     : 'hockey',
    'kxwta'     : 'tennis',
    'kxatp'     : 'tennis',
    'kxhigh'    : 'weather',
    'kxlowt'    : 'weather',
    'kxprecip'  : 'weather',
    'kxtruev'   : 'politics',
    'kxaaagasd' : 'politics',
}

STOP_WORDS = [
    'will','the','a','an','be','is','are','was','were','in','on','at','to',
    'for','of','and','or','by','with','from','that','this','it','its','as',
    'have','has','had','do','does','did','before','after','than','more',
    'less','over','under','above','below','between','during','through',
    'what','which','who','how','when','where','why','yes','no','not',
    'any','all','both','end','first','last','next','per','each','would',
    'could','should','may','might','get','make','take','least','most',
    'new','old','big','high','low','win','lose','pass','fail','hit','about'
]

# ─── CATEGORY FILTER ─────────────────────────────────────────────────────────
def get_market_category(url: str, title: str = '') -> str:
    url_l   = url.lower()
    title_l = title.lower()
    m = re.search(r'kalshi\.com/markets/([^/\s]+)', url_l)
    if m:
        slug = m.group(1)
        for prefix, cat in KALSHI_CATEGORY_MAP.items():
            if slug.startswith(prefix):
                return cat
    combined = url_l + ' ' + title_l
    if any(k in combined for k in ['bitcoin','btc','eth','crypto','price','dollar','usd']):
        return 'crypto'
    if any(k in combined for k in ['election','senate','house','mayor','party','democrat',
                                   'republican','vote','congress','president','governor']):
        return 'politics'
    if any(k in combined for k in ['temperature','rain','weather','humid','forecast',
                                   'celsius','fahrenheit','highs','lows']):
        return 'weather'
    if any(k in combined for k in ['wta','atp','tennis','soccer','nba','nfl','mlb',
                                   'nhl','run','score','match','game','league','player']):
        return 'sports'
    return 'unknown'

def categories_compatible(cat_a: str, cat_b: str) -> bool:
    if 'unknown' in (cat_a, cat_b):
        return True
    return cat_a == cat_b

# ─── DEDUP / SEEN ─────────────────────────────────────────────────────────────
def load_seen():
    if os.path.exists(SEEN_FILE):
        try:
            return json.load(open(SEEN_FILE))
        except Exception:
            return {}
    return {}

def save_seen(seen):
    json.dump(seen, open(SEEN_FILE, 'w'))

def make_key(tipo, label, title_a, title_b=''):
    raw = f"{tipo}|{label}|{title_a[:60]}|{title_b[:60]}"
    return hashlib.md5(raw.encode()).hexdigest()

def already_seen(seen, key):
    if key not in seen:
        return False
    last    = datetime.fromisoformat(seen[key])
    elapsed = (datetime.now(timezone.utc) - last).total_seconds()
    return elapsed < COOLDOWN_HOURS * 3600

def mark_seen(seen, key):
    seen[key] = datetime.now(timezone.utc).isoformat()

# ─── LOGGING CSV ──────────────────────────────────────────────────────────────
def log_signal(signal_type, edge_pct, similarity,
               platform_a, platform_b, category_a, category_b,
               title_a, title_b, url_a, url_b,
               profit_per_1, sent_ntfy, skip_reason=''):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=LOG_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'timestamp_utc' : datetime.now(timezone.utc).isoformat(),
            'signal_type'   : signal_type,
            'edge_pct'      : round(edge_pct, 4),
            'similarity'    : round(similarity, 4),
            'platform_a'    : platform_a,
            'platform_b'    : platform_b,
            'category_a'    : category_a,
            'category_b'    : category_b,
            'title_a'       : title_a.replace('\n', ' ')[:120],
            'title_b'       : title_b.replace('\n', ' ')[:120],
            'url_a'         : url_a,
            'url_b'         : url_b,
            'profit_per_1'  : round(profit_per_1, 4),
            'sent_ntfy'     : sent_ntfy,
            'skip_reason'   : skip_reason,
        })

# ─── NLP UTILS ────────────────────────────────────────────────────────────────
def clean(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_vecs(mlist):
    vec = TfidfVectorizer(stop_words=STOP_WORDS, ngram_range=(1,2),
                          min_df=1, sublinear_tf=True)
    v   = vec.fit_transform([m['clean'] for m in mlist])
    return vec, v

def best_match(src, tlist, vec, tvecs):
    sv   = vec.transform([src['clean']])
    sims = cosine_similarity(sv, tvecs)[0]
    j    = int(np.argmax(sims))
    return tlist[j], float(sims[j])

# ─── FETCH PIATTAFORME ────────────────────────────────────────────────────────
def get_polymarket():
    try:
        result, seen = [], set()
        for offset in [0, 200, 400]:
            r = requests.get(
                'https://gamma-api.polymarket.com/events',
                params={'active':'true','closed':'false','limit':200,'offset':offset,
                        'order':'id','ascending':'false'},
                timeout=15
            )
            events = r.json()
            if not events:
                break
            for ev in events:
                for m in ev.get('markets', []):
                    mid = m.get('id')
                    if mid in seen:
                        continue
                    seen.add(mid)
                    try:
                        prices = json.loads(m.get('outcomePrices', '[]'))
                        if len(prices) < 2:
                            continue
                        yes = float(prices[0])
                        no  = float(prices[1])
                        liq = float(m.get('liquidity', 0))
                        if liq >= 500 and 0.03 < yes < 0.97:
                            slug = ev.get('slug') or str(ev.get('id', ''))
                            url  = 'https://polymarket.com/event/' + slug
                            q    = m.get('question', '')
                            result.append({
                                'source'   : 'polymarket',
                                'title'    : q,
                                'clean'    : clean(q),
                                'yes'      : yes, 'no': no, 'liquidity': liq,
                                'fee'      : POLY_FEE,
                                'url'      : url,
                                'category' : get_market_category(url, q),
                            })
                    except Exception:
                        continue
        print(f'  [DEBUG] Poly usable: {len(result)}')
        return result
    except Exception as e:
        print(f'[Polymarket ERROR] {e}')
        return []

def get_predictit():
    try:
        r    = requests.get('https://www.predictit.org/api/marketdata/all/', timeout=15)
        data = r.json().get('markets', [])
        result = []
        for m in data:
            mid   = m.get('id')
            title = m.get('name', '')
            for c in m.get('contracts', []):
                yp  = c.get('bestBuyYesCost')
                np_ = c.get('bestBuyNoCost')
                if yp is None or np_ is None:
                    continue
                try:
                    yes = float(yp)
                    no  = float(np_)
                except Exception:
                    continue
                if yes <= 0.05 or no <= 0.05:
                    continue
                yes = min(yes, 1.0)
                no  = min(no, 1.0)
                ft  = title + ' - ' + c.get('name', '')
                url = 'https://www.predictit.org/markets/detail/' + str(mid)
                result.append({
                    'source'   : 'predictit',
                    'title'    : ft,
                    'clean'    : clean(ft),
                    'yes'      : yes, 'no': no,
                    'liquidity': float(c.get('sharesTraded', 0)),
                    'fee'      : PREDICTIT_FEE,
                    'url'      : url,
                    'category' : get_market_category(url, ft),
                })
        print(f'  [DEBUG] PredictIt usable: {len(result)}')
        return result
    except Exception as e:
        print(f'[PredictIt ERROR] {e}')
        return []

def get_kalshi():
    try:
        result, cursor = [], None
        for _ in range(5):
            params = {'limit': 1000, 'status': 'open', 'mve_filter': 'exclude'}
            if cursor:
                params['cursor'] = cursor
            r    = requests.get(
                'https://external-api.kalshi.com/trade-api/v2/markets',
                params=params, timeout=15
            )
            data    = r.json()
            markets = data.get('markets', [])
            if not markets:
                break
            for m in markets:
                if m.get('market_type') != 'binary':
                    continue
                bid_raw = m.get('yes_bid_dollars')
                ask_raw = m.get('yes_ask_dollars')
                if bid_raw is None or ask_raw is None:
                    continue
                try:
                    bid = float(bid_raw)
                    ask = float(ask_raw)
                    yes = round((bid + ask) / 2.0, 4)
                    no  = round(1.0 - yes, 4)
                except Exception:
                    continue
                if not (0.03 < yes < 0.97):
                    continue
                try:
                    vol = float(m.get('volume_fp', 0))
                except Exception:
                    vol = 0.0
                if vol < KALSHI_MIN_VOLUME:
                    continue
                title        = m.get('yes_sub_title', '') or m.get('title', '')
                event_ticker = m.get('event_ticker', '')
                url          = 'https://kalshi.com/markets/' + event_ticker.lower()
                result.append({
                    'source'   : 'kalshi',
                    'title'    : title,
                    'clean'    : clean(title),
                    'yes'      : yes, 'no': no, 'liquidity': vol,
                    'fee'      : KALSHI_FEE,
                    'url'      : url,
                    'category' : get_market_category(url, title),
                })
            cursor = data.get('cursor')
            if not cursor:
                break
        print(f'  [DEBUG] Kalshi usable: {len(result)}')
        return result
    except Exception as e:
        print(f'[Kalshi ERROR] {e}')
        return []

def get_zeitgeist():
    print('  [DEBUG] Zeitgeist usable: 0 (non configurato)')
    return []

def get_manifold():
    try:
        r    = requests.get(
            'https://api.manifold.markets/v0/search-markets',
            params={'term':'','filter':'open','contractType':'BINARY',
                    'sort':'liquidity','limit':500},
            timeout=15
        )
        data = r.json()
        if isinstance(data, dict):
            data = data.get('markets', data.get('data', []))
        if not isinstance(data, list):
            data = []
        print(f'  [DEBUG] Manifold raw items: {len(data)}')
        result = []
        for m in data:
            if not isinstance(m, dict):
                continue
            if m.get('outcomeType') != 'BINARY':
                continue
            if m.get('isResolved', False):
                continue
            prob = m.get('probability')
            if prob is None:
                continue
            try:
                prob = float(prob)
            except Exception:
                continue
            if 0.0 < prob < 1.0:
                result.append({
                    'source': 'manifold',
                    'title' : m.get('question', ''),
                    'clean' : clean(m.get('question', '')),
                    'yes'   : prob, 'no': 1 - prob,
                    'url'   : m.get('url', 'https://manifold.markets')
                })
        print(f'  [DEBUG] Manifold usable: {len(result)}')
        return result
    except Exception as e:
        print(f'[Manifold ERROR] {e}')
        return []

# ─── RILEVAMENTO OPPORTUNITÀ ──────────────────────────────────────────────────
def cross_exchange_arbs(ma, mb, fa, fb):
    if not ma or not mb:
        return []
    net_fee     = fa + fb
    vecb, vecsb = build_vecs(mb)
    found, seen = [], set()
    for a in ma:
        b, score = best_match(a, mb, vecb, vecsb)
        if score < SIMILARITY_THRESHOLD:
            continue
        cat_a = a.get('category', 'unknown')
        cat_b = b.get('category', 'unknown')
        if not categories_compatible(cat_a, cat_b):
            continue
        key = (a['url'], b['url'])
        if key in seen:
            continue
        seen.add(key)
        combos = [
            ('YES ' + a['source'] + ' + NO ' + b['source'],
             a['yes'] + b['no'],
             'BUY YES@' + str(round(a['yes'],2)) + ' ' + a['source'],
             'BUY NO@'  + str(round(b['no'],2))  + ' ' + b['source']),
            ('NO ' + a['source'] + ' + YES ' + b['source'],
             a['no'] + b['yes'],
             'BUY NO@'  + str(round(a['no'],2))  + ' ' + a['source'],
             'BUY YES@' + str(round(b['yes'],2)) + ' ' + b['source']),
        ]
        for label, cost, sA, sB in combos:
            edge = (1 - cost) - net_fee
            if MIN_ARB_EDGE < edge <= MAX_ARB_EDGE:
                found.append({
                    'type'   : 'ARB', 'label': label,
                    'edge'   : round(edge*100, 2), 'cost': round(cost,4),
                    'profit' : round(1-cost,4), 'sA': sA, 'sB': sB,
                    'score'  : round(score,3),
                    'title_a': a['title'][:80], 'title_b': b['title'][:80],
                    'url_a'  : a['url'], 'url_b': b['url'],
                    'cat_a'  : cat_a, 'cat_b': cat_b,
                    'plat_a' : a['source'], 'plat_b': b['source'],
                })
    return found

def find_ladder_opportunities(markets):
    ladders   = []
    by_source = {}
    for m in markets:
        by_source.setdefault(m['source'], []).append(m)
    pattern = re.compile(r'(over|under|above|below|greater than|less than)\s+(\d+\.?\d*)')
    for src, mks in by_source.items():
        buckets = {}
        for m in mks:
            match = pattern.search(m['title'].lower())
            if not match:
                continue
            direction = match.group(1)
            level     = float(match.group(2))
            base      = pattern.sub('', m['title'].lower())
            key       = (src, clean(base).strip(), direction)
            buckets.setdefault(key, []).append((level, m))
        for key, lst in buckets.items():
            if len(lst) < 2:
                continue
            lst.sort(key=lambda x: x[0])
            probs_y    = [m['yes'] for _, m in lst]
            decreasing = key[2] in ['over', 'above', 'greater than']
            violations = []
            for i in range(len(probs_y) - 1):
                if decreasing and probs_y[i+1] > probs_y[i] + 0.03:
                    violations.append((lst[i], lst[i+1]))
                if not decreasing and probs_y[i+1] < probs_y[i] - 0.03:
                    violations.append((lst[i], lst[i+1]))
            for (lv1, m1), (lv2, m2) in violations:
                ladders.append({
                    'type'   : 'LADDER', 'source': src, 'base': key[1],
                    'level_1': lv1, 'level_2': lv2,
                    'p1'     : m1['yes'], 'p2': m2['yes'],
                    'title_1': m1['title'], 'title_2': m2['title'],
                    'url_1'  : m1['url'], 'url_2': m2['url']
                })
    return ladders

def find_correlated_pairs(markets):
    if not markets:
        return []
    MIN_CORRELATED_DIFF = 0.60
    by_source = {}
    for m in markets:
        by_source.setdefault(m['source'], []).append(m)
    correlated = []
    for src, mks in by_source.items():
        if len(mks) < 3:
            continue
        vec, vecs = build_vecs(mks)
        n = len(mks)
        for i in range(n):
            mi   = mks[i]
            sims = cosine_similarity(vec.transform([mi['clean']]), vecs)[0]
            for j in range(i+1, n):
                score = float(sims[j])
                if score < SIMILARITY_THRESHOLD:
                    continue
                mj   = mks[j]
                diff = abs(mi['yes'] - mj['yes'])
                if diff < MIN_CORRELATED_DIFF:
                    continue
                correlated.append({
                    'type'   : 'CORRELATED', 'source': src,
                    'score'  : round(score,3), 'diff': round(diff*100,1),
                    'title_a': mi['title'], 'title_b': mj['title'],
                    'p_yes_a': mi['yes'], 'p_yes_b': mj['yes'],
                    'url_a'  : mi['url'], 'url_b': mj['url']
                })
    return correlated

def find_ev_signals(rlist, mlist, platform):
    if not rlist or not mlist:
        return []
    vecm, vecsm   = build_vecs(mlist)
    signals, seen = [], set()
    for r in rlist:
        m, score = best_match(r, mlist, vecm, vecsm)
        if score < SIMILARITY_THRESHOLD:
            continue
        key = r['url']
        if key in seen:
            continue
        seen.add(key)
        div = m['yes'] - r['yes']
        if abs(div) >= MIN_EV_EDGE:
            direction = 'YES' if div > 0 else 'NO'
            signals.append({
                'type'      : '+EV', 'platform': platform, 'direction': direction,
                'edge'      : round(abs(div)*100, 2),
                'real_prob' : r['yes'] if direction == 'YES' else r['no'],
                'mani_prob' : m['yes'] if direction == 'YES' else m['no'],
                'score'     : round(score,3),
                'title_real': r['title'][:80], 'title_mani': m['title'][:80],
                'url_real'  : r['url'], 'url_mani': m['url']
            })
    return signals

# ─── NTFY ─────────────────────────────────────────────────────────────────────
def send_ntfy(title, message, priority='default'):
    try:
        r = requests.post(
            'https://ntfy.sh/' + NTFY_TOPIC,
            data=message.encode('utf-8'),
            headers={'Title': title, 'Priority': priority},
            timeout=10
        )
        print(f'  -> ntfy: {r.status_code}')
    except Exception as e:
        print(f'  [ntfy ERROR] {e}')

# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('[' + datetime.utcnow().strftime('%H:%M:%S') + ' UTC] Scansione in corso...')

    seen = load_seen()

    poly      = get_polymarket()
    predictit = get_predictit()
    kalshi    = get_kalshi()
    zeitgeist = get_zeitgeist()
    manifold  = get_manifold()
    all_markets = poly + predictit + kalshi + zeitgeist

    print(f'  -> Poly:{len(poly)} | PredictIt:{len(predictit)} | '
          f'Kalshi:{len(kalshi)} | Manifold:{len(manifold)}')

    arbs_poly_pi     = cross_exchange_arbs(poly,      predictit, POLY_FEE,      PREDICTIT_FEE)
    arbs_poly_kalshi = cross_exchange_arbs(poly,      kalshi,    POLY_FEE,      KALSHI_FEE)
    arbs_pi_kalshi   = cross_exchange_arbs(predictit, kalshi,    PREDICTIT_FEE, KALSHI_FEE)
    all_arbs         = arbs_poly_pi + arbs_poly_kalshi + arbs_pi_kalshi

    ladder_ops = find_ladder_opportunities(all_markets)
    corr_pairs = find_correlated_pairs(all_markets)
    ev_poly    = find_ev_signals(poly,      manifold, 'Polymarket')
    ev_pi      = find_ev_signals(predictit, manifold, 'PredictIt')
    ev_kalshi  = find_ev_signals(kalshi,    manifold, 'Kalshi')
    ev_all     = ev_poly + ev_pi + ev_kalshi

    print(f'  -> ARB:{len(all_arbs)} | LADDER:{len(ladder_ops)} | '
          f'CORRELATED:{len(corr_pairs)} | +EV:{len(ev_all)}')

    arbs_sorted   = sorted(all_arbs,   key=lambda x: -x['edge'])[:3]
    corr_sorted   = sorted(corr_pairs, key=lambda x: -x['diff'])[:1]
    ladder_sorted = ladder_ops[:1]
    ev_sorted     = sorted(ev_all,     key=lambda x: -x['edge'])[:1]

    sent = 0

    for a in arbs_sorted:
        k = make_key('ARB', a['label'], a['title_a'], a['title_b'])
        if already_seen(seen, k):
            print(f'  -> [SKIP dedup] ARB {a["label"]}')
            log_signal('ARB', a['edge']/100, a['score'],
                       a['plat_a'], a['plat_b'], a['cat_a'], a['cat_b'],
                       a['title_a'], a['title_b'], a['url_a'], a['url_b'],
                       a['profit'], sent_ntfy=False, skip_reason='cooldown')
            continue
        msg = (a['sA'] + '\n' + a['sB'] + '\n' +
               'Costo: $' + str(a['cost']) + ' | Profitto: $' + str(a['profit']) + ' per $1\n' +
               'Similarity: ' + str(a['score']) + '\n' +
               a['title_a'] + '\n' + a['title_b'] + '\n' +
               a['url_a'] + '\n' + a['url_b'])
        send_ntfy('ARB +' + str(a['edge']) + '% | ' + a['label'], msg, priority='urgent')
        log_signal('ARB', a['edge']/100, a['score'],
                   a['plat_a'], a['plat_b'], a['cat_a'], a['cat_b'],
                   a['title_a'], a['title_b'], a['url_a'], a['url_b'],
                   a['profit'], sent_ntfy=True)
        mark_seen(seen, k)
        sent += 1

    for l in ladder_sorted:
        k = make_key('LADDER', l['source'], l['title_1'], l['title_2'])
        if already_seen(seen, k):
            print(f'  -> [SKIP dedup] LADDER {l["base"]}')
            log_signal('LADDER', abs(l['p2']-l['p1']), 1.0,
                       l['source'], l['source'], 'unknown', 'unknown',
                       l['title_1'], l['title_2'], l['url_1'], l['url_2'],
                       0.0, sent_ntfy=False, skip_reason='cooldown')
            continue
        msg = ('Exchange: ' + l['source'] + '\n' +
               'Base: ' + l['base'] + '\n' +
               'Livello ' + str(l['level_1']) + ' -> p=' + str(round(l['p1'],2)) + '\n' +
               'Livello ' + str(l['level_2']) + ' -> p=' + str(round(l['p2'],2)) + '\n' +
               l['title_1'] + '\n' + l['title_2'] + '\n' +
               l['url_1'] + '\n' + l['url_2'])
        send_ntfy('LADDER anomaly', msg, priority='high')
        log_signal('LADDER', abs(l['p2']-l['p1']), 1.0,
                   l['source'], l['source'], 'unknown', 'unknown',
                   l['title_1'], l['title_2'], l['url_1'], l['url_2'],
                   0.0, sent_ntfy=True)
        mark_seen(seen, k)
        sent += 1

    for c in corr_sorted:
        k = make_key('CORRELATED', c['source'], c['title_a'], c['title_b'])
        if already_seen(seen, k):
            print(f'  -> [SKIP dedup] CORRELATED {c["title_a"][:40]}')
            log_signal('CORRELATED', c['diff']/100, c['score'],
                       c['source'], c['source'], 'unknown', 'unknown',
                       c['title_a'], c['title_b'], c['url_a'], c['url_b'],
                       0.0, sent_ntfy=False, skip_reason='cooldown')
            continue
        msg = ('Exchange: ' + c['source'] + '\n' +
               'Similarity: ' + str(c['score']) + '\n' +
               'Diff YES: ' + str(c['diff']) + '%\n' +
               c['title_a'] + ' (p_yes=' + str(round(c['p_yes_a'],2)) + ')\n' +
               c['title_b'] + ' (p_yes=' + str(round(c['p_yes_b'],2)) + ')\n' +
               c['url_a'] + '\n' + c['url_b'])
        send_ntfy('Correlated spread', msg, priority='default')
        log_signal('CORRELATED', c['diff']/100, c['score'],
                   c['source'], c['source'], 'unknown', 'unknown',
                   c['title_a'], c['title_b'], c['url_a'], c['url_b'],
                   0.0, sent_ntfy=True)
        mark_seen(seen, k)
        sent += 1

    for s in ev_sorted:
        k = make_key('+EV', s['platform'] + s['direction'], s['title_real'], s['title_mani'])
        if already_seen(seen, k):
            print(f'  -> [SKIP dedup] +EV {s["title_real"][:40]}')
            log_signal('+EV', s['edge']/100, s['score'],
                       s['platform'], 'manifold', 'unknown', 'unknown',
                       s['title_real'], s['title_mani'], s['url_real'], s['url_mani'],
                       0.0, sent_ntfy=False, skip_reason='cooldown')
            continue
        msg = ('Piattaforma: ' + s['platform'] + '\n' +
               'Direzione: ' + s['direction'] + '\n' +
               'Prezzo attuale: ' + str(round(s['real_prob'],2)) +
               ' | Stima Manifold: ' + str(round(s['mani_prob'],2)) + '\n' +
               'Divergenza: +' + str(s['edge']) + '%\n' +
               'Mercato: ' + s['title_real'] + '\n' +
               'Manifold: ' + s['title_mani'] + '\n' +
               s['url_real'] + '\n' + s['url_mani'])
        send_ntfy('+EV ' + s['direction'] + ' +' + str(s['edge']) + '% su ' + s['platform'],
                  msg, priority='high')
        log_signal('+EV', s['edge']/100, s['score'],
                   s['platform'], 'manifold', 'unknown', 'unknown',
                   s['title_real'], s['title_mani'], s['url_real'], s['url_mani'],
                   0.0, sent_ntfy=True)
        mark_seen(seen, k)
        sent += 1

    save_seen(seen)
    print(f'  -> Notifiche inviate: {sent}')
    if sent == 0:
        print('  -> Nessuna nuova opportunita (tutto gia notificato o sotto soglia).')

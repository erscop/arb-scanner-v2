import requests, json, re, os
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 70)
print("ARB SCANNER - DEBUG COMPLETO")
print("Run: " + datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S') + " UTC")
print("=" * 70)

NTFY_TOPIC = 'arb_erscop_83041'
MIN_ARB_EDGE = 0.10
MIN_EV_EDGE = 0.15
SIMILARITY_THRESHOLD = 0.75
POLY_FEE = 0.02
PREDICTIT_FEE = 0.10
KALSHI_FEE = 0.07
KALSHI_MIN_VOLUME = 2000

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

def clean(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_vecs(mlist):
    vec = TfidfVectorizer(stop_words=STOP_WORDS, ngram_range=(1,2), min_df=1, sublinear_tf=True)
    v = vec.fit_transform([m['clean'] for m in mlist])
    return vec, v

def best_match(src, tlist, vec, tvecs):
    sv = vec.transform([src['clean']])
    sims = cosine_similarity(sv, tvecs)[0]
    j = int(np.argmax(sims))
    return tlist[j], float(sims[j])

print("\n[1/6] TEST CONNETTIVITA API")
print("-" * 50)
apis = {
    "Polymarket": "https://gamma-api.polymarket.com/events?active=true&closed=false&limit=1",
    "PredictIt":  "https://www.predictit.org/api/marketdata/all/",
    "Kalshi":     "https://external-api.kalshi.com/trade-api/v2/markets?limit=1&status=open",
    "Manifold":   "https://api.manifold.markets/v0/search-markets?term=&filter=open&contractType=BINARY&limit=1",
    "ntfy":       "https://ntfy.sh/" + NTFY_TOPIC + "/json?poll=1&since=1m",
}
for name, url in apis.items():
    try:
        r = requests.get(url, timeout=10)
        print("  " + ("OK  " if r.status_code == 200 else "ERR ") + name + ": HTTP " + str(r.status_code) + " | " + str(len(r.content)) + " bytes")
    except Exception as e:
        print("  FAIL " + name + ": " + str(e))

print("\n[2/6] FETCH & QUALITA DATI")
print("-" * 50)

def get_polymarket_debug():
    result, seen, raw_total, skipped = [], set(), 0, defaultdict(int)
    for offset in [0, 200, 400]:
        try:
            r = requests.get('https://gamma-api.polymarket.com/events',
                params={'active':'true','closed':'false','limit':200,'offset':offset,'order':'id','ascending':'false'}, timeout=15)
            events = r.json()
            if not events:
                break
            for ev in events:
                for m in ev.get('markets', []):
                    raw_total += 1
                    mid = m.get('id')
                    if mid in seen:
                        skipped['duplicate'] += 1
                        continue
                    seen.add(mid)
                    try:
                        prices = json.loads(m.get('outcomePrices', '[]'))
                        if len(prices) < 2:
                            skipped['no_prices'] += 1
                            continue
                        yes = float(prices[0])
                        liq = float(m.get('liquidity', 0))
                        if liq < 500:
                            skipped['low_liquidity'] += 1
                            continue
                        if not (0.03 < yes < 0.97):
                            skipped['extreme_price'] += 1
                            continue
                        slug = ev.get('slug') or str(ev.get('id', ''))
                        result.append({
                            'source': 'polymarket', 'title': m.get('question', ''),
                            'clean': clean(m.get('question', '')),
                            'yes': yes, 'no': float(prices[1]), 'liquidity': liq,
                            'fee': POLY_FEE, 'url': 'https://polymarket.com/event/' + slug
                        })
                    except Exception:
                        skipped['parse_error'] += 1
        except Exception as e:
            print("  [POLY ERROR offset=" + str(offset) + "] " + str(e))
    liqs = [m['liquidity'] for m in result]
    print("  Polymarket  | raw:" + str(raw_total) + " | usable:" + str(len(result)) + " | skip:" + str(dict(skipped)))
    if liqs:
        print("              | liq media:$" + str(round(sum(liqs)/len(liqs))) + " | max:$" + str(round(max(liqs))) + " | min:$" + str(round(min(liqs))))
    return result

def get_predictit_debug():
    result, skipped, raw_markets = [], defaultdict(int), 0
    try:
        r = requests.get('https://www.predictit.org/api/marketdata/all/', timeout=15)
        data = r.json().get('markets', [])
        raw_markets = len(data)
        for m in data:
            mid = m.get('id')
            title = m.get('name', '')
            for c in m.get('contracts', []):
                yp = c.get('bestBuyYesCost')
                np_ = c.get('bestBuyNoCost')
                if yp is None or np_ is None:
                    skipped['no_price'] += 1
                    continue
                try:
                    yes, no = float(yp), float(np_)
                except Exception:
                    skipped['parse_error'] += 1
                    continue
                if yes <= 0.05 or no <= 0.05:
                    skipped['extreme_price'] += 1
                    continue
                yes, no = min(yes, 1.0), min(no, 1.0)
                ft = title + ' - ' + c.get('name', '')
                result.append({
                    'source': 'predictit', 'title': ft, 'clean': clean(ft),
                    'yes': yes, 'no': no, 'liquidity': float(c.get('sharesTraded', 0)),
                    'fee': PREDICTIT_FEE, 'url': 'https://www.predictit.org/markets/detail/' + str(mid)
                })
    except Exception as e:
        print("  [PREDICTIT ERROR] " + str(e))
    yes_prices = [m['yes'] for m in result]
    print("  PredictIt   | raw_markets:" + str(raw_markets) + " | usable:" + str(len(result)) + " | skip:" + str(dict(skipped)))
    if yes_prices:
        print("              | yes medio:" + str(round(sum(yes_prices)/len(yes_prices),3)) + " | range:[" + str(round(min(yes_prices),2)) + "-" + str(round(max(yes_prices),2)) + "]")
    return result

def get_kalshi_debug():
    result, skipped = [], defaultdict(int)
    raw_total, cursor, pages = 0, None, 0
    try:
        for _ in range(5):
            pages += 1
            params = {'limit': 1000, 'status': 'open', 'mve_filter': 'exclude'}
            if cursor:
                params['cursor'] = cursor
            r = requests.get('https://external-api.kalshi.com/trade-api/v2/markets', params=params, timeout=15)
            data = r.json()
            markets = data.get('markets', [])
            if not markets:
                break
            raw_total += len(markets)
            for m in markets:
                if m.get('market_type') != 'binary':
                    skipped['not_binary'] += 1
                    continue
                bid_raw = m.get('yes_bid_dollars')
                ask_raw = m.get('yes_ask_dollars')
                if bid_raw is None or ask_raw is None:
                    skipped['no_price'] += 1
                    continue
                try:
                    bid, ask = float(bid_raw), float(ask_raw)
                    yes = round((bid + ask) / 2.0, 4)
                    no = round(1.0 - yes, 4)
                except Exception:
                    skipped['parse_error'] += 1
                    continue
                if not (0.03 < yes < 0.97):
                    skipped['extreme_price'] += 1
                    continue
                try:
                    vol = float(m.get('volume_fp', 0))
                except Exception:
                    vol = 0.0
                if vol < KALSHI_MIN_VOLUME:
                    skipped['low_volume'] += 1
                    continue
                title = m.get('yes_sub_title', '') or m.get('title', '')
                event_ticker = m.get('event_ticker', '')
                result.append({
                    'source': 'kalshi', 'title': title, 'clean': clean(title),
                    'yes': yes, 'no': no, 'liquidity': vol, 'fee': KALSHI_FEE,
                    'url': 'https://kalshi.com/markets/' + event_ticker.lower()
                })
            cursor = data.get('cursor')
            if not cursor:
                break
    except Exception as e:
        print("  [KALSHI ERROR] " + str(e))
    vols = [m['liquidity'] for m in result]
    print("  Kalshi      | raw:" + str(raw_total) + " | pages:" + str(pages) + " | usable:" + str(len(result)) + " | skip:" + str(dict(skipped)))
    if vols:
        print("              | vol medio:$" + str(round(sum(vols)/len(vols))) + " | max:$" + str(round(max(vols))))
    return result

def get_manifold_debug():
    result, skipped, data = [], defaultdict(int), []
    try:
        r = requests.get('https://api.manifold.markets/v0/search-markets',
            params={'term':'','filter':'open','contractType':'BINARY','sort':'liquidity','limit':500}, timeout=15)
        data = r.json()
        if isinstance(data, dict):
            data = data.get('markets', data.get('data', []))
        if not isinstance(data, list):
            data = []
        for m in data:
            if not isinstance(m, dict):
                skipped['not_dict'] += 1
                continue
            if m.get('outcomeType') != 'BINARY':
                skipped['not_binary'] += 1
                continue
            if m.get('isResolved', False):
                skipped['resolved'] += 1
                continue
            prob = m.get('probability')
            if prob is None:
                skipped['no_prob'] += 1
                continue
            try:
                prob = float(prob)
            except Exception:
                skipped['parse_error'] += 1
                continue
            if not (0.0 < prob < 1.0):
                skipped['extreme'] += 1
                continue
            result.append({
                'source': 'manifold', 'title': m.get('question', ''),
                'clean': clean(m.get('question', '')),
                'yes': prob, 'no': 1 - prob,
                'url': m.get('url', 'https://manifold.markets')
            })
    except Exception as e:
        print("  [MANIFOLD ERROR] " + str(e))
    print("  Manifold    | raw:" + str(len(data)) + " | usable:" + str(len(result)) + " | skip:" + str(dict(skipped)))
    return result

poly      = get_polymarket_debug()
predictit = get_predictit_debug()
kalshi    = get_kalshi_debug()
manifold  = get_manifold_debug()

print("\n[3/6] ANALISI MATCHING - FALSI POSITIVI")
print("-" * 50)

def debug_cross_arbs(ma, mb, fa, fb, label):
    if not ma or not mb:
        print("  " + label + ": lista vuota, skip")
        return []
    net_fee = fa + fb
    vecb, vecsb = build_vecs(mb)
    found, seen_keys = [], set()
    sim_dist = defaultdict(int)
    for a in ma:
        b, score = best_match(a, mb, vecb, vecsb)
        if score >= SIMILARITY_THRESHOLD:
            sim_dist[round(score, 1)] += 1
        if score < SIMILARITY_THRESHOLD:
            continue
        key = (a['source'], a['title'][:40], b['source'])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        combos = [
            ('YES '+a['source']+'+NO '+b['source'],  a['yes']+b['no'],  a['yes'], b['no']),
            ('NO ' +a['source']+'+YES '+b['source'], a['no'] +b['yes'], a['no'],  b['yes']),
        ]
        for lbl, cost, p_a, p_b in combos:
            edge = (1 - cost) - net_fee
            if edge > MIN_ARB_EDGE:
                found.append({
                    'label': lbl, 'edge': round(edge*100, 2), 'cost': round(cost, 4),
                    'score': round(score, 3), 'p_a': round(p_a, 3), 'p_b': round(p_b, 3),
                    'title_a': a['title'][:70], 'title_b': b['title'][:70],
                    'url_a': a['url'], 'url_b': b['url']
                })
    suspicious = [f for f in found if f['edge'] > 25 or f['score'] < 0.85]
    credible   = [f for f in found if f['edge'] <= 20 and f['score'] >= 0.85]
    print("\n  " + label)
    print("    Trovati:" + str(len(found)) + " | Credibili(edge<=20%,sim>=0.85):" + str(len(credible)) + " | Sospetti:" + str(len(suspicious)))
    print("    Sim distribution: " + str(dict(sorted(sim_dist.items()))))
    if found:
        print("    TOP 3 per edge:")
        for i, f in enumerate(sorted(found, key=lambda x: -x['edge'])[:3]):
            flag = "!! SOSPETTO" if f['edge'] > 25 or f['score'] < 0.85 else "OK"
            print("      #" + str(i+1) + " [" + flag + "] edge:" + str(f['edge']) + "% sim:" + str(f['score']) + " cost:" + str(f['cost']))
            print("           A(p=" + str(f['p_a']) + "): " + f['title_a'])
            print("           B(p=" + str(f['p_b']) + "): " + f['title_b'])
    if credible:
        print("    TOP 3 CREDIBILI (verifica manuale):")
        for i, f in enumerate(sorted(credible, key=lambda x: -x['edge'])[:3]):
            print("      #" + str(i+1) + " edge:" + str(f['edge']) + "% sim:" + str(f['score']))
            print("           A(p=" + str(f['p_a']) + "): " + f['title_a'])
            print("           B(p=" + str(f['p_b']) + "): " + f['title_b'])
            print("           " + f['url_a'])
            print("           " + f['url_b'])
    return found

arbs_pk  = debug_cross_arbs(poly,      kalshi,    POLY_FEE,      KALSHI_FEE,    "Poly vs Kalshi")
arbs_pik = debug_cross_arbs(predictit, kalshi,    PREDICTIT_FEE, KALSHI_FEE,    "PredictIt vs Kalshi")
arbs_ppi = debug_cross_arbs(poly,      predictit, POLY_FEE,      PREDICTIT_FEE, "Poly vs PredictIt")
all_arbs = arbs_pk + arbs_pik + arbs_ppi

print("\n[4/6] DEBUG +EV SIGNALS")
print("-" * 50)

def debug_ev(rlist, mlist, platform):
    if not rlist or not mlist:
        print("  " + platform + ": lista vuota, skip")
        return []
    vecm, vecsm = build_vecs(mlist)
    signals, seen_keys = [], set()
    for r in rlist:
        m, score = best_match(r, mlist, vecm, vecsm)
        if score < SIMILARITY_THRESHOLD:
            continue
        key = r['title'][:40]
        if key in seen_keys:
            continue
        seen_keys.add(key)
        div = m['yes'] - r['yes']
        if abs(div) >= MIN_EV_EDGE:
            direction = 'YES' if div > 0 else 'NO'
            signals.append({
                'platform': platform, 'direction': direction,
                'edge': round(abs(div)*100, 2), 'score': round(score, 3),
                'real_prob': round(r['yes'], 3), 'mani_prob': round(m['yes'], 3),
                'title_real': r['title'][:70], 'title_mani': m['title'][:70],
                'url_real': r['url'], 'url_mani': m['url']
            })
    print("  " + platform + ": " + str(len(signals)) + " segnali +EV")
    for s in signals[:5]:
        print("    [" + s['direction'] + "] edge:" + str(s['edge']) + "% | sim:" + str(s['score']) + " | real:" + str(s['real_prob']) + " vs mani:" + str(s['mani_prob']))
        print("      Real: " + s['title_real'])
        print("      Mani: " + s['title_mani'])
        print("      " + s['url_real'])
    return signals

ev_poly   = debug_ev(poly,      manifold, "Polymarket")
ev_pi     = debug_ev(predictit, manifold, "PredictIt")
ev_kalshi = debug_ev(kalshi,    manifold, "Kalshi")

print("\n[5/6] DEBUG DEDUP CACHE")
print("-" * 50)
SEEN_FILE = 'seen_alerts.json'
COOLDOWN_HOURS = 4
if os.path.exists(SEEN_FILE):
    seen_data = json.load(open(SEEN_FILE))
    now = datetime.now(timezone.utc)
    active, expired = 0, 0
    for k, ts in seen_data.items():
        last = datetime.fromisoformat(ts)
        elapsed_h = (now - last).total_seconds() / 3600
        if elapsed_h < COOLDOWN_HOURS:
            active += 1
        else:
            expired += 1
    print("  File trovato: " + SEEN_FILE)
    print("  Totale chiavi: " + str(len(seen_data)))
    print("  Attive (< " + str(COOLDOWN_HOURS) + "h): " + str(active))
    print("  Scadute (> " + str(COOLDOWN_HOURS) + "h): " + str(expired))
    print("  Ultima entry: " + (max(seen_data.values()) if seen_data else 'N/A'))
else:
    print("  ATTENZIONE: " + SEEN_FILE + " non trovato!")
    print("  -> Verifica il blocco actions/cache nel .yml")

print("\n[6/6] RIEPILOGO FINALE E RACCOMANDAZIONI")
print("=" * 70)
credible_arbs = [a for a in all_arbs if a['edge'] <= 20 and a['score'] >= 0.85]
suspicious    = [a for a in all_arbs if a['edge'] > 25]
total_ev      = len(ev_poly) + len(ev_pi) + len(ev_kalshi)
print("  Mercati  | Poly:" + str(len(poly)) + " PredictIt:" + str(len(predictit)) + " Kalshi:" + str(len(kalshi)) + " Manifold:" + str(len(manifold)))
print("  ARB      | totali:" + str(len(all_arbs)) + " credibili:" + str(len(credible_arbs)) + " sospetti:" + str(len(suspicious)))
print("  +EV      | " + str(total_ev))
print("  Dedup    | " + ("OK" if os.path.exists(SEEN_FILE) else "MANCANTE - problema!"))
print()
print("  THRESHOLD ATTUALI:")
print("    MIN_ARB_EDGE         = " + str(MIN_ARB_EDGE))
print("    MIN_EV_EDGE          = " + str(MIN_EV_EDGE))
print("    SIMILARITY_THRESHOLD = " + str(SIMILARITY_THRESHOLD))
print("    KALSHI_MIN_VOLUME    = " + str(KALSHI_MIN_VOLUME))
print()
print("  RACCOMANDAZIONI:")
if len(all_arbs) > 0:
    pct_sus = len(suspicious)/len(all_arbs)*100
    if pct_sus > 50:
        print("  !! " + str(round(pct_sus)) + "% ARB sospetti -> aumenta SIMILARITY_THRESHOLD a 0.85")
    else:
        print("  OK ratio sospetti accettabile (" + str(round(pct_sus)) + "%)")
if total_ev > 5:
    print("  !! Troppi +EV (" + str(total_ev) + ") -> aumenta MIN_EV_EDGE a 0.20")
if not os.path.exists(SEEN_FILE):
    print("  !! Cache dedup mancante -> verifica actions/cache nel .yml")
if len(credible_arbs) == 0:
    print("  !! Nessun ARB credibile trovato in questo run")
else:
    print("  OK " + str(len(credible_arbs)) + " ARB credibili trovati")
print("=" * 70)

import requests, json, re
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

NTFY_TOPIC = 'arb_erscop_83041'
MIN_ARB_EDGE = 0.03
MIN_EV_EDGE = 0.08
SIMILARITY_THRESHOLD = 0.55
POLY_FEE = 0.02
PREDICTIT_FEE = 0.10
KALSHI_FEE = 0.07

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

def get_polymarket():
    try:
        result = []
        seen = set()
        for offset in [0, 200, 400]:
            r = requests.get(
                'https://gamma-api.polymarket.com/events',
                params={'active':'true','closed':'false','limit':200,'offset':offset,'order':'id','ascending':'false'},
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
                        no = float(prices[1])
                        liq = float(m.get('liquidity', 0))
                        if liq >= 50 and 0.03 < yes < 0.97:
                            slug = ev.get('slug') or str(ev.get('id', ''))
                            result.append({
                                'source': 'polymarket',
                                'title': m.get('question', ''),
                                'clean': clean(m.get('question', '')),
                                'yes': yes,
                                'no': no,
                                'liquidity': liq,
                                'fee': POLY_FEE,
                                'url': 'https://polymarket.com/event/' + slug
                            })
                    except Exception:
                        continue
        print('  [DEBUG] Poly usable: ' + str(len(result)))
        return result
    except Exception as e:
        print('[Polymarket ERROR] ' + str(e))
        return []

def get_predictit():
    try:
        r = requests.get('https://www.predictit.org/api/marketdata/all/', timeout=15)
        data = r.json().get('markets', [])
        result = []
        for m in data:
            mid = m.get('id')
            title = m.get('name', '')
            for c in m.get('contracts', []):
                yp = c.get('bestBuyYesCost')
                np_ = c.get('bestBuyNoCost')
                if yp is None or np_ is None:
                    continue
                try:
                    yes = float(yp)
                    no = float(np_)
                except Exception:
                    continue
                if yes <= 0 or no <= 0:
                    continue
                yes = min(yes, 1.0)
                no = min(no, 1.0)
                ft = title + ' - ' + c.get('name', '')
                result.append({
                    'source': 'predictit',
                    'title': ft,
                    'clean': clean(ft),
                    'yes': yes,
                    'no': no,
                    'liquidity': float(c.get('sharesTraded', 0)),
                    'fee': PREDICTIT_FEE,
                    'url': 'https://www.predictit.org/markets/detail/' + str(mid)
                })
        print('  [DEBUG] PredictIt usable: ' + str(len(result)))
        return result
    except Exception as e:
        print('[PredictIt ERROR] ' + str(e))
        return []

def get_kalshi():
    try:
        result = []
        cursor = None
        for _ in range(3):
            params = {'limit': 200, 'status': 'open'}
            if cursor:
                params['cursor'] = cursor
            r = requests.get(
                'https://external-api.kalshi.com/trade-api/v2/markets',
                params=params, timeout=15
            )
            data = r.json()
            markets = data.get('markets', [])
            if not markets:
                break
            for m in markets:
                yr = m.get('last_price_dollars')
                if yr is None:
                    continue
                try:
                    yes = float(yr)
                    no = round(1.0 - yes, 4)
                except Exception:
                    continue
                if not (0.03 < yes < 0.97):
                    continue
                title = m.get('title', '')
                ticker = m.get('ticker', '')
                liq = float(m.get('liquidity_dollars', 0))
                result.append({
                    'source': 'kalshi',
                    'title': title,
                    'clean': clean(title),
                    'yes': yes,
                    'no': no,
                    'liquidity': liq,
                    'fee': KALSHI_FEE,
                    'url': 'https://kalshi.com/markets/' + ticker.lower()
                })
            cursor = data.get('cursor')
            if not cursor:
                break
        print('  [DEBUG] Kalshi usable: ' + str(len(result)))
        return result
    except Exception as e:
        print('[Kalshi ERROR] ' + str(e))
        return []

def get_zeitgeist():
    print('  [DEBUG] Zeitgeist usable: 0 (non configurato)')
    return []

def get_manifold():
    try:
        r = requests.get(
            'https://api.manifold.markets/v0/search-markets',
            params={'term':'','filter':'open','contractType':'BINARY','sort':'liquidity','limit':500},
            timeout=15
        )
        data = r.json()
        if isinstance(data, dict):
            data = data.get('markets', data.get('data', []))
        if not isinstance(data, list):
            data = []
        print('  [DEBUG] Manifold raw items: ' + str(len(data)))
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
                    'title': m.get('question', ''),
                    'clean': clean(m.get('question', '')),
                    'yes': prob,
                    'no': 1 - prob,
                    'url': m.get('url', 'https://manifold.markets')
                })
        print('  [DEBUG] Manifold usable: ' + str(len(result)))
        return result
    except Exception as e:
        print('[Manifold ERROR] ' + str(e))
        return []

def cross_exchange_arbs(ma, mb, fa, fb):
    if not ma or not mb:
        return []
    net_fee = fa + fb
    vecb, vecsb = build_vecs(mb)
    found, seen = [], set()
    for a in ma:
        b, score = best_match(a, mb, vecb, vecsb)
        if score < SIMILARITY_THRESHOLD:
            continue
        key = (a['source'], a['title'][:40], b['source'])
        if key in seen:
            continue
        seen.add(key)
        combos = [
            ('YES ' + a['source'] + ' + NO ' + b['source'], a['yes'] + b['no'],
             'BUY YES@' + str(round(a['yes'],2)) + ' ' + a['source'],
             'BUY NO@'  + str(round(b['no'],2))  + ' ' + b['source']),
            ('NO ' + a['source'] + ' + YES ' + b['source'], a['no'] + b['yes'],
             'BUY NO@'  + str(round(a['no'],2))  + ' ' + a['source'],
             'BUY YES@' + str(round(b['yes'],2)) + ' ' + b['source']),
        ]
        for label, cost, sA, sB in combos:
            edge = (1 - cost) - net_fee
            if edge > MIN_ARB_EDGE:
                found.append({
                    'type': 'ARB', 'label': label,
                    'edge': round(edge*100, 2), 'cost': round(cost,4),
                    'profit': round(1-cost,4), 'sA': sA, 'sB': sB,
                    'score': round(score,3),
                    'title_a': a['title'][:80], 'title_b': b['title'][:80],
                    'url_a': a['url'], 'url_b': b['url']
                })
    return found

def find_ladder_opportunities(markets):
    ladders = []
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
            level = float(match.group(2))
            base = pattern.sub('', m['title'].lower())
            key = (src, clean(base).strip(), direction)
            buckets.setdefault(key, []).append((level, m))
        for key, lst in buckets.items():
            if len(lst) < 2:
                continue
            lst.sort(key=lambda x: x[0])
            probs_y = [m['yes'] for _, m in lst]
            decreasing = key[2] in ['over', 'above', 'greater than']
            violations = []
            for i in range(len(probs_y) - 1):
                if decreasing and probs_y[i+1] > probs_y[i] + 0.03:
                    violations.append((lst[i], lst[i+1]))
                if not decreasing and probs_y[i+1] < probs_y[i] - 0.03:
                    violations.append((lst[i], lst[i+1]))
            for (lv1, m1), (lv2, m2) in violations:
                ladders.append({
                    'type': 'LADDER', 'source': src, 'base': key[1],
                    'level_1': lv1, 'level_2': lv2,
                    'p1': m1['yes'], 'p2': m2['yes'],
                    'title_1': m1['title'], 'title_2': m2['title'],
                    'url_1': m1['url'], 'url_2': m2['url']
                })
    return ladders

def find_correlated_pairs(markets):
    if not markets:
        return []
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
            mi = mks[i]
            sims = cosine_similarity(vec.transform([mi['clean']]), vecs)[0]
            for j in range(i+1, n):
                score = float(sims[j])
                if score < SIMILARITY_THRESHOLD:
                    continue
                mj = mks[j]
                diff = abs(mi['yes'] - mj['yes'])
                if diff < 0.40:
                    continue
                correlated.append({
                    'type': 'CORRELATED', 'source': src,
                    'score': round(score,3), 'diff': round(diff*100,1),
                    'title_a': mi['title'], 'title_b': mj['title'],
                    'p_yes_a': mi['yes'], 'p_yes_b': mj['yes'],
                    'url_a': mi['url'], 'url_b': mj['url']
                })
    return correlated

def find_ev_signals(rlist, mlist, platform):
    if not rlist or not mlist:
        return []
    vecm, vecsm = build_vecs(mlist)
    signals, seen = [], set()
    for r in rlist:
        m, score = best_match(r, mlist, vecm, vecsm)
        if score < SIMILARITY_THRESHOLD:
            continue
        key = r['title'][:40]
        if key in seen:
            continue
        seen.add(key)
        div = m['yes'] - r['yes']
        if abs(div) >= MIN_EV_EDGE:
            direction = 'YES' if div > 0 else 'NO'
            signals.append({
                'type': '+EV', 'platform': platform, 'direction': direction,
                'edge': round(abs(div)*100, 2),
                'real_prob': r['yes'] if direction == 'YES' else r['no'],
                'mani_prob': m['yes'] if direction == 'YES' else m['no'],
                'score': round(score,3),
                'title_real': r['title'][:80], 'title_mani': m['title'][:80],
                'url_real': r['url'], 'url_mani': m['url']
            })
    return signals

def send_ntfy(title, message, priority='default'):
    try:
        r = requests.post(
            'https://ntfy.sh/' + NTFY_TOPIC,
            data=message.encode('utf-8'),
            headers={'Title': title, 'Priority': priority},
            timeout=10
        )
        print('  -> ntfy: ' + str(r.status_code))
    except Exception as e:
        print('  [ntfy ERROR] ' + str(e))

if __name__ == '__main__':
    print('[' + datetime.utcnow().strftime('%H:%M:%S') + ' UTC] Scansione in corso...')
    poly      = get_polymarket()
    predictit = get_predictit()
    kalshi    = get_kalshi()
    zeitgeist = get_zeitgeist()
    manifold  = get_manifold()
    all_markets = poly + predictit + kalshi + zeitgeist
    print('  -> Poly:' + str(len(poly)) + ' | PredictIt:' + str(len(predictit)) +
          ' | Kalshi:' + str(len(kalshi)) + ' | Zeitgeist:' + str(len(zeitgeist)) +
          ' | Manifold:' + str(len(manifold)))
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
    print('  -> ARB total:' + str(len(all_arbs)) + ' | LADDER:' + str(len(ladder_ops)) +
          ' | CORRELATED:' + str(len(corr_pairs)) + ' | +EV total:' + str(len(ev_all)))
    arbs_sorted   = sorted(all_arbs,   key=lambda x: -x['edge'])
    corr_sorted   = sorted(corr_pairs, key=lambda x: -x['diff'])[:1]
    ladder_sorted = ladder_ops[:3]
    ev_sorted     = sorted(ev_all,     key=lambda x: -x['edge'])[:3]
    for a in arbs_sorted:
        msg = (a['sA'] + chr(10) + a['sB'] + chr(10) +
               'Costo: $' + str(a['cost']) + ' | Profitto: $' + str(a['profit']) + ' per $1' + chr(10) +
               'Similarity: ' + str(a['score']) + chr(10) +
               a['title_a'] + chr(10) + a['title_b'] + chr(10) +
               a['url_a'] + chr(10) + a['url_b'])
        send_ntfy('ARB +' + str(a['edge']) + '% | ' + a['label'], msg, priority='urgent')
    for l in ladder_sorted:
        msg = ('Exchange: ' + l['source'] + chr(10) +
               'Base: ' + l['base'] + chr(10) +
               'Livello ' + str(l['level_1']) + ' -> p=' + str(round(l['p1'],2)) + chr(10) +
               'Livello ' + str(l['level_2']) + ' -> p=' + str(round(l['p2'],2)) + chr(10) +
               l['title_1'] + chr(10) + l['title_2'] + chr(10) +
               l['url_1'] + chr(10) + l['url_2'])
        send_ntfy('LADDER anomaly', msg, priority='high')
    for c in corr_sorted:
        msg = ('Exchange: ' + c['source'] + chr(10) +
               'Similarity: ' + str(c['score']) + chr(10) +
               'Diff YES: ' + str(c['diff']) + '%' + chr(10) +
               c['title_a'] + ' (p_yes=' + str(round(c['p_yes_a'],2)) + ')' + chr(10) +
               c['title_b'] + ' (p_yes=' + str(round(c['p_yes_b'],2)) + ')' + chr(10) +
               c['url_a'] + chr(10) + c['url_b'])
        send_ntfy('Correlated spread', msg, priority='default')
    for s in ev_sorted:
        msg = ('Piattaforma: ' + s['platform'] + chr(10) +
               'Direzione: ' + s['direction'] + chr(10) +
               'Prezzo attuale: ' + str(round(s['real_prob'],2)) +
               ' | Stima Manifold: ' + str(round(s['mani_prob'],2)) + chr(10) +
               'Divergenza: +' + str(s['edge']) + '%' + chr(10) +
               'Mercato: ' + s['title_real'] + chr(10) +
               'Manifold: ' + s['title_mani'] + chr(10) +
               s['url_real'] + chr(10) + s['url_mani'])
        send_ntfy('+EV ' + s['direction'] + ' +' + str(s['edge']) + '% su ' + s['platform'],
                  msg, priority='high')
    if not (arbs_sorted or ladder_sorted or corr_sorted or ev_sorted):
        print('  -> Nessuna opportunita inviata.')

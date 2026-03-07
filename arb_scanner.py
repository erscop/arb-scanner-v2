import requests, json, os, re
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

NTFY_TOPIC     = "arb_erscop_83041"
MIN_ARB_EDGE   = 0.03
MIN_EV_EDGE    = 0.08
MIN_SIMILARITY = 0.20
POLY_FEE       = 0.02
KALSHI_FEE     = 0.02

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

def get_polymarket():
    try:
        r = requests.get("https://gamma-api.polymarket.com/markets",
                         params={"active":"true","closed":"false","limit":500},
                         timeout=15)
        result = []
        for m in r.json():
            try:
                prices = json.loads(m.get("outcomePrices","[]"))
                yes, no = float(prices[0]), float(prices[1])
                if float(m.get("liquidity",0)) >= 50 and 0.03 < yes < 0.97:
                    result.append({
                        "source":    "polymarket",
                        "title":     m.get("question",""),
                        "clean":     clean(m.get("question","")),
                        "yes":       yes, "no": no,
                        "liquidity": float(m.get("liquidity",0)),
                        "url":       f"https://polymarket.com/event/{m.get('slug','')}"
                    })
            except: continue
        return result
    except Exception as e:
        print(f"[Polymarket ERROR] {e}"); return []

def get_kalshi():
    try:
        r = requests.get("https://api.elections.kalshi.com/trade-api/v2/markets",
                         params={"status":"open","limit":500},
                         headers={"accept":"application/json"}, timeout=15)
        result = []
        for m in r.json().get("markets",[]):
            ya = m.get("yes_ask")
            na = m.get("no_ask")
            if ya is not None and na is not None and ya > 2 and na > 2:
                result.append({
                    "source":  "kalshi",
                    "title":   m.get("title",""),
                    "clean":   clean(m.get("title","")),
                    "yes":     ya / 100,
                    "no":      na / 100,
                    "url":     f"https://kalshi.com/markets/{m.get('ticker','')}"
                })
        return result
    except Exception as e:
        print(f"[Kalshi ERROR] {e}"); return []

def get_manifold():
    try:
        r = requests.get("https://api.manifold.markets/v0/markets",
                         params={"limit":500,"sort":"liquidity","order":"desc"},
                         timeout=15)
        data = r.json()
        if isinstance(data, dict):
            data = data.get("markets", data.get("data", []))
        result = []
        for m in data:
            if not isinstance(m, dict): continue
            if m.get("outcomeType") != "BINARY": continue
            if m.get("isResolved", False): continue
            prob = m.get("probability")
            if prob is None: continue
            try:
                prob = float(prob)
            except: continue
            if 0.04 < prob < 0.96:
                result.append({
                    "source": "manifold",
                    "title":  m.get("question",""),
                    "clean":  clean(m.get("question","")),
                    "yes":    prob,
                    "no":     1 - prob,
                    "url":    m.get("url","https://manifold.markets")
                })
        return result
    except Exception as e:
        print(f"[Manifold ERROR] {e}"); return []

def build_vectorizer_and_vecs(market_list):
    vectorizer = TfidfVectorizer(
        stop_words=STOP_WORDS,
        ngram_range=(1,2),
        min_df=1,
        sublinear_tf=True
    )
    vecs = vectorizer.fit_transform([m["clean"] for m in market_list])
    return vectorizer, vecs

def best_match(source_market, target_list, vectorizer, target_vecs):
    src_vec = vectorizer.transform([source_market["clean"]])
    sims    = cosine_similarity(src_vec, target_vecs)[0]
    best_j  = int(np.argmax(sims))
    return target_list[best_j], float(sims[best_j])

def find_arb(poly_list, kalshi_list):
    if not poly_list or not kalshi_list: return []
    net_fee = POLY_FEE + KALSHI_FEE
    vec_k, vecs_k = build_vectorizer_and_vecs(kalshi_list)
    found, seen = [], set()
    for p in poly_list:
        k, score = best_match(p, kalshi_list, vec_k, vecs_k)
        if score < MIN_SIMILARITY: continue
        key = p["title"][:40]
        if key in seen: continue
        seen.add(key)
        print(f"  [ARB SCAN {score:.3f}] '{p['title'][:40]}' <-> '{k['title'][:40]}'")
        print(f"    Poly YES:{p['yes']:.2f} NO:{p['no']:.2f} | Kalshi YES:{k['yes']:.2f} NO:{k['no']:.2f}")
        for label, cost, sA, sB in [
            ("YES Poly + NO Kalshi", p["yes"]+k["no"],
             f"BUY YES@{p['yes']:.2f} Polymarket", f"BUY NO@{k['no']:.2f} Kalshi"),
            ("NO Poly + YES Kalshi", p["no"]+k["yes"],
             f"BUY NO@{p['no']:.2f} Polymarket",   f"BUY YES@{k['yes']:.2f} Kalshi"),
        ]:
            edge = (1 - cost) - net_fee
            if edge > MIN_ARB_EDGE:
                found.append({
                    "type": "ARB", "label": label,
                    "edge": round(edge*100,2), "cost": round(cost,4),
                    "profit": round(1-cost,4), "sA": sA, "sB": sB,
                    "score": round(score,3),
                    "title_a": p["title"][:55], "title_b": k["title"][:55],
                    "url_a": p["url"], "url_b": k["url"]
                })
    return found

def find_ev_signals(real_list, manifold_list, platform_name):
    if not real_list or not manifold_list: return []
    vec_m, vecs_m = build_vectorizer_and_vecs(manifold_list)
    signals, seen = [], set()
    for r in real_list:
        m, score = best_match(r, manifold_list, vec_m, vecs_m)
        if score < MIN_SIMILARITY: continue
        key = r["title"][:40]
        if key in seen: continue
        seen.add(key)
        divergence = m["yes"] - r["yes"]
        if abs(divergence) >= MIN_EV_EDGE:
            direction = "YES" if divergence > 0 else "NO"
            signals.append({
                "type": "+EV", "platform": platform_name,
                "direction": direction,
                "edge": round(abs(divergence)*100, 2),
                "real_prob": r["yes"] if direction=="YES" else r["no"],
                "mani_prob": m["yes"] if direction=="YES" else m["no"],
                "score": round(score,3),
                "title_real": r["title"][:55], "title_mani": m["title"][:55],
                "url_real": r["url"], "url_mani": m["url"]
            })
    return signals

def send_ntfy(title, message, priority="default"):
    try:
        r = requests.post(f"https://ntfy.sh/{NTFY_TOPIC}",
                          data=message.encode("utf-8"),
                          headers={"Title": title, "Priority": priority},
                          timeout=10)
        print(f"  → ntfy: {r.status_code}")
    except Exception as e:
        print(f"  [ntfy ERROR] {e}")

if __name__ == "__main__":
    print(f"[{datetime.utcnow().strftime('%H:%M:%S')} UTC] Scansione in corso...")
    poly     = get_polymarket()
    kalshi   = get_kalshi()
    manifold = get_manifold()
    print(f"  → Poly:{len(poly)} | Kalshi:{len(kalshi)} | Manifold:{len(manifold)}")
    arbs      = find_arb(poly, kalshi)
    ev_poly   = find_ev_signals(poly,   manifold, "Polymarket")
    ev_kalshi = find_ev_signals(kalshi, manifold, "Kalshi")
    print(f"  → ARB:{len(arbs)} | +EV Poly:{len(ev_poly)} | +EV Kalshi:{len(ev_kalshi)}")
    for a in arbs:
        msg = (f"{a['sA']}\n{a['sB']}\n"
               f"Costo: ${a['cost']} | Profitto: ${a['profit']} per $1\n"
               f"Similarity: {a['score']}\n"
               f"Poly: {a['title_a']}\nKalshi: {a['title_b']}\n"
               f"{a['url_a']}\n{a['url_b']}")
        send_ntfy(f"ARB +{a['edge']}% | {a['label']}", msg, priority="urgent")
    for s in ev_poly + ev_kalshi:
        msg = (f"Piattaforma: {s['platform']}\n"
               f"Direzione: {s['direction']}\n"
               f"Prezzo attuale: {s['real_prob']:.2f} | Stima Manifold: {s['mani_prob']:.2f}\n"
               f"Divergenza: +{s['edge']}%\n"
               f"Mercato: {s['title_real']}\nManifold: {s['title_mani']}\n"
               f"{s['url_real']}\n{s['url_mani']}")
        send_ntfy(f"+EV {s['direction']} +{s['edge']}% su {s['platform']}",
                  msg, priority="high")
    if not arbs and not ev_poly and not ev_kalshi:
        print("  → Nessuna opportunita trovata.")

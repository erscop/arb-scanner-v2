import requests, json, os, re
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------- CONFIG -----------------

NTFY_TOPIC           = "arb_erscop_83041"

MIN_ARB_EDGE         = 0.03   # 3% netto minimo per ARB cross-exchange
MIN_EV_EDGE          = 0.08   # 8% divergenza contro oracle (Manifold)
SIMILARITY_THRESHOLD = 0.55   # match testuale minimo
POLY_FEE             = 0.02
PREDICTIT_FEE        = 0.10   # trading fee + withdrawal ecc. (conservativo)
GENERIC_FEE          = 0.03   # per mercati L1/L2 vari
NET_FEE_CROSS        = POLY_FEE + GENERIC_FEE

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

# ------------- TEXT / VECTORS -------------

def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

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

# ------------- DATA SOURCES --------------

def get_polymarket():
    try:
        r = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"active": "true", "closed": "false", "limit": 500},
            timeout=15
        )
        result = []
        for m in r.json():
            try:
                prices = json.loads(m.get("outcomePrices", "[]"))
                if len(prices) < 2:
                    continue
                yes, no = float(prices[0]), float(prices[1])
                liq = float(m.get("liquidity", 0))
                if liq >= 50 and 0.03 < yes < 0.97:
                    result.append({
                        "source":    "polymarket",
                        "title":     m.get("question", ""),
                        "clean":     clean(m.get("question", "")),
                        "yes":       yes,
                        "no":        no,
                        "liquidity": liq,
                        "fee":       POLY_FEE,
                        "url":       f"https://polymarket.com/event/{m.get('slug') or m.get('id','')}"
                    })
            except:
                continue
        print(f"  [DEBUG] Poly usable: {len(result)}")
        return result
    except Exception as e:
        print(f"[Polymarket ERROR] {e}")
        return []

def get_predictit():
    """
    Usa API pubblica PredictIt:
    https://www.predictit.org/api/marketdata/all/ [web:170][web:174]
    """
    try:
        r = requests.get(
            "https://www.predictit.org/api/marketdata/all/",
            timeout=15
        )
        data = r.json().get("markets", [])
        result = []
        for m in data:
            market_id = m.get("id")
            title = m.get("name", "")
            for c in m.get("contracts", []):
                # Consideriamo solo contratti che hanno sia YES che NO quote
                yes_price = c.get("bestBuyYesCost")
                no_price  = c.get("bestBuyNoCost")
                if yes_price is None or no_price is None:
                    continue
                try:
                    yes = float(yes_price)
                    no  = float(no_price)
                except:
                    continue
                if yes <= 0 or no <= 0:
                    continue
                # Normalizziamo da dollari a 0–1 (max 1.00)
                if yes > 1.01 or no > 1.01:
                    yes = min(yes, 1.00)
                    no  = min(no, 1.00)
                full_title = f"{title} – {c.get('name','')}"
                result.append({
                    "source":    "predictit",
                    "title":     full_title,
                    "clean":     clean(full_title),
                    "yes":       yes,
                    "no":        no,
                    "liquidity": float(c.get("sharesTraded", 0)),
                    "fee":       PREDICTIT_FEE,
                    "url":       f"https://www.predictit.org/markets/detail/{market_id}"
                })
        print(f"  [DEBUG] PredictIt usable: {len(result)}")
        return result
    except Exception as e:
        print(f"[PredictIt ERROR] {e}")
        return []

def get_omen():
    """
    Stub per Omen.
    Da collegare a un subgraph / API custom (es. tooling Gnosis). [web:163]
    """
    try:
        result = []
        # TODO: popola result con stessi campi di Polymarket/PredictIt
        print(f"  [DEBUG] Omen usable: {len(result)}")
        return result
    except Exception as e:
        print(f"[Omen ERROR] {e}")
        return []

def get_augur():
    """
    Stub per Augur. Richiede accesso a nodo / subgraph Augur. [web:168]
    """
    try:
        result = []
        print(f"  [DEBUG] Augur usable: {len(result)}")
        return result
    except Exception as e:
        print(f"[Augur ERROR] {e}")
        return []

def get_zeitgeist():
    """
    Stub per Zeitgeist (Polkadot). Da collegare a RPC / indexer. [web:173][web:177]
    """
    try:
        result = []
        print(f"  [DEBUG] Zeitgeist usable: {len(result)}")
        return result
    except Exception as e:
        print(f"[Zeitgeist ERROR] {e}")
        return []

def get_manifold():
    """
    Manteniamo Manifold come 'oracle' per EV (non per esecuzione).
    """
    try:
        r = requests.get(
            "https://api.manifold.markets/v0/markets",
            params={"limit": 500, "sort": "liquidity", "order": "desc", "closed": "false"},
            timeout=15
        )
        data = r.json()
        if isinstance(data, dict):
            data = data.get("markets", data.get("data", []))
        print(f"  [DEBUG] Manifold raw items: {len(data)}")

        result = []
        for m in data:
            if not isinstance(m, dict):
                continue
            if m.get("outcomeType") != "BINARY":
                continue
            if m.get("isResolved", False):
                continue
            prob = m.get("probability")
            if prob is None:
                continue
            try:
                prob = float(prob)
            except:
                continue
            if 0.0 < prob < 1.0:
                result.append({
                    "source": "manifold",
                    "title":  m.get("question", ""),
                    "clean":  clean(m.get("question", "")),
                    "yes":    prob,
                    "no":     1 - prob,
                    "url":    m.get("url", "https://manifold.markets")
                })
        print(f"  [DEBUG] Manifold usable: {len(result)}")
        return result
    except Exception as e:
        print(f"[Manifold ERROR] {e}")
        return []

# ------------- STRATEGIE: ARB, LADDER, CORRELATI -------------

def cross_exchange_arbs(markets_a, markets_b, fee_a, fee_b):
    if not markets_a or not markets_b:
        return []
    net_fee = fee_a + fee_b
    vec_b, vecs_b = build_vectorizer_and_vecs(markets_b)
    found, seen = [], set()
    for a in markets_a:
        b, score = best_match(a, markets_b, vec_b, vecs_b)
        if score < SIMILARITY_THRESHOLD:
            continue
        key = (a["source"], a["title"][:40], b["source"])
        if key in seen:
            continue
        seen.add(key)

        for label, cost, sA, sB in [
            (f"YES {a['source']} + NO {b['source']}", a["yes"] + b["no"],
             f"BUY YES@{a['yes']:.2f} {a['source']}",
             f"BUY NO@{b['no']:.2f} {b['source']}"),
            (f"NO {a['source']} + YES {b['source']}", a["no"] + b["yes"],
             f"BUY NO@{a['no']:.2f} {a['source']}",
             f"BUY YES@{b['yes']:.2f} {b['source']}"),
        ]:
            edge = (1 - cost) - net_fee
            if edge > MIN_ARB_EDGE:
                found.append({
                    "type":  "ARB",
                    "label": label,
                    "edge":  round(edge*100, 2),
                    "cost":  round(cost, 4),
                    "profit": round(1 - cost, 4),
                    "sA":    sA,
                    "sB":    sB,
                    "score": round(score, 3),
                    "title_a": a["title"][:70],
                    "title_b": b["title"][:70],
                    "url_a": a["url"],
                    "url_b": b["url"]
                })
    return found

def find_ladder_opportunities(markets):
    """
    Cerca 'ladder probability' su stesso exchange:
    es. serie di mercati tipo 'BTC > 40k', 'BTC > 50k', 'BTC > 60k'
    e segnala violazioni di monotonia P(>60k) <= P(>50k) <= P(>40k).
    """
    ladders = []
    # Raggruppa per exchange
    by_source = {}
    for m in markets:
        by_source.setdefault(m["source"], []).append(m)

    pattern = re.compile(r'(over|under|above|below|greater than|less than)\s+(\d+\.?\d*)')

    for src, mks in by_source.items():
        buckets = {}
        for m in mks:
            match = pattern.search(m["title"].lower())
            if not match:
                continue
            direction, level = match.group(1), float(match.group(2))
            base = pattern.sub('', m["title"].lower())
            key = (src, clean(base).strip(), direction)
            buckets.setdefault(key, []).append((level, m))

        for key, lst in buckets.items():
            if len(lst) < 2:
                continue
            # Ordina per livello
            lst.sort(key=lambda x: x[0])
            levels  = [lv for lv, _ in lst]
            probs_y = [m["yes"] for _, m in lst]

            # controlli di monotonia:
            # per "over/above/greater than": prob deve DECREASE con il livello
            decreasing = key[2] in ["over", "above", "greater than"]
            violations = []
            for i in range(len(probs_y) - 1):
                if decreasing and probs_y[i+1] > probs_y[i] + 0.02:
                    violations.append((lst[i], lst[i+1]))
                if not decreasing and probs_y[i+1] < probs_y[i] - 0.02:
                    violations.append((lst[i], lst[i+1]))

            for (lv1, m1), (lv2, m2) in violations:
                ladders.append({
                    "type":   "LADDER",
                    "source": src,
                    "base":   key[1],
                    "level_1": lv1,
                    "level_2": lv2,
                    "p1":     m1["yes"],
                    "p2":     m2["yes"],
                    "title_1": m1["title"],
                    "title_2": m2["title"],
                    "url_1":   m1["url"],
                    "url_2":   m2["url"]
                })
    return ladders

def find_correlated_pairs(markets):
    """
    Usa similarity per trovare coppie sullo STESSO exchange
    con score alto ma prezzi molto diversi (potenziali hedge / spread).
    """
    if not markets:
        return []
    by_source = {}
    for m in markets:
        by_source.setdefault(m["source"], []).append(m)

    correlated = []
    for src, mks in by_source.items():
        if len(mks) < 3:
            continue
        vec, vecs = build_vectorizer_and_vecs(mks)
        n = len(mks)
        for i in range(n):
            mi = mks[i]
            sims = cosine_similarity(vec.transform([mi["clean"]]), vecs)[0]
            for j in range(i+1, n):
                score = float(sims[j])
                if score < SIMILARITY_THRESHOLD:
                    continue
                mj = mks[j]
                diff = abs(mi["yes"] - mj["yes"])
                if diff < 0.25:   # cerchiamo divergenze > 15 punti
                    continue
                correlated.append({
                    "type":   "CORRELATED",
                    "source": src,
                    "score":  round(score, 3),
                    "diff":   round(diff*100, 1),
                    "title_a": mi["title"],
                    "title_b": mj["title"],
                    "p_yes_a": mi["yes"],
                    "p_yes_b": mj["yes"],
                    "url_a":   mi["url"],
                    "url_b":   mj["url"]
                })
    return correlated

def find_ev_signals(real_list, manifold_list, platform_name):
    if not real_list or not manifold_list:
        return []
    vec_m, vecs_m = build_vectorizer_and_vecs(manifold_list)
    signals, seen = [], set()
    for r in real_list:
        m, score = best_match(r, manifold_list, vec_m, vecs_m)
        if score < SIMILARITY_THRESHOLD:
            continue
        key = r["title"][:40]
        if key in seen:
            continue
        seen.add(key)
        divergence = m["yes"] - r["yes"]
        if abs(divergence) >= MIN_EV_EDGE:
            direction = "YES" if divergence > 0 else "NO"
            signals.append({
                "type":      "+EV",
                "platform":  platform_name,
                "direction": direction,
                "edge":      round(abs(divergence)*100, 2),
                "real_prob": r["yes"] if direction=="YES" else r["no"],
                "mani_prob": m["yes"] if direction=="YES" else m["no"],
                "score":     round(score,3),
                "title_real": r["title"][:70],
                "title_mani": m["title"][:70],
                "url_real":   r["url"],
                "url_mani":   m["url"]
            })
    return signals

# ------------- NOTIFICHE ------------------

def send_ntfy(title, message, priority="default"):
    try:
        r = requests.post(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=message.encode("utf-8"),
            headers={"Title": title, "Priority": priority},
            timeout=10
        )
        print(f"  → ntfy: {r.status_code}")
    except Exception as e:
        print(f"  [ntfy ERROR] {e}")

# ------------- MAIN -----------------------

if __name__ == "__main__":
    print(f"[{datetime.utcnow().strftime('%H:%M:%S')} UTC] Scansione in corso...")

    poly       = get_polymarket()
    predictit  = get_predictit()
    omen       = get_omen()
    augur      = get_augur()
    zeitgeist  = get_zeitgeist()
    manifold   = get_manifold()

    all_markets = poly + predictit + omen + augur + zeitgeist

    print(f"  → Poly:{len(poly)} | PredictIt:{len(predictit)} | "
          f"Omen:{len(omen)} | Augur:{len(augur)} | Zeitgeist:{len(zeitgeist)} | "
          f"Manifold:{len(manifold)}")

    arbs_poly_pi = cross_exchange_arbs(poly, predictit, POLY_FEE, PREDICTIT_FEE)
    ladder_ops   = find_ladder_opportunities(all_markets)
    corr_pairs   = find_correlated_pairs(all_markets)
    ev_poly      = find_ev_signals(poly,      manifold, "Polymarket")
    ev_pi        = find_ev_signals(predictit, manifold, "PredictIt")

    print(f"  → ARB Poly-PredictIt:{len(arbs_poly_pi)} | "
          f"LADDER:{len(ladder_ops)} | CORRELATED:{len(corr_pairs)} | "
          f"+EV Poly:{len(ev_poly)} | +EV PredictIt:{len(ev_pi)}")

    # ---- LIMITI NOTIFICHE ----
    MAX_CORRELATED = 5      # massimo 5 notifiche spread
    MAX_LADDER     = 5
    MAX_EV         = 5

    # Ordina e taglia
    arbs_poly_pi_sorted = sorted(arbs_poly_pi, key=lambda x: -x["edge"])
    corr_pairs_sorted   = sorted(corr_pairs,   key=lambda x: -x["diff"])[:MAX_CORRELATED]
    ladder_ops_sorted   = ladder_ops[:MAX_LADDER]
    ev_all_sorted       = sorted(ev_poly + ev_pi, key=lambda x: -x["edge"])[:MAX_EV]

    # --- ARB: li vogliamo tutti, ma di solito sono pochissimi ---
    for a in arbs_poly_pi_sorted:
        msg = (f"{a['sA']}\n{a['sB']}\n"
               f"Costo: ${a['cost']} | Profitto: ${a['profit']} per $1\n"
               f"Similarity: {a['score']}\n"
               f"{a['title_a']}\n{a['title_b']}\n"
               f"{a['url_a']}\n{a['url_b']}")
        send_ntfy(f"ARB +{a['edge']}% | {a['label']}", msg, priority="urgent")

    # --- LADDER: al massimo MAX_LADDER ---
    for l in ladder_ops_sorted:
        msg = (f"Exchange: {l['source']}\n"
               f"Base: {l['base']}\n"
               f"Livello {l['level_1']} -> p={l['p1']:.2f}\n"
               f"Livello {l['level_2']} -> p={l['p2']:.2f}\n"
               f"{l['title_1']}\n{l['title_2']}\n"
               f"{l['url_1']}\n{l['url_2']}")
        send_ntfy("LADDER anomaly", msg, priority="high")

    # --- CORRELATED: solo i top diff ---
    for c in corr_pairs_sorted:
        msg = (f"Exchange: {c['source']}\n"
               f"Similarity: {c['score']}\n"
               f"Diff YES: {c['diff']}%\n"
               f"{c['title_a']} (p_yes={c['p_yes_a']:.2f})\n"
               f"{c['title_b']} (p_yes={c['p_yes_b']:.2f})\n"
               f"{c['url_a']}\n{c['url_b']}")
        send_ntfy("Correlated spread", msg, priority="default")

    # --- EV: top edge ---
    for s in ev_all_sorted:
        msg = (f"Piattaforma: {s['platform']}\n"
               f"Direzione: {s['direction']}\n"
               f"Prezzo attuale: {s['real_prob']:.2f} | Stima Manifold: {s['mani_prob']:.2f}\n"
               f"Divergenza: +{s['edge']}%\n"
               f"Mercato: {s['title_real']}\n"
               f"Manifold: {s['title_mani']}\n"
               f"{s['url_real']}\n{s['url_mani']}")
        send_ntfy(f"+EV {s['direction']} +{s['edge']}% su {s['platform']}",
                  msg, priority="high")

    if not (arbs_poly_pi_sorted or ladder_ops_sorted or corr_pairs_sorted or ev_all_sorted):
        print("  → Nessuna opportunita inviata.")

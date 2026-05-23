import requests, json, os, re
from datetime import datetime, timezone
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------- CONFIG -----------------

NTFY_TOPIC           = "arb_erscop_83041"

MIN_ARB_EDGE         = 0.03
MIN_EV_EDGE          = 0.08
SIMILARITY_THRESHOLD = 0.55

POLY_FEE             = 0.02
PREDICTIT_FEE        = 0.10
KALSHI_FEE           = 0.07
GENERIC_FEE          = 0.03

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
    """
    FIX: usa /events endpoint con paginazione (2 pagine x 200) per recuperare
    tutti i mercati attivi. L'approccio via /events è il più efficiente
    secondo la doc aggiornata Polymarket 2026.
    """
    try:
        result = []
        seen_ids = set()

        for offset in [0, 200, 400]:
            r = requests.get(
                "https://gamma-api.polymarket.com/events",
                params={
                    "active":     "true",
                    "closed":     "false",
                    "limit":      200,
                    "offset":     offset,
                    "order":      "id",
                    "ascending":  "false",
                },
                timeout=15
            )
            events = r.json()
            if not events:
                break

            for event in events:
                for m in event.get("markets", []):
                    mid = m.get("id")
                    if mid in seen_ids:
                        continue
                    seen_ids.add(mid)
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
                                "url":       f"https://polymarket.com/event/{event.get('slug') or event.get('id','')}"
                            })
                    except:
                        continue

        print(f"  [DEBUG] Poly usable: {len(result)}")
        return result
    except Exception as e:
        print(f"[Polymarket ERROR] {e}")
        return []

def get_predictit():
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

def get_kalshi():
    """
    FIX: sostituisce Omen (endpoint The Graph deprecato).
    Kalshi API pubblica senza auth: external-api.kalshi.com/trade-api/v2
    Prezzi in centesimi (0-100) -> dividi per 100.
    """
    try:
        result = []
        cursor = None

        for _ in range(3):  # max 3 pagine
            params = {"limit": 200, "status": "open"}
            if cursor:
                params["cursor"] = cursor

            r = requests.get(
                "https://external-api.kalshi.com/trade-api/v2/markets",
                params=params,
                timeout=15
            )
            data = r.json()
            markets = data.get("markets", [])
            if not markets:
                break

            for m in markets:
                # yes_ask in centesimi (es. 34 = $0.34)
                yes_raw = m.get("yes_ask") or m.get("yes_bid")
                no_raw  = m.get("no_ask")  or m.get("no_bid")
                if yes_raw is None or no_raw is None:
                    continue
                try:
                    yes = float(yes_raw) / 100.0
                    no  = float(no_raw)  / 100.0
                except:
                    continue
                if not (0.03 < yes < 0.97):
                    continue

                title = m.get("title", "")
                ticker = m.get("ticker", "")
                result.append({
                    "source":    "kalshi",
                    "title":     title,
                    "clean":     clean(title),
                    "yes":       yes,
                    "no":        no,
                    "liquidity": float(m.get("volume", 0)),
                    "fee":       KALSHI_FEE,
                    "url":       f"https://kalshi.com/markets/{ticker.lower()}"
                })

            cursor = data.get("cursor")
            if not cursor:
                break

        print(f"  [DEBUG] Kalshi usable: {len(result)}")
        return result
    except Exception as e:
        print(f"[Kalshi ERROR] {e}")
        return []

def get_zeitgeist():
    print("  [DEBUG] Zeitgeist usable: 0 (non configurato)")
    return []

def get_manifold():
    """
    FIX: usa search-markets con filter=open e contractType=BINARY,
    più robusto della /v0/markets semplice che restituiva 0 items.
    """
    try:
        r = requests.get(
            "https://api.manifold.markets/v0/search-markets",
            params={
                "term":         "",
                "filter":       "open",
                "contractType": "BINARY",
                "sort":         "liquidity",
                "limit":        500,
            },
            timeout=15
        )
        data = r.json()
        if isinstance(data, dict):
            data = data.get("markets", data.get("data", []))
        if not isinstance(data, list):
            data = []

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

# ------------- STRATEGIE -----------------

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

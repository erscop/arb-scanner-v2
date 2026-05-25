import requests, json, re
from datetime import datetime, timezone
from collections import defaultdict

NTFY_TOPIC = 'arb_erscop_83041'

# ── SCARICA STORICO ────────────────────────────────────────────────────────────
print("Scaricando storico ntfy...")
r = requests.get(
    f'https://ntfy.sh/{NTFY_TOPIC}/json',
    params={'poll': '1', 'since': 'all'},
    stream=True, timeout=30
)

messages = []
for line in r.iter_lines():
    if line:
        try:
            msg = json.loads(line)
            if msg.get('event') == 'message':
                messages.append(msg)
        except Exception:
            continue

print(f"Messaggi trovati: {len(messages)}")

# ── PARSING ────────────────────────────────────────────────────────────────────
arbs, ladders, correlated, ev_signals = [], [], [], []

for m in messages:
    title = m.get('title', '')
    body  = m.get('message', '')
    ts    = datetime.fromtimestamp(m.get('time', 0), tz=timezone.utc)

    if title.startswith('ARB'):
        edge_match = re.search(r'ARB \+(\d+\.\d+)%', title)
        sim_match  = re.search(r'Similarity: ([\d.]+)', body)
        cost_match = re.search(r'Costo: \$([\d.]+)', body)
        arbs.append({
            'ts': ts,
            'title': title,
            'edge': float(edge_match.group(1)) if edge_match else None,
            'similarity': float(sim_match.group(1)) if sim_match else None,
            'cost': float(cost_match.group(1)) if cost_match else None,
            'body': body
        })
    elif title.startswith('LADDER'):
        ladders.append({'ts': ts, 'title': title, 'body': body})
    elif title.startswith('Correlated'):
        diff_match = re.search(r'Diff YES: ([\d.]+)%', body)
        correlated.append({
            'ts': ts,
            'diff': float(diff_match.group(1)) if diff_match else None,
            'body': body
        })
    elif title.startswith('+EV'):
        edge_match = re.search(r'\+([\d.]+)%', title)
        ev_signals.append({
            'ts': ts,
            'title': title,
            'edge': float(edge_match.group(1)) if edge_match else None,
            'body': body
        })

# ── STATISTICHE ────────────────────────────────────────────────────────────────
print("\n=== RIEPILOGO STORICO ===")
print(f"ARB totali notificati:         {len(arbs)}")
print(f"LADDER totali notificati:      {len(ladders)}")
print(f"CORRELATED totali notificati:  {len(correlated)}")
print(f"+EV totali notificati:         {len(ev_signals)}")

if arbs:
    edges = [a['edge'] for a in arbs if a['edge']]
    costs = [a['cost'] for a in arbs if a['cost']]
    print(f"\n--- ARB ---")
    print(f"  Edge medio:        {sum(edges)/len(edges):.2f}%")
    print(f"  Edge max:          {max(edges):.2f}%")
    print(f"  Edge min:          {min(edges):.2f}%")
    if costs:
        profits = [1 - c for c in costs]
        print(f"  Profitto medio/$1: ${sum(profits)/len(profits):.3f}")
        print(f"  Profitto max/$1:   ${max(profits):.3f}")

    fasce = defaultdict(int)
    for e in edges:
        if e < 15: fasce['10-15%'] += 1
        elif e < 20: fasce['15-20%'] += 1
        elif e < 30: fasce['20-30%'] += 1
        else: fasce['30%+'] += 1
    print(f"\n  Distribuzione edge:")
    for k, v in sorted(fasce.items()):
        print(f"    {k}: {v} opportunità")

    platforms = defaultdict(int)
    for a in arbs:
        t = a['title']
        if 'polymarket' in t.lower() and 'kalshi' in t.lower():
            platforms['Poly+Kalshi'] += 1
        elif 'polymarket' in t.lower() and 'predictit' in t.lower():
            platforms['Poly+PredictIt'] += 1
        elif 'predictit' in t.lower() and 'kalshi' in t.lower():
            platforms['PredictIt+Kalshi'] += 1
        else:
            platforms['Altro'] += 1
    print(f"\n  Distribuzione per coppia:")
    for k, v in sorted(platforms.items(), key=lambda x: -x[1]):
        print(f"    {k}: {v}")

if ev_signals:
    ev_edges = [e['edge'] for e in ev_signals if e['edge']]
    print(f"\n--- +EV ---")
    print(f"  Edge medio:  {sum(ev_edges)/len(ev_edges):.2f}%")
    print(f"  Edge max:    {max(ev_edges):.2f}%")

# Salva JSON per analisi successiva
with open('ntfy_history.json', 'w') as f:
    json.dump({
        'arbs': [{**a, 'ts': a['ts'].isoformat()} for a in arbs],
        'ladders': [{**l, 'ts': l['ts'].isoformat()} for l in ladders],
        'correlated': [{**c, 'ts': c['ts'].isoformat()} for c in correlated],
        'ev_signals': [{**e, 'ts': e['ts'].isoformat()} for e in ev_signals],
    }, f, indent=2)
print("\nSalvato ntfy_history.json")

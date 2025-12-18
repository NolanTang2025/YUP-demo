import json
from collections import Counter

try:
    with open('data.json', 'r') as f:
        data = json.load(f)

    categories = []
    intents = []
    
    for uid, udata in data.items():
        if 'sessions' in udata:
            for s in udata['sessions']:
                if 'intent_category' in s:
                    categories.append(s['intent_category'])
                if 'intent' in s:
                    intents.append(s['intent'])

    print(f"Total users: {len(data)}")
    print(f"Unique categories: {len(set(categories))}")
    print("Category counts:", Counter(categories))
    print(f"First 5 intents: {intents[:5]}")

except Exception as e:
    print(e)



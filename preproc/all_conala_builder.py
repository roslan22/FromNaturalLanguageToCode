with open('data/raw/conala-test-v.intent', 'r', encoding="UTF-8") as f:
    test_int = f.readlines()
with open('data/raw/conala-test-v.snippet', 'r', encoding="UTF-8") as f:
    test_snip = f.readlines()

with open('data/raw/train-v.intent', 'r', encoding="UTF-8") as f:
    train_int = f.readlines()
with open('data/raw/train-v.snippet', 'r', encoding="UTF-8") as f:
    train_snip = f.readlines()

with open('data/raw/conala-mined-v.intent', 'r', encoding="UTF-8") as f:
    mined_int = f.readlines()
with open('data/raw/conala-mined-v.snippet', 'r', encoding="UTF-8") as f:
    mined_snip = f.readlines()

with open('data/raw/conala-validation.intent', 'r', encoding="UTF-8") as f:
    valid_intent = f.readlines()
with open('data/raw/conala-validation.snippet', 'r', encoding="UTF-8") as f:
    valid_snip = f.readlines()

all_intents = test_int + train_int + mined_int + valid_intent
all_snippets = test_snip + train_snip + mined_snip + valid_snip
print(f'all_intents: {len(all_intents)}, all_snippets{len(all_snippets)}')

with open('data/raw/conala-all-data.intent', 'w', encoding="UTF-8") as f:
    f.writelines(all_intents)
with open('data/raw/conala-all-data.snippet', 'w', encoding="UTF-8") as f:
    f.writelines(all_snippets)

print('all done')

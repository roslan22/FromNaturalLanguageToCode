import os

BASE_DIR = os.path.join( os.path.dirname( __file__ ), '..' )

with open(BASE_DIR + '/data/val_conala.intent', 'r',  encoding="utf8") as f:
  dev_intent = f.readlines()

with open(BASE_DIR + '/data/val_conala.snippet', 'r',  encoding="utf8") as c:
   dev_snippet = c.readlines()

with open(BASE_DIR + '/data/test_conala.intent', 'r',  encoding="utf8") as f:
  test_intent = f.readlines()

with open(BASE_DIR + '/data/test_conala.snippet', 'r',  encoding="utf8") as c:
   test_snippet = c.readlines()

with open(BASE_DIR + '/data/train_conala.intent', 'r',  encoding="utf8") as f:
  train_intent = f.readlines()

with open(BASE_DIR + '/data/train_conala.snippet', 'r',  encoding="utf8") as c:
   train_snippet = c.readlines()

all_conala_intent = dev_intent +  test_intent + train_intent
all_conala_snippet = dev_snippet +  test_snippet + train_snippet

with open(BASE_DIR + '/data/all_conala.intent', mode ='w', encoding="utf8") as c:
   for sample in all_conala_intent:
       c.write(sample)

with open(BASE_DIR + '/data/all_conala.snippet', mode='w', encoding="utf8") as c:
   for sample in all_conala_snippet:
       c.write(sample)

# small version 
with open(BASE_DIR + '/data/small/all_small_conala.intent', mode ='w', encoding="utf8") as c:
   for sample in all_conala_intent[:10_000]:
       c.write(sample)

with open(BASE_DIR + '/data/small/all_small_conala.snippet', mode='w', encoding="utf8") as c:
   for sample in all_conala_snippet[:10_000]:
       c.write(sample)

# train 
with open(BASE_DIR + '/data/small/train_conala.intent', mode ='w', encoding="utf8") as c:
   for sample in all_conala_intent[:8_000]:
       c.write(sample)

with open(BASE_DIR + '/data/small/train_conala.snippet', mode ='w', encoding="utf8") as c:
   for sample in all_conala_snippet[:8_000]:
       c.write(sample)

# validation
with open(BASE_DIR + '/data/small/val_conala.intent', mode ='w', encoding="utf8") as c:
   for sample in all_conala_intent[8_001:9_000]:
       c.write(sample)

with open(BASE_DIR + '/data/small/val_conala.snippet', mode ='w', encoding="utf8") as c:
   for sample in all_conala_snippet[8_001:9_000]:
       c.write(sample)

# train 
with open(BASE_DIR + '/data/small/test_small_conala.intent', mode ='w', encoding="utf8") as c:
   for sample in all_conala_intent[9_001:9_999]:
       c.write(sample)

with open(BASE_DIR + '/data/small/test_small_conala.snippet', mode ='w', encoding="utf8") as c:
   for sample in all_conala_snippet[9_001:9_999]:
       c.write(sample)

print('all done!')



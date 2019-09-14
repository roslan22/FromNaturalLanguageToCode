def build_train_test_val_gold_data():
    # take 237 out of 2379 samples for validation other for test
    with open('data_new/conala-train.intent', 'r', encoding="UTF-8") as f:
        train_int = f.readlines()
    with open('data_new/conala-train.snippet', 'r', encoding="UTF-8") as f:
        train_snip = f.readlines()

    with open('data_new/conala-test.intent', 'r', encoding="UTF-8") as f:
        test_int = f.readlines()
    with open('data_new/conala-test.snippet', 'r', encoding="UTF-8") as f:
        test_snip = f.readlines()

    # write validation intents
    with open('data_new/gold/valid_conala2k.intent', 'w', encoding="UTF-8") as f:
        for intent in train_int[0:237]:
            f.write(intent)
    # write validation snippets
    with open('data_new/gold/valid_conala2k.snippet', 'w', encoding="UTF-8") as f:
        for snip in train_snip[0:237]:
            f.write(snip)

    # write train intents
    with open('data_new/gold/train_conala2k.intent', 'w', encoding="UTF-8") as f:
        for intent in train_int[238:]:
            f.write(intent)
    # write train snippets
    with open('data_new/gold/train_conala2k.snippet', 'w', encoding="UTF-8") as f:
        for snip in train_snip[238:]:
            f.write(snip)        

    # copy test intents
    with open('data_new/gold/test_conala2k.intent', 'w', encoding="UTF-8") as f:
        for intent in test_int:
            f.write(intent)
    # copy test file
    with open('data_new/gold/test_conala2k.snippet', 'w', encoding="UTF-8") as f:
        for snip in test_snip:
            f.write(snip)     

if __name__ == "__main__":
    build_train_test_val_gold_data()
import os
from transformers import BertTokenizer
from transformers import XLMRobertaTokenizer
import argparse
import pickle
from utils import *
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default='./data/se2', help='path to dataset folder')
    parser.add_argument('-vocab', default='entity2id.txt', help='vocab file')
    parser.add_argument('-sent', default='sentences.json', help='sent file')
    parser.add_argument('-pkl_e2s', default='entity2sents_beit3.pkl', help='name of entity2sents pkl file')
    parser.add_argument('-path_num_sents', default='num_sents-wiki.txt')
    parser.add_argument('-max_len', default=512, help='max sentence len') 
    args = parser.parse_args()
    print(args)

    tokenizer = XLMRobertaTokenizer("./beit3/beit3.spm", mask_token = '<mask>')
    mask_token = tokenizer.mask_token

    print("mask_token_id:", tokenizer.mask_token_id)
    if "Onto" in args.dataset or "ConLL" in args.dataset:
        print("encode '_'")
    else:
        print("encode '<mask>'")

    eid2name, vocab, eid2idx = load_vocab(os.path.join(args.dataset, args.vocab))

    entity2sents = dict()
    for eid in vocab:
        entity2sents[eid] = []
    
    filename = os.path.join(args.dataset, args.sent)

    with open(filename, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=80344528):
            obj = json.loads(line)
            # if len(obj['entityMentions']) == 0 or len(obj['tokens']) > args.max_len:
            #     continue

            raw_sent = [token.lower() for token in obj['tokens']]
            for entity in obj['entityMentions']:
                eid = entity['entityId']
                if eid not in entity2sents:
                    continue
                if len(entity2sents[eid]) > 20:
                    continue
                sent = copy.deepcopy(raw_sent)

                if sent[:entity['start']] != []:
                    front_sent = tokenizer.encode(' '.join(sent[:entity['start']]))[1:-1]
                else:
                    front_sent = []
                
                if sent[entity['end']+1:] != []:
                    behind_sent = tokenizer.encode(' '.join(sent[:entity['start']]))[1:-1]
                else:
                    behind_sent = []

                sent_ids = [tokenizer.mask_token_id]
                
                while True:
                    if front_sent != [] and len(sent_ids) < args.max_len - 2:
                        sent_ids = [front_sent[-1]] + sent_ids
                        front_sent = front_sent[:-1]

                    if behind_sent != [] and len(sent_ids) < args.max_len - 2:
                        sent_ids = sent_ids + [behind_sent[0]]
                        behind_sent = behind_sent[1:]
                    
                    if behind_sent == [] and front_sent == []:
                        break
                    
                    if len(sent_ids) == args.max_len - 2:
                        break

                sent_ids = [tokenizer.bos_token_id] + sent_ids + [tokenizer.eos_token_id]
                        
                while len(sent_ids) < args.max_len: 
                    sent_ids.append(tokenizer.pad_token_id)
                # sent[entity['start']:entity['end'] + 1] = [mask_token]
                entity2sents[eid].append(sent_ids)
    

    print(len(entity2sents))
    for k, v in entity2sents.items():
        print(entity2sents[k][0])
        break
    
    pickle.dump(entity2sents, open(os.path.join(args.dataset, args.pkl_e2s), 'wb'))

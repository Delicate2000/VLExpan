import os
import json

dataset = "NERD"
resfile = "text-kl_beit3_42_(beit3)_lr=2e-05_epoch=1_batchsize=32_num_sen_per_entity=128"
text_file = "_text"
visual_file = "_visual"
multi_file = "_multi2"


TOPK = 10
LASTK = 20

class_names = []
gt = dict()
query_sets = dict()
bad_case = dict()
negative_sample = dict()
num_query_per_class = 0


for file in os.listdir(os.path.join("../data", dataset, "query")):
    class_name = file.split('.')[0]
    class_names.append(class_name)
    query_sets[class_name] = []
    gt[class_name] = set()
    bad_case[class_name] = []
    negative_sample[class_name] = []
    num_query_per_class = 0

    with open(os.path.join("../data", dataset, 'query', file), encoding='utf-8') as f:
        for line in f:
            if line == 'EXIT\n':
                break
            num_query_per_class += 1
            temp = line.strip().split(' ')
            query_sets[class_name].append([int(eid) for eid in temp])

    with open(os.path.join("../data", dataset, 'gt', file), encoding='utf-8') as f:
        for line in f:
            temp = line.strip().split('\t')
            eid = int(temp[0])
            if int(temp[2]) >= 1:
                gt[class_name].add(eid)



data_dir = "../data/" + dataset
def load_vocab(filename):
    eid2name = {}
    keywords = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            temp = line.strip().split('\t')
            eid = int(temp[1])
            eid2name[eid] = temp[0]
            keywords.append(eid)
    print(f'Vocabulary: {len(keywords)} keywords loaded')
    return eid2name

eid2name = load_vocab(os.path.join(data_dir, "entity2id.txt"))

# find the position of each entity
t_eid2index = dict()
i_eid2index = dict()
m_eid2index = dict()
candidate_len = 450

for cls in class_names:
    t_eid2index[cls] = []
    i_eid2index[cls] = []
    m_eid2index[cls] = []
    

for cls in class_names:
    for i in range(num_query_per_class):
        eid2index = dict()

        with open(os.path.join(dataset, resfile, str(i)+"_"+cls+text_file), encoding='utf-8') as f:
            lines = f.readlines()
            for index, line in enumerate(lines[2:2+candidate_len]):
                eid = int(line.strip().split('\t')[0])
                eid2index[eid] = index
            
        t_eid2index[cls].append(eid2index)

for cls in class_names:
    for i in range(num_query_per_class):
        eid2index = dict()
        
        with open(os.path.join(dataset, resfile, str(i)+"_"+cls+visual_file), encoding='utf-8') as f:
            lines = f.readlines()
            for index, line in enumerate(lines[2:2+candidate_len]):
                eid = int(line.strip().split('\t')[0])
                eid2index[eid] = index
                
        i_eid2index[cls].append(eid2index)


for cls in class_names:
    for i in range(num_query_per_class):
        eid2index = dict()
        
        with open(os.path.join(dataset, resfile, str(i)+"_"+cls+multi_file), encoding='utf-8') as f:
            lines = f.readlines()
            for index, line in enumerate(lines[2:2+candidate_len]):
                eid = int(line.strip().split('\t')[0])
                eid2index[eid] = index
                
        m_eid2index[cls].append(eid2index)
        
import math

win_size = 10
need_rerank = dict()
no_rank = dict()

for cls in class_names:
    need_rerank[cls] = []

for cls in class_names:
    for i in range(len(m_eid2index[cls])):
        rerank = []
        for eid in m_eid2index[cls][i]:
            m_index = m_eid2index[cls][i][eid]
            try:
                t_index = t_eid2index[cls][i][eid]
            except:
                t_index = 999
            try:
                i_index = i_eid2index[cls][i][eid]
            except:
                i_index = 999
            if abs(m_index - i_index) >= win_size or abs(m_index - t_index) >= win_size: # 有一个大就要重排了
                continue
            else:
                no_rank[f"{cls}_{i}"].append((eid, m_index))



become_first = dict()
m_before_k = 15

for cls in class_names:
    for i in range(5):
        become_first[f"{cls}_{i}"] = []
        
for k, v in no_rank.items():
    for case in v:
        if case[1] <= m_before_k:
            become_first[k].append(case)

with open(f"./become_first_{dataset}_{resfile}.json","w") as f:
    json.dump(become_first,f)

tv_win_size = 10
t_norank = dict()
for cls in class_names:
    for i in range(len(m_eid2index[cls])):
        for eid in t_eid2index[cls][i]:
            t_index = t_eid2index[cls][i][eid]
            try:
                m_index = m_eid2index[cls][i][eid]
            except:
                m_index = 999
            try:
                i_index = i_eid2index[cls][i][eid]
            except:
                i_index = 999
            if abs(t_index - i_index) < tv_win_size:
                try:
                    t_norank[f"{cls}_{i}"].append((eid, m_index, t_index))
                except:
                    t_norank[f"{cls}_{i}"] = []
                    t_norank[f"{cls}_{i}"].append((eid, m_index, t_index))

become_second = dict()
t_before_k = 20
for cls in class_names:
    for i in range(5):
        become_second[f"{cls}_{i}"] = []
for k, v in t_norank.items():
    for case in v:
        if case[2] <= t_before_k:
            become_second[k].append(case)
with open(f"./become_second_{dataset}_{resfile}.json","w") as f:
    json.dump(become_second,f)


negative_cases = dict()

least = 420
num = 2
for cls in class_names:
    for i in range(len(m_eid2index[cls])):
        negative_cases[f"{cls}_{i}"] = []
        for index, eid in enumerate(list(m_eid2index[cls][i])[least:]):
            m_index = m_eid2index[cls][i][eid]
            try:
                t_index = t_eid2index[cls][i][eid]
            except:
                t_index = 999
            try:
                i_index = i_eid2index[cls][i][eid]
            except:
                i_index = 999
            if t_index == 999 and i_index == 999:
                if eid not in gt[cls]:
                    negative_cases[f"{cls}_{i}"].append((eid, m_index, 0))
                else:
                    negative_cases[f"{cls}_{i}"].append((eid ,m_index, 1))
            negative_cases[f"{cls}_{i}"] = negative_cases[f"{cls}_{i}"][:num]

with open(f"./negative_cases_{dataset}_{resfile}.json","w") as f:
    json.dump(negative_cases,f)
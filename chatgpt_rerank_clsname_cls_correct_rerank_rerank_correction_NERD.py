import json

import os
import openai
from tqdm import tqdm
import time
from askGPT import ASK_GPT

keys = '''...''' # put your key here
ask_GPT = ASK_GPT(keys)

dataset = 'NERD'
win_size = 10
data_dir = os.path.join("data", dataset)
q_files = os.listdir(os.path.join(data_dir, "query"))
model_name = "text-kl_beit3_42_(beit3)_lr=2e-05_epoch=1_batchsize=32_num_sen_per_entity=128"
cand_len = 450
use_corrected_class = True
gt_files = os.listdir(os.path.join(data_dir, "gt"))

output_dir = os.path.join("results", dataset, f"cls_correction+rerank_correction_cand_length_{cand_len}_" + model_name)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

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

cls2query = dict()
for q_file in q_files:
    cls2query[q_file.replace(".txt", "")] = []

for q_file in q_files:
    with open(os.path.join(data_dir, "query", q_file), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == "EXIT":
                break
            else:
                q_s = line.split(' ')
                one_query = [int(q_id) for q_id in q_s]
                cls2query[q_file.replace(".txt", "")].append(one_query)

cls2origin_class = dict()
origin_class_dir = os.path.join("results", dataset, "chatgpt_classname_" + model_name)
origin_class_files = os.listdir(origin_class_dir)
for origin_class_file in origin_class_files:
    cls2origin_class[origin_class_file.replace(".txt", "")] = []

for origin_class_file in origin_class_files:
    with open(os.path.join(origin_class_dir, origin_class_file), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        line = lines[1].strip()
        origin_class = line.split('''"''')[1]
        cls2origin_class[origin_class_file.replace(".txt", "")] = origin_class

with open(f"results/candidate_list_{dataset}_{win_size}_{model_name}_{cand_len}.json","r",encoding='utf-8') as f1:
    cls2candidate_list = json.load(f1)

with open(f"results/m_eid2index_{dataset}_{model_name}.json","r",encoding='utf-8') as f1:
    cls2m_eid2index = json.load(f1)

with open(f"results/become_first_{dataset}_{model_name}.json","r",encoding='utf-8') as f1:
    become_first = json.load(f1)

with open(f"results/negative_cases_{dataset}_{model_name}.json","r",encoding='utf-8') as f1:
    negative_cases = json.load(f1)

if use_corrected_class:
    with open(f"results/change_class_{dataset}_{model_name}.json","r",encoding='utf-8') as f1:
        origin2correct = json.load(f1)
print("data has loaded")


def ask_chatgpt(candidate_list, seed_list, origin_class, good_cases, negative_samples, cls, q_index):
    Instruction = f"We will conduct 'entity set expansion' task. Given the seed and a candidate entity list, you need to find all the entities belong to the seeds from the candidate entity list."

    Demo = '''Here is a positive sample: Input: given seeds 'Christians, Judaism, Protestantism' and a candidate entity list:
            Japan
            German
            Shanxi
            Catholics
            Zoroastrianism
            Free Church of Scotland
            Taoism
            Apple
            Gelugpa
        Please find all the entities possibly belong to the seeds from the candidate entity list.
        Output: Based on the seeds Christians, Judaism, Protestantism' , the class name is "organization-religion".
        Entities similar to Christians, Judaism, Protestantism' and belonging to "organization-religion" should be selected out. The entities are:
            Catholics
            Zoroastrianism
            Free Church of Scotland
            Taoism
            Gelugpa'''

    seeds = ', '.join(seed_list)
    seeds = "'" + seeds + "'"

    candidates = '\n'
    for index, entity in enumerate(candidate_list):
        candidates += f'\t{entity}\n'
    candidates += f'''Please find all the entities possibly belong to the seeds from the candidate entity list.
        Output: Based on the seeds {seeds}, the class name is {origin_class}.
        Entities similar to '{seeds}' and belonging to "{origin_class}" should be selected out.'''

    # 输入问题
    Question  = f"Input: given seeds {seeds} and a candidate list:{candidates}"

    messages=[
                {"role": "user", "content": Instruction + '\n' + Demo + '\n' + Question }
            ]
    
    res = ask_GPT.askGPT4Use_nround(messages)
    print("res:", res)

    for i in range(2):
        add_prompt = ''
        check_flag = True
        if len(good_cases) != 0:
            if good_cases[0] not in res:
                add_prompt = f"\n Note that '{good_cases[0]}' is similar to the seeds '{seeds}' and should be found out."
                check_flag = False

        if len(negative_samples) != 0:
            for negative_sample in negative_samples:
                if negative_sample in res:
                    add_prompt += f"\n Note that '{negative_sample}' is different with the seeds '{seeds}' and should be neglected."
                    check_flag = False

        if check_flag:
            break
        else:
            print("start correction")
            candidates = '\n'

            Demo = '''Here is a positive sample: Input: given seeds 'Christians, Judaism, Protestantism' and a candidate entity list:
                        Japan
                        German
                        Shanxi
                        Catholics
                        Zoroastrianism
                        Free Church of Scotland
                        Taoism
                        Apple
                        Gelugpa
                    Please find all the entities possibly belong to the seeds from the candidate entity list.
                    Note that 'Free Church of Scotland' is similar to the seeds 'Christians, Judaism, Protestantism' and should be found out.
                    Note that 'Shanxi' is different with the seeds 'Christians, Judaism, Protestantism' and should be neglected.
                    Output: Based on the seeds Christians, Judaism, Protestantism' , the class name is "organization-religion".
                    Entities similar to Christians, Judaism, Protestantism' and belonging to "organization-religion" should be selected out. The entities are:
                        Catholics
                        Zoroastrianism
                        Free Church of Scotland
                        Taoism
                        Gelugpa
            '''


            for index, entity in enumerate(candidate_list):
                candidates += f'\t{entity}\n'
            candidates += f'''Please find all the entities possibly belong to the seeds from the candidate entity list. {add_prompt}
                Output: Based on the seeds {seeds}, the class name is {origin_class}.
                Entities similar to '{seeds}' and belonging to "{origin_class}" should be selected out.'''

            Question  = f'''Input: given seeds "{seeds}", class name "{origin_class}" and a candidate list:{candidates}'''

            messages=[
                {"role": "user", "content": Instruction + '\n' + Demo + '\n' + Question }
            ]
            res = ask_GPT.askGPT4Use_nround(messages)

    if not os.path.exists(output_dir):
        print("os wrong")
    with open(os.path.join(output_dir, f"{q_index}_{cls}.txt"), 'w', encoding='utf-8') as file:
        file.write("##res##\n")
        file.write(res)
        file.write("\n")

name2eid = {}
for k,v in eid2name.items():
    name2eid[v] = int(k)
len(name2eid)

gt_files = os.listdir(os.path.join(data_dir, "gt"))
        
cls2gt = dict()
for gt_file in gt_files:
    cls2gt[gt_file.replace(".txt", "")] = []
for gt_file in gt_files:
    with open(os.path.join(data_dir, "gt", gt_file), 'r', encoding='utf-8') as f:
        for line in f:
            temp = line.strip().split('\t')
            eid = int(temp[0])
            if int(temp[2]) >= 1:
                cls2gt[gt_file.replace(".txt", "")].append(eid)

def apk(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]

    num_hits = 0.0
    score = 0.0

    for i, p in enumerate(predicted):
        if int(p) in actual:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def get_predict(cls, candidate_list, start, end, num, index): 
    q_index = index
    origin_rank = cls2candidate_list[f"{cls}_{index}"].copy()

    origin_rank = [int(eid) for eid in origin_rank]
    changed_rank = []
    
    with open(os.path.join(output_dir, f"{q_index}_{cls}.txt") , 'r', encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines[1:]:

            if line[:2] == "- ":
               line = line[2:]
            ename = line.strip()     
            
            try:
                eid = name2eid[ename]
                if eid in origin_rank:
                    changed_rank.append(eid)
                    origin_rank.remove(eid)
                    
            except:
                continue
    changed_rank += origin_rank
    return changed_rank

all_p10 = 0.0
all_p20 = 0.0
all_p50 = 0.0

all_tp10 = 0.0
all_tp20 = 0.0
all_tp50 = 0.0

q_num = 0
q2expand = dict()
for cls in list(cls2query.keys()):
    for index in tqdm(range(5)):
        print(cls, index)
        query = cls2query[cls][index]
        seed_list = [eid2name[int(i)] for i in query]
        candidate_list = [eid2name[eid] for eid in cls2candidate_list[f"{cls}_{index}"][:100]]
        print(len(candidate_list))
        origin_class = cls2origin_class[f"{index}_{cls}"]

        good_cases = [eid2name[int(triple[0])] for triple in become_first[f"{cls}_{index}"]] 
        negative_samples = [eid2name[int(triple[0])] for triple in negative_cases[f"{cls}_{index}"]]
        candidate_list = negative_samples + candidate_list

        try:
            origin_class = origin2correct[f"{index}_{cls}"][-1]
        except:
            pass

        if not os.path.exists(os.path.join(output_dir, f"{index}_{cls}.txt")): 
            ask_chatgpt(candidate_list, seed_list, origin_class, good_cases, negative_samples, cls, index)

        start = 0
        end = 0
        num = 0
        actual = set(cls2gt[cls]) - set(cls2query[cls][index])
        final_predict = get_predict(cls, candidate_list, start, end, num, index)
        q2expand[f"{cls}_{index}"] = final_predict

        p10 = apk(actual, final_predict, 10)
        p20 = apk(actual, final_predict, 20)
        p50 = apk(actual, final_predict, 50)
        print(f'final_{cls}_{index}:, {p10}, {p20}, {p50}')

        t_p10 = apk(actual, cls2candidate_list[f"{cls}_{index}"], 10)
        t_p20 = apk(actual, cls2candidate_list[f"{cls}_{index}"], 20)
        t_p50 = apk(actual, cls2candidate_list[f"{cls}_{index}"], 50)
        print(f't_{cls}_{index}:, {t_p10}, {t_p20}, {t_p50}')

        all_p10 += p10
        all_p20 += p20
        all_p50 += p50

        all_tp10 += t_p10
        all_tp20 += t_p20
        all_tp50 += t_p50




print(f'final:, {all_p10/q_num}, {all_p20/q_num}, {all_p50/q_num}')
print(f'origin:, {all_tp10/q_num}, {all_tp20/q_num}, {all_tp50/q_num}')

with open(f"q2expand.json","w") as f1:
        json.dump(q2expand, f1)
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
model_name = "text-kl_beit3_42_(beit3)_lr=4.5e-05_epoch=1_batchsize=32_num_sen_per_entity=32"
start = 0
end = 100
# v_files = sorted(os.listdir(os.path.join("results", dataset, "text-kl-norank")))
# # print(v_files)
gt_files = os.listdir(os.path.join(data_dir, "gt"))
output_dir = os.path.join("results", dataset, "chatgpt_classname_" + model_name)
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
                # one_query = line.split(' ')
                q_s = line.split(' ')
                one_query = [int(q_id) for q_id in q_s]
                cls2query[q_file.replace(".txt", "")].append(one_query)

with open(f"results/become_first_{dataset}.json","r",encoding='utf-8') as f1:
    become_first = json.load(f1)

with open(f"results/m_eid2index_{dataset}.json","r",encoding='utf-8') as f1:
    cls2m_eid2index = json.load(f1)

print("data has loaded")



def ask_chatgpt(seed_list, cls, q_index, good_case=None, bad_case=None):
    Instruction = f"We will conduct 'entity set expansion' task. Given three entities, you need to accurately conclude the class name of them."

    Demo = '''Here is a positive sample:
        Input: 'Christians, Judaism, Protestantism';
        Output: the class name is "organization-religion".
      '''

    seeds = ', '.join(seed_list)
    seeds = "'" + seeds + "'"

    Question  = f'''Please conclude the class name of the given entities. Input: {seeds};
    Output: '''

    messages=[
                # {"role": "system", "content": Instruction}, 
                {"role": "user", "content": Instruction + '\n' + Demo + '\n' + Question }
                # {"role": "user", "content": "Is the Los Angeles Dodgers won the World Series in 2020?"},
            ]
    
    res = ask_GPT.askGPT4Use_nround(messages)
    print("res:", res)

    with open(os.path.join(output_dir, f"{q_index}_{cls}.txt"), 'w', encoding='utf-8') as file:
        # file.write(Instruction + '\n' + Demo + '\n' + Question )
        file.write("##res##\n")
        file.write(res)
        file.write("\n")

for cls in list(cls2query.keys()):
    for index in tqdm(range(5)):
        # print(cls, index)
        if os.path.exists(os.path.join(output_dir, f"{index}_{cls}.txt")):
            continue
        query = cls2query[cls][index]
        seed_list = [eid2name[int(i)] for i in query]
        print("seed_list:", seed_list)
        ask_chatgpt(seed_list, cls, index)

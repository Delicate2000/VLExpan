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
gt_files = os.listdir(os.path.join(data_dir, "gt"))
output_dir = os.path.join("results", dataset, "chatgpt_classname_correction_" + model_name)
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


with open(f"results/become_first_{dataset}_{model_name}.json","r",encoding='utf-8') as f1:
    become_first = json.load(f1)

with open(f"results/negative_cases_{dataset}_{model_name}.json","r",encoding='utf-8') as f1:
    negative_cases = json.load(f1)

print("data has loaded")



def ask_chatgpt(q_index, cls, seed_list, added_prompt):
    Instruction = f"We will conduct 'entity set expansion' task. Given three entities, you need to accurately conclude the class name of them."

    Demo = '''Here are two samples:
        Please conclude the class name of the given entities. Your previous anwser is "organization", you need to conclude a new class name (maybe more fine-grained) based on the input entities to make 'University of Nottingham' not belong to the new class name.
        Input: 'Christians, Judaism, Protestantism';
        Output: the class name is "organization-religion".

        Please conclude the class name of the given entities. Your previous anwser is "unknown", you need to conclude a new class name (maybe more coarse-grained) based on the input entities to make '{entity}' belong to the new class name.
        Input: 'Epcot Center Drive, I-110, Edgware Road';
        Output: the class name is "location-road_railway_highway_transit".
      '''

    seeds = ', '.join(seed_list)
    seeds = "'" + seeds + "'"

    Question  = f'''Please conclude the class name of the given entities. {added_prompt}. Input: {seeds};
    Output: '''

    messages=[
                {"role": "user", "content": Instruction + '\n' + Demo + '\n' + Question }
            ]
    
    res = ask_GPT.askGPT4Use_nround(messages)
    print("new class:", res.split('''"''')[1])
    
    return res.split('''"''')[1]


def check(seed_list, cls, q_index, origin_class, good_cases=None, negative_samples=None):
    Instruction = f"We will conduct 'entity set expansion' task. Given the class name and the entity, please judge whether the entity belong to the class"

    Demo = '''Here is a sample:
        Input: The entities are "Catholics" and "German". The class name is "organization-religion" (concluded from 'Christians, Judaism, Protestantism');
        Question:
        1. Does "Catholics" belong to the class "organization-religion"? 
        2. Does "German" belong to the class "organization-religion"?
        Anwser: 
        1. True
        2. False
      '''

    seeds = ', '.join(seed_list)
    seeds = "'" + seeds + "'"
    
    entites = []

    for good_case in good_cases:
        entites.append(good_case)
    
    for negative_sample in negative_samples:
        entites.append(negative_sample)

    entity_sent = f'''The entities are "{','.join(entites)}.'''
    question_sent = ''
    for index, entity in enumerate(entites):
        question_sent += f'''{index+1}. Does "{entity}" belong to the class "{origin_class}"?\n'''

    Question  = f'''Given the class name and the entity, please judge whether the entity belong to the class. 
    Input: {entity_sent}. The class name is "{origin_class}" (concluded from '{seeds}');
    Question:\n{question_sent}
    Anwser: '''

    messages=[
                {"role": "user", "content": Instruction + '\n' + Demo + '\n' + Question }
            ]


    print(f"{Instruction} + '\n' + {Demo} + '\n' + {Question}")
    
    res = ask_GPT.askGPT4Use_nround(messages)
    print("res:", res.split('\n'))
    answers = res.split('\n')

    check_flag = True
    added_prompt = f"Your previous anwser is {origin_class}, "

    for index, ans in enumerate(answers):
        print(ans.lower())
        if index < len(good_cases) and "true" not in ans.lower():
            added_prompt += f"you need to conclude a new class name (maybe more coarse-grained) based on the input entities to make '{entites[index]}' belong to the new class name."
            check_flag = False
        if index >= len(good_cases) and "false" not in ans.lower():
            added_prompt += f"you need to conclude a new class name (maybe more fine-grained) based on the input entities to make '{entites[index]}' not belong to the new class name."
            check_flag = False

    return added_prompt, check_flag

cls2generate_class = dict()
change_class = dict()
for cls in list(cls2query.keys()):
    for index in tqdm(range(5)):
        if os.path.exists(os.path.join(output_dir, f"{index}_{cls}.txt")):
            continue
        query = cls2query[cls][index]
        seed_list = [eid2name[int(i)] for i in query]
        good_cases = [eid2name[int(triple[0])] for triple in become_first[f"{cls}_{index}"]] 
        good_cases += seed_list
        negative_samples = [eid2name[int(triple[0])] for triple in negative_cases[f"{cls}_{index}"]] 

        origin_class = cls2origin_class[f"{index}_{cls}"]
        cls2generate_class[f"{index}_{cls}"] = origin_class
        tmp_class = origin_class

        for i in range(2):
            added_prompt, check_flag = check(seed_list, cls, index, tmp_class, good_cases, negative_samples)
            if check_flag:
                cls2generate_class[f"{index}_{cls}"] = tmp_class
                if origin_class != tmp_class:
                    change_class[f"{index}_{cls}"] = [origin_class, tmp_class]
                break
            else:
                tmp_class = ask_chatgpt(index, cls, seed_list, added_prompt)

with open(f"./results/cls2generate_class_{dataset}_{model_name}.json","w") as f:
    json.dump(cls2generate_class,f)

with open(f"./results/change_class_{dataset}_{model_name}.json","w") as f:
    json.dump(change_class,f)

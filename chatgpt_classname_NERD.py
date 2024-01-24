import json

import os
import openai
from tqdm import tqdm
import time
from askGPT import ASK_GPT
# 利用chatgpt的结果 调参

# openai.api_key = "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"
# openai.api_key = "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"
# openai.api_key = "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 "

keys = '''xptmakpbjn86@outlook.com----rrgxrtkxem15----sess-hNflAw6XmX8mPONk1sYv4wtqopnzj5ZQBXPbwf0h
exbynsnbpx93@outlook.com----grnhpbagpj99----sess-jFH2RjHV2p4BDubMPrIhdNPbwrwpQnu0o6uEWF9S
fxycraghwb18@outlook.com----aagsstejaw14----sess-CXjTRU6PI59IK2flPNmkukit4P8a6aDO6IuuxAL9
efxakwamha98@outlook.com----bdsemrybar35----sess-GhALCXJ13us8Exu3rIlHwn2n5HHeNATCI4vfov8J
rkymmcwhey24@outlook.com----mfmfdcpehp49----sess-OhRvKyaCM6VYVu2VVKrqOVX4fvqzcKz0gFApKhkX
maanppajnj94@outlook.com----pawhdmsant61----sess-JJQi1gUD1bZJdEGG86fooLNDAU59OI1zEntzGJlj
mmdffgaehs58@outlook.com----awfjkxmxst27----sess-m6qicHikuffORW5DhZEdTMPkwLJm6tPi9drNwKn7
ffsgjbkehm96@outlook.com----fygxybebht13----sess-MBXETIeKGiWXq2wJDVCNgjKOWM1qi735mK4oaxJk
cdhmbncwfh58@outlook.com----bppcdncbsc65----sess-pQjpcLGj2Z3RBeMyA3yZirOjwSlUwynLCI8aAoX7
srbxamhwky63@outlook.com----mtesfmghsm29----sess-icIPSIEvWFj07fwxsjtThD6zEnpU9Xr4caMiC8f8
eeehcsswmd23@outlook.com----xyrcxjnspy99----sess-w2JjKPgXIRoFiYcnZejJ1n8eLv1tRgzJWWfPspBS
tpeywdwbsn33@outlook.com----rmygcbrpjj25----sess-8Kjdo7iST1qdJkYE2fOEgKZCsgeAdBQvjTTzzA3g
pxxmmswtxm91@outlook.com----ehrbpjtets26----sess-3OrSMIx8m1EuFulOjWuNbkEc0FJOXVIcCiTyPykS
wcwnaywact54@outlook.com----pktkhdfdbc26----sess-wxNAvcuPN1rxX5oWJc0PMTZg8kzN8Ei5TcGHCnzB
xwrsjceesx92@outlook.com----pdpaywsjbw65----sess-0A68KYBaaVJxqmgPdVhKVdGu9ddpSbBoJ1kwENvi
bssdhwjngm54@outlook.com----jprnybejap78----sess-szQ9sciR8DOSWitAKO0NRgLiej0kyu6tIQ4PER6D
ccmjemrmhy34@outlook.com----pcytctcppf96----sess-xsG47fCNO00UJWZT8Lh87kCHWCxtNiPMrPuPw78v
sysxkycrpf19@outlook.com----yxpraahssa74----sess-i81ZORagl32N4pkg5BDteFQcIQFGYPKAcyERkMAd
hmgchdpdwc53@outlook.com----xfdewhawnp85----sess-gxvMAYVlPzqaBIQQKy0PkBL9lEhcZWtn8kQuNFVJ
bjjkxpgtja73@outlook.com----hsjeshpdkb57----sess-JK0ZD9L0jDxdH3ZRffYsDF6viET189J1lgvGuL5s'''
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
    # 给chatgpt设定角色或者任务 
    # k = 50
    Instruction = f"We will conduct 'entity set expansion' task. Given three entities, you need to accurately conclude the class name of them."

    Demo = '''Here is a positive sample:
        Input: 'Christians, Judaism, Protestantism';
        Output: the class name is "organization-religion".
      '''

    seeds = ', '.join(seed_list)
    seeds = "'" + seeds + "'"

    # 输入问题
    Question  = f'''Please conclude the class name of the given entities. Input: {seeds};
    Output: '''
    
    # 交集控制随机性
    # with open(os.path.join(output_dir, f"prompt_{q_index}_{cls}.txt"), 'w', encoding='utf-8') as file:
    #     file.write(Instruction + '\n' + Demo + '\n' + Question )

    # while 1:
    #         try:
                # completion = openai.ChatCompletion.create(
                #      model="gpt-3.5-turbo-16k-0613",
                #     # model="gpt-4",
                #     messages=[
                #             # {"role": "system", "content": Instruction}, # 系统？
                #             {"role": "user", "content": Instruction + '\n' + Demo + '\n' + Question }
                #             # {"role": "user", "content": "Is the Los Angeles Dodgers won the World Series in 2020?"}
                #             # {"role": "user", "content": "Where was it played?"}
                #         ],
                #         # ,top_p = 0.1
                #     temperature=0,
                #     seed=42
                # )

    messages=[
                # {"role": "system", "content": Instruction}, # 系统？
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

                # print("ans:", completion.choices[0].message.content)

                # # 结果写入文件
                # with open(os.path.join(output_dir, f"{q_index}_{cls}.txt"), 'w', encoding='utf-8') as file:
                #     # file.write(Instruction + '\n' + Demo + '\n' + Question )
                #     file.write("##res##\n")
                #     file.write(completion.choices[0].message.content)
                #     file.write("\n")
                # # print("ans:", completion.choices[0].message.content)
            #     break
            #     # return completion.choices[0].message.content
            # except Exception as e:
            #       time.sleep(10)
            #       print(e)

for cls in list(cls2query.keys()):
    # cls = 'person-politician'
    # print("cls:", cls)
    for index in tqdm(range(5)):
        # print(cls, index)
        if os.path.exists(os.path.join(output_dir, f"{index}_{cls}.txt")): # 有就不算了
            continue
        query = cls2query[cls][index]
        seed_list = [eid2name[int(i)] for i in query]
        print("seed_list:", seed_list)
        ask_chatgpt(seed_list, cls, index)
    #     break
    # break
import time
import openai
import copy
from openai.error import RateLimitError
# from retrying import retry



############################################################
DelKey_message = [
    'You exceeded your current quota, please check your plan and billing details.', 
]

ChangeKey_message = ['Limit: 3 / min.', 'Limit: 200 / day.', 'requests per min (RPM)', 'requests per day (RPD)']

# 'Rate limit reached for default-gpt-3.5-turbo in organization org-eO6tQeu985woe2PGVPc6KvC2 on requests per min. Limit: 3 / min. Please try again in 20s. Contact us through our help center at help.openai.com if you continue to have issues. Please add a payment method to your account to increase your rate limit. Visit https://platform.openai.com/account/billing to add a payment method.'
# 'Rate limit reached for default-gpt-3.5-turbo in organization org-gxbughMIHxzHPMOXL8Q54crk on requests per day. Limit: 200 / day. Please try again in 7m12s. Contact us through our help center at help.openai.com if you continue to have issues. Please add a payment method to your account to increase your rate limit. Visit https://platform.openai.com/account/billing to add a payment method.'



class KEYQueue(object):
    def __init__(self, key_list):
        self.init_key_list = key_list
        self.key_queue = copy.deepcopy(self.init_key_list)
        self.drained_key = []
    
    def getKey(self):
        output_key = self.key_queue.pop(0)
        self.key_queue.append(output_key)
        return output_key
    
    def lookCurrQueue(self):
        return self.key_queue
    
    def delKey(self, toDelKey):
        self.key_queue.remove(toDelKey)
        self.drained_key.append(toDelKey)
        return


# @retry(stop_max_attempt_number=5, wait_random_min=10000, wait_random_max=15000, stop_max_delay=120000)
def askGPT_1round(instruction, user_content):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_content},
            ],
        temperature=0,  #0.0 to 2.0 (默认 1.0) 温度，越高越随机，越低越有规律（或确定性）。
        # top_p=1,  # 0.0~1.0(默认1.0)不要跟温度同时用, top_p表示只考虑概率最高的top_p token，如 top_p=0.1，表示模型只考虑概率最高的 10% 的 token。
        # n=1,  #number (默认 1) 生成的回复数量.
        # stream=False,  # boolean (default False)
        # stop=None,  # string or array (default None)
        # presence_penalty=0,  # -2.0 to 2.0 (default 0)
        # frequency_penalty=0,  # -2.0 to 2.0 (default 0)
        # max_tokens=10,  # inf (default 4096-prompt_token)
        # logit_bias=
        # user=
    )
    return completion.choices[0].message.content


def askGPT_meta(messages):
    """role：有效值为“system”、“user”、“assistant”
       content
    """
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        # model="gpt-3.5-turbo-1106",
        # model="text-davinci-003",
        
        messages=messages,
        temperature=0,
        seed=42,
    )
    return completion.choices[0].message.content


# key_list = keys.split('\n')
# keyQueue = KEYQueue(key_list)
# openai.api_key = keyQueue.getKey()

# def askGPT4Use_nround(messages):
#     res = 'None'
#     status_code = 0
#     for _ in range(10):
#         status_code += 1
#         if len(keyQueue.lookCurrQueue()) < 3:
#             print('api_key池不足3个！！！')
#         try:
#             time.sleep(2)
#             res = askGPT_meta(messages)
#             status_code -= 1
#             break
#         except RateLimitError as e:
#             if any(sign in e._message for sign in ChangeKey_message):
#                 # print('更换key')
#                 openai.api_key = keyQueue.getKey()
#             elif e._message in DelKey_message:
#                 # print('删除key, 更换key')
#                 keyQueue.delKey(openai.api_key)
#                 openai.api_key = keyQueue.getKey()
#             else:
#                 print(e)
#                 time.sleep(20)
#         except Exception as e:
#             print(e)
#             time.sleep(20)
#     if status_code == 10:
#         print('失败')
#     return res


class ASK_GPT(object):
    def __init__(self, keys, time_sleep=2, error_sleep=20):
        self.time_sleep = time_sleep # 正常两次请求之间的sleep间隔
        self.error_sleep = error_sleep # 发生错误后的sleep间隔

        if isinstance(keys, str): # keys:使用"\n"来分割key
            key_list = keys.split('\n')
            key_list = [i.split('--')[-1] for i in key_list]
        elif isinstance(keys, list):
            key_list = keys
        else:
            raise
        self.keyQueue = KEYQueue(key_list)

        openai.api_key = self.keyQueue.getKey()
    
    def askGPT4Use_nround(self, messages):
        res = 'None'
        status_code = 0
        for _ in range(10):
            status_code += 1 # status_code指示正处于第几轮循环
            if len(self.keyQueue.lookCurrQueue()) < 3:
                # print('api_key池不足3个！！！')
                pass
            try:
                time.sleep(self.time_sleep)
                res = askGPT_meta(messages)
                status_code -= 1
                break
            except RateLimitError as e:
                if any(sign in e._message for sign in ChangeKey_message):
                    # print('更换key')
                    openai.api_key = self.keyQueue.getKey()
                elif e._message in DelKey_message:
                    # print('删除key, 更换key')
                    self.keyQueue.delKey(openai.api_key)
                    openai.api_key = self.keyQueue.getKey()
                else:
                    print(e)
                    time.sleep(self.error_sleep)
            except Exception as e:
                print(e)
                time.sleep(self.error_sleep)
        if status_code == 10:
            print('失败')
        return res



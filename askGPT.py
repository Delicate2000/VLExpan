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
        temperature=0, 
    )
    return completion.choices[0].message.content


def askGPT_meta(messages):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=messages,
        temperature=0,
    )
    return completion.choices[0].message.content


class ASK_GPT(object):
    def __init__(self, keys, time_sleep=2, error_sleep=20):
        self.time_sleep = time_sleep 
        self.error_sleep = error_sleep 

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
            status_code += 1 
            if len(self.keyQueue.lookCurrQueue()) < 3:
                pass
            try:
                time.sleep(self.time_sleep)
                res = askGPT_meta(messages)
                status_code -= 1
                break
            except RateLimitError as e:
                if any(sign in e._message for sign in ChangeKey_message):
                    openai.api_key = self.keyQueue.getKey()
                elif e._message in DelKey_message:
                    self.keyQueue.delKey(openai.api_key)
                    openai.api_key = self.keyQueue.getKey()
                else:
                    print(e)
                    time.sleep(self.error_sleep)
            except Exception as e:
                print(e)
                time.sleep(self.error_sleep)
        if status_code == 10:
            print('Fail')
        return res



import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertConfig, AdamW
from transformers import XLMRobertaTokenizer

from torch.utils.data import Dataset, ConcatDataset, DataLoader
from utils import *
from HCL import cl_criterion, get_cl_dataset, cl_collate_fn
from sklearn.metrics.pairwise import cosine_similarity as cos
import time
import pickle

from PIL import Image
import torch
from torch import nn
import torch.utils.data as Data
import torch.optim as optim
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel


# from beit3.modeling_finetune import beit3_base_patch16_480_vqav2

from torchvision import transforms
from torchvision.datasets.folder import default_loader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data.transforms import RandomResizedCropAndInterpolation
import numpy as np
from modeling_finetune import beit3_base_patch16_480_vqav2, beit3_large_patch16_224_nlvr2, beit3_large_patch16_480_captioning, beit3_base_patch16_480_captioning
import random
import json
class Model(nn.Module):
    def __init__(self, len_vocab, mask_token_id, mask_id=None, model_name='./beit3/beit3_base_patch16_480_vqa.pth', dim=768):
        super().__init__()
        self.beit3 = beit3_base_patch16_480_vqav2(drop_path_rate=0.0,
        vocab_size=64010)
        self.beit3.load_state_dict(torch.load(model_name, map_location='cpu')['model'], False)

        self.head = nn.Sequential(nn.Linear(dim, dim),
                                  nn.GELU(),
                                  nn.Linear(dim, len_vocab),
                                  nn.LogSoftmax(dim=-1))


        self.tokenizer = XLMRobertaTokenizer("./beit3/beit3.spm", mask_token = '<mask>')
        self.mask_token_id = self.tokenizer.mask_token_id
        print("self.mask_token_id :", self.mask_token_id)

    def forward(self, input_image, text_inputs, CL=False):
        mask_pos = (text_inputs == self.mask_token_id).nonzero(as_tuple=True)[1]
        bs = text_inputs.shape[0]
        assert len(mask_pos) == bs

        padding_mask = (text_inputs == self.tokenizer.pad_token_id).long()
        beit3_output = self.beit3(input_image, text_inputs, padding_mask)

        set_embeddings = []
        for i in range(bs):
            set_embeddings.append(beit3_output[i][mask_pos[i]+901])
        set_embeddings = torch.stack(set_embeddings, dim=0)
        set_distributions = self.head(set_embeddings)
        projection = None

        return set_distributions, projection
    

    def visual_features(self, input_image, CL=False):

        outputs = self.beit3(
            question=None, 
            image=input_image, 
            padding_mask=None, 
        )

        vision_cls = outputs[:, 0, :]

        return vision_cls
    

    def text_features(self, text_inputs, CL=False):
        mask_pos = (text_inputs == self.mask_token_id).nonzero(as_tuple=True)[1]
        bs = text_inputs.shape[0]
        assert len(mask_pos) == bs
        padding_mask = (text_inputs == self.tokenizer.pad_token_id).long()

        outputs = self.beit3(
            question=text_inputs, 
            image=torch.zeros(bs, 3 ,480, 480).cuda(), 
            padding_mask=padding_mask, 
        )

        set_embeddings = []
        for i in range(bs):
            set_embeddings.append(outputs[i][mask_pos[i]+901])
        set_embeddings = torch.stack(set_embeddings, dim=0)

        return set_embeddings


    def multi_features(self, input_image, text_inputs, CL=False):
        mask_pos = (text_inputs == self.mask_token_id).nonzero(as_tuple=True)[1]
        bs = text_inputs.shape[0]
        assert len(mask_pos) == bs

        padding_mask = (text_inputs == self.tokenizer.pad_token_id).long()
        outputs = self.beit3(input_image, text_inputs, padding_mask)

        set_embeddings = []
        for i in range(bs):
            set_embeddings.append(outputs[i][mask_pos[i]+901])
        set_embeddings = torch.stack(set_embeddings, dim=0)

        return set_embeddings


class Eid2Data(Dataset):
    def __init__(self, input_image, eid, eid2name, eid2sents, label_indexs, DATASET='wiki', siz=None, train=False):
        self.eid = eid
        self.entity = eid2name[eid]
        self.sents = eid2sents[eid]
        self.label_indexs = label_indexs

        if siz is not None:
            if siz <= len(self.sents):
                indexs = np.random.choice(len(self.sents), siz, replace=False)
                self.sents = [self.sents[i] for i in indexs]
            else: 
                if train:
                    indexs = np.random.choice(len(self.sents), siz, replace=True)
                    self.sents = [self.sents[i] for i in indexs]

        self.num_sents = len(self.sents)
        self.input_image = input_image

    def __len__(self):
        return self.num_sents

    def __getitem__(self, index):
        token_ids = self.sents[index]
        labels = self.label_indexs

        return self.input_image, token_ids, labels


def build_transform(is_train=False, input_size=480, train_interpolation='bicubic', randaug=True):

    if is_train:
        t = [
            RandomResizedCropAndInterpolation(input_size, scale=(0.5, 1.0), interpolation=train_interpolation), 
            transforms.RandomHorizontalFlip(),
        ]
        t += [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD), 
        ]
        t = transforms.Compose(t)
    else:
        t = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=3), 
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])

    return t


def collate_fn(batch):
    # print("batch:", batch)
    input_images, batch_ids, batch_labels = zip(*batch)

    batch_max_length = max(len(ids) for ids in batch_ids)
    batch_ids = torch.tensor([ids + [0 for _ in range(batch_max_length - len(ids))] for ids in batch_ids]).long()

    return torch.stack(input_images), batch_ids, batch_labels


def run_epoch(model, data_iter, loss_compute, optimizer, path, log_step=100):
    total_loss_predict = 0
    tic = time.time()
    log_step_tic = time.time()

    # masked enity prediction task
    for i, batch in enumerate(tqdm(data_iter)):
        out, _ = model.forward(batch[0].cuda(), batch[1].cuda())

        optimizer.zero_grad()
        loss_predict = loss_compute(out, batch[2])
        loss_predict.backward()
        optimizer.step()

        total_loss_predict += loss_predict.item()
        if (i + 1) % log_step == 0:
            print("Step: %4d        Loss: %.4f" % (i + 1, total_loss_predict / log_step))
            total_loss_predict = 0

            log_step_toc = time.time()
            print("log_step_time:", log_step_toc - log_step_tic)
            log_step_tic = time.time()
        
        if (i + 1) % 10000 == 0:
            torch.save(model.state_dict(), path+f"_step={i}")
    
    toc = time.time()
    print("epoch_time:", toc - tic)


class Loss_Compute(nn.Module):
    def __init__(self, criterion, len_vocab, smoothing=0):
        super(Loss_Compute, self).__init__()
        self.criterion = criterion
        self.len_vocab = len_vocab
        self.smoothing = smoothing

    def forward(self, output, batch_labels):
        dists = []
        for labels in batch_labels:
            len_set = len(labels)
            dist = torch.zeros(self.len_vocab)
            dist.fill_(self.smoothing / (self.len_vocab - len_set))
            dist.scatter_(0, torch.tensor(labels), (1 - self.smoothing) / len_set)
            dists.append(dist)
        tensor_dists = torch.stack(dists).cuda()

        return self.criterion(output, tensor_dists)

class Expan(object):
    def __init__(self, args, cls_names, eid2name, list_eids, eid2sents, eid2index, len_vocab, model_name='./beit3/beit3_base_patch16_480_coco_captioning.pth'):

        self.args = args
        self.tokenizer = XLMRobertaTokenizer("./beit3/beit3.spm", mask_token = '<mask>')
        self.mask_token_id = self.tokenizer.mask_token_id
        # dict of entity names, list of entity ids, dict of line index
        self.eid2name = eid2name

        # dict: eid to sentences
        self.eid2sents = eid2sents
        self.list_eids = list_eids
        self.len_vocab = len_vocab
        self.eid2index = eid2index

        self.model = Model(self.len_vocab, self.mask_token_id, model_name=model_name)

        self.cls_names = cls_names
        self.num_cls = len(cls_names)

        self.pkl_path_e2d = os.path.join(args.dataset, args.pkl_e2d)
        self.eindex2dists = None

        self.pkl_path_e2d = os.path.join(args.dataset, args.pkl_e2d)
        self.pkl_path_e2logd = os.path.join(args.dataset, args.pkl_e2d + '_log')
        self.pkl_path_img2d = os.path.join(args.dataset, args.pkl_img2d)
        
        self.eindex2dist = None
        self.eindex2logdist = None
        self.eid2imgdist = None
        
        print('multi_path:', self.pkl_path_e2logd+"_multi")
        if os.path.exists(self.pkl_path_e2logd+"_multi"):
            self.eindex2multi_logdist = pickle.load(open(self.pkl_path_e2logd+"_multi", 'rb'))
            print("load_multi")
        if os.path.exists(self.pkl_path_e2logd+"_multi2"):
            self.eindex2multi_logdist2 = pickle.load(open(self.pkl_path_e2logd+"_multi2", 'rb'))
            print("load_multi2")
        if os.path.exists(self.pkl_path_e2logd+"_text"):
            self.eindex2text_logdist = pickle.load(open(self.pkl_path_e2logd+"_text", 'rb'))
            print("load_text")
        if os.path.exists(self.pkl_path_e2logd+"_visual"):
            self.eindex2visual_logdist = pickle.load(open(self.pkl_path_e2logd+"_visual", 'rb'))
            print("load_visual")

        self.eid2MeanLogDist = dict()
        self.eid2dist = dict()

        self.cls2eids = None
        if os.path.exists(os.path.join(args.dataset, args.pkl_cls2eids)):
            self.cls2eids = pickle.load(open(os.path.join(args.dataset, args.pkl_cls2eids), 'rb'))

        self.dataset_name = args.dataset.split('/')[-1]

    def pretrain(self, save_path, list_dataset, lr=1e-5, epoch=5, batchsize=128, num_sen_per_entity=256, smoothing=0.1):
        self.model = Model(self.len_vocab, self.mask_token_id, model_name="./beit3/beit3_base_patch16_480_vqa.pth")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        unfreeze_layers = ['encoder.layers.9', 'encoder.layers.10', 'encoder.layers.11', 'head', 'Project', 'projection_head']
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break


        # optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

        loss_compute = Loss_Compute(nn.KLDivLoss(reduction='batchmean'), self.len_vocab,
                                    smoothing=smoothing)

        self.model.cuda()    
        dataset = ConcatDataset(list_dataset)
        data_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, collate_fn=collate_fn, num_workers=4)
      
        for i in range(0, epoch):
            print('\n[Epoch %d]' % (i + 1))
            model_pkl_name = "epoch_%d.pkl" % (i + 1) 
            path = os.path.join(save_path, model_pkl_name)
            run_epoch(self.model, data_loader, loss_compute, optimizer, path=path, log_step=100) 
            torch.save(self.model.state_dict(), path)

    def expand(self, query_sets, q_len, target_size=500, ranking=False, mu=9,
               init_win_size=1, win_grow_rate=3, win_grow_step=20, total_iter=1, mode='multi'):
        pre_expanded_sets = [None for _ in range(self.num_cls)]
        expanded_sets = query_sets 

        cnt_iter = 0
        flag_stop = False
        pre_cursor = 13

        print('Start expanding:')
        for i in range(self.num_cls): 
            print(str([self.eid2name[eid] for eid in query_sets[i]]))
        print('')

        while cnt_iter < total_iter and flag_stop is False:
            print('[Iteration %d]' % (cnt_iter + 1))
            flag_stop = True
            seed_sets = []
            cursor = target_size

            # check whether the expanded_set of each class is changed in last iteration
            # if so, renew seed set
            for i, expanded_set in enumerate(expanded_sets):
                changed = False
                if cnt_iter == 0:
                    seed_set = expanded_set 
                    changed = True
                elif cnt_iter == 1:
                    seed_set = expanded_set[:13] 
                    changed = True
                else:
                    # seed set is updated as the longest common set between pre_expanded_set and expanded_set
                    for j in range(pre_cursor, target_size):
                        for k in range(3, j):
                            if pre_expanded_sets[i][k] not in expanded_set[:j]:
                                changed = True
                                break
                        if changed and j < cursor:
                            cursor = j
                            pre_cursor = cursor
                            break
                    seed_set = expanded_set
                seed_sets.append(seed_set)

                if changed:
                    flag_stop = False
                else:
                    print(self.cls_names[i] + '  UNCHANGED')

            # truncate seed sets to same length
            if cnt_iter > 1:
                print('Cursor: ', cursor)
                print('')
                seed_sets = [seed_set[:cursor] for seed_set in seed_sets]

            pre_expanded_sets = expanded_sets
            expanded_sets = self.expand_beit3(seed_sets, target_size, ranking, mu + cnt_iter * 2,
                                         init_win_size, win_grow_rate, win_grow_step, mode=mode)

            for i, expanded_set in enumerate(expanded_sets):
                print(self.cls_names[i])
                print(str([self.eid2name[eid] for eid in expanded_set]))
            print('\n')
            cnt_iter += 1
        return [eid_set[q_len:] for eid_set in expanded_sets] 

    
    def expand_beit3(self, seed_sets, target_size, ranking, mu, init_win_size, win_grow_rate, win_grow_step, mode='multi'):
        expanded_sets = seed_sets
        init_win_size = 3
        eid_out_of_sets = set()
        
        for eid in self.list_eids:
            eid_out_of_sets.add(eid)
        
        for eid in set(sum(seed_sets, [])):
            eid_out_of_sets.remove(eid)
        
        rounds = len(expanded_sets[0]) - 3
        expand_flag = True
        while expand_flag:
            if rounds < win_grow_step:
                size_window = init_win_size
            elif rounds < 50:
                size_window = int(init_win_size + (rounds / win_grow_step) * win_grow_rate)
            else:
                size_window = int(init_win_size + (rounds / win_grow_step) * win_grow_rate * 1.25)
            if rounds >= 100:
                size_window = 1
            rounds += 1

            """ Expand """
            
            for i, cls in enumerate(self.cls_names):
                if mode == 'multi':
                    scores = np.zeros(self.len_vocab)
                else:
                    scores = np.zeros(768)

                eid_set = expanded_sets[i]
                for eid in eid_set:   
                    dist = self.get_mean_log_dist_beit3(eid, mode=mode)
                    scores += dist

                set_dist = scores / len(eid_set)
                
                mindex2eid = [eid for eid in eid_out_of_sets]
                global_scores = cos([self.get_mean_log_dist_beit3(eid, mode=mode) for eid in eid_out_of_sets], set_dist.reshape(1, -1)).reshape(-1)
                largest_k = np.argpartition(global_scores, -size_window)[-size_window:]
                indexs = largest_k[np.argsort(global_scores[largest_k])[::-1]]
                ans = [mindex2eid[index] for index in indexs]
                for tgt_eid in ans:
                    expanded_sets[i].append(tgt_eid)
                    eid_out_of_sets.remove(tgt_eid)

        return expanded_sets
    

    def load_model(self, path):
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(path))
        else:
            self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    def get_mean_log_dist_beit3(self, eid, mode):
        if mode == 'multi2':
            return self.eindex2multi_logdist2[self.eid2index[eid]]
        if mode == 'text':
            return self.eindex2text_logdist[self.eid2index[eid]]
        if mode == 'visual':
            return self.eindex2visual_logdist[self.eid2index[eid]]
        return 0

    def get_feature_dist(self, eid):
        feature_dist = self.eindex2dist[self.eid2index[eid]]
        return feature_dist

    def get_feature_dist2(self, eid, batchsize=128): 
        dataset = Eid2Data(eid, self.eid2sents, [])
        data_loader = DataLoader(dataset, batch_size=batchsize, collate_fn=collate_fn)
        list_dists = []
        for _, batch in enumerate(data_loader):
            with torch.no_grad():
                output = self.model.forward(batch[0].cuda(), batch[1].cuda())
                output = output[1] if output[0] is None else output[0]
                list_dists.append(output)
        dist = torch.cat(list_dists).cpu().numpy()
        dist = np.mean(np.exp(dist), axis=0)
        return dist
    

    def make_eindex2dist_beit3(self, eid2dataset, batchsize=256, model_id=None): 
        if torch.cuda.is_available():
            self.model.cuda()
        eindex2multi_logdist = []
        eindex2text_logdist = []
        eindex2visual_logdist = []
        eindex2multi_logdist2 = [] 


        pkl_path_e2d = self.pkl_path_e2d
        pkl_path_e2logd = self.pkl_path_e2logd
        if model_id is not None:
            pkl_path_e2d += str(model_id)
            pkl_path_e2logd += str(model_id)
        print('Total entities: %d' % len(self.eid2sents))
        print('Making %s and %s ...' % (pkl_path_e2d, pkl_path_e2logd))

        self.model.eval()

        for i, eid in enumerate(tqdm(self.eid2sents)): 
            multi_list_dists = []
            multi_list_dists2 = []
            text_list_dists = []
            list_dataset = []
            
            list_dataset = eid2dataset[eid]
            dataset = ConcatDataset(list_dataset)
            data_loader = DataLoader(dataset, batch_size=batchsize, shuffle=False, collate_fn=collate_fn)

            with torch.no_grad():
                tic = time.time()
                for j, batch in enumerate(data_loader):
                    
                    if torch.cuda.is_available():
                        multi_output = self.model.forward(batch[0].cuda(), batch[1].cuda())
                        multi_output2 = self.model.multi_features(batch[0].cuda(), batch[1].cuda())
                        text_output = self.model.text_features(batch[1].cuda())
                    else:
                        multi_output = self.model.forward(batch[0], batch[1])
                        multi_output2 = self.model.multi_features(batch[0], batch[1])
                        text_output = self.model.text_features(batch[1])
                    multi_output = multi_output[1] if multi_output[0] is None else multi_output[0]
                    multi_list_dists.append(multi_output)
                    multi_list_dists2.append(multi_output2)
                    text_list_dists.append(text_output)

                    if j == 0:
                        visual_output = self.model.visual_features(batch[0][0].unsqueeze(0).cuda()).squeeze(0).cpu().numpy()


            tic = time.time()
            multi_log_dists = torch.cat(multi_list_dists).cpu().numpy()
            text_log_dists = torch.cat(text_list_dists).cpu().numpy()
            multi_log_dists2 = torch.cat(multi_list_dists2).cpu().numpy()

            eindex2multi_logdist.append(np.mean(multi_log_dists, axis=0)) 
            eindex2multi_logdist2.append(np.mean(multi_log_dists2, axis=0)) 
            eindex2text_logdist.append(np.mean(text_log_dists, axis=0)) 
            eindex2visual_logdist.append(visual_output)

            toc = time.time()
            torch.cuda.empty_cache()
            if i % 2000 == 0:
                print(i)

        print('Writing to disk ...')
        pickle.dump(eindex2multi_logdist, open(pkl_path_e2logd+'_multi', 'wb'))
        pickle.dump(eindex2multi_logdist2, open(pkl_path_e2logd+'_multi2', 'wb'))
        pickle.dump(eindex2text_logdist, open(pkl_path_e2logd+'_text', 'wb'))
        pickle.dump(eindex2visual_logdist, open(pkl_path_e2logd+'_visual', 'wb'))
        print("Finish writing")


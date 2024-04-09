import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "4"

import torch
import argparse
from utils import *
# from Expan_Beit3_Onto_large import Expan, build_transform, Eid2Data
from Expan_Beit3_Onto import Expan, build_transform, Eid2Data
from torchvision.datasets.folder import default_loader
import sys
import copy
import random
import pickle
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default='./data/NERD', help='path to dataset folder')
    parser.add_argument('-vocab', default='entity2id.txt', help='vocab file')
    parser.add_argument('-pkl_e2s', default='entity2sents_beit3.pkl', help='name of entity2sents pkl file')
    parser.add_argument('-pretrained_model', default=None, help='name of pretrained model parameters')
    parser.add_argument('-pretrained_vitmodel', default='./vit-model/wiki/epoch_1.pkl', help='name of pretrained vit model parameters')
    parser.add_argument('-save_path', default='./model/model_wiki_beit3_all_imgs', help='path to place model parameters')
    parser.add_argument('-pkl_e2d', default='entity2dist_beit3', help='name of entity2dist pkl file')
    parser.add_argument('-pkl_img2d', default='entity2imgdist_beit3', help='name of entity2imgdist pkl file')
    parser.add_argument('-img_path', default='../image-crawler/images/NERD', help='images dir path')
    parser.add_argument('-output', default='results/wiki/text-img-multi_beit3', help='file name for output')
    parser.add_argument('-ensemble', default=False)
    parser.add_argument('-num_model', default=5)
    parser.add_argument('-num_top_model', default=2)
    parser.add_argument('-use_img', action="store_true")
    parser.add_argument('-CL', action="store_true")
    parser.add_argument('-pkl_cls2eids', default='cls2eids_beit3.pkl', help='name of cls2eids pkl file')
    parser.add_argument('-only_first_img', action="store_true")
    parser.add_argument('-make_dist', action="store_true")
    parser.add_argument('-pretrain', action="store_true")
    parser.add_argument('-test', action="store_true")
    parser.add_argument('-mode', default="multi2")
    parser.add_argument('-epoch', default=1)
    parser.add_argument('-batch_size', default=32)
    parser.add_argument('-lr', default=4e-5)
    parser.add_argument('-num_sen_per_entity', default=32)
    args = parser.parse_args()

    class_names = []
    query_sets = dict()
    gt = dict()
    num_query_per_class = 0
    for file in os.listdir(os.path.join(args.dataset, 'query')):
        class_name = file.split('.')[0]
        class_names.append(class_name)
        query_sets[class_name] = []
        gt[class_name] = set()
        num_query_per_class = 0

        with open(os.path.join(args.dataset, 'query', file), encoding='utf-8') as f:
            for line in f:
                if line == 'EXIT\n':
                    break
                num_query_per_class += 1
                temp = line.strip().split(' ')
                query_sets[class_name].append([int(eid) for eid in temp])

        with open(os.path.join(args.dataset, 'gt', file), encoding='utf-8') as f:
            for line in f:
                temp = line.strip().split('\t')
                eid = int(temp[0])
                if int(temp[2]) >= 1:
                    gt[class_name].add(eid)

    dataset_name = args.dataset.split('/')[-1]
    print("dataset_name:", dataset_name)

    transform = build_transform()
    default_image = default_loader("ind.jpg")
    default_input_image = transform(default_image)

    eid2name, _, _ = load_vocab(os.path.join(args.dataset, args.vocab))
    raw_eid2sents = pickle.load(open(os.path.join(args.dataset, args.pkl_e2s), 'rb'))
    eid2sents = dict()
    for eid in list(eid2name.keys()):
        eid2sents[eid] = raw_eid2sents[eid]
    del(raw_eid2sents)
    list_eids = list(eid2name.keys())
    len_vocab = len(list_eids)
    eid2index = {eid: i for i, eid in enumerate(list_eids)}

    # num_sen_per_entity = 32
    num_sen_per_entity = int(args.num_sen_per_entity)
    epoch = int(args.epoch)
    pkl_name = "epoch_{}.pkl".format(str(epoch))

    learn_rate=float(args.lr)
    batch_size = int(args.batch_size)
    print("learn_rate:", learn_rate)
    args.pkl_e2d += f"_lr={learn_rate}_epoch={epoch}_batchsize={batch_size}_num_sen_per_entity={num_sen_per_entity}"
    args.save_path += f"_lr={learn_rate}_epoch={epoch}_batchsize={batch_size}_num_sen_per_entity={num_sen_per_entity}"
    args.output += f"_lr={learn_rate}_epoch={epoch}_batchsize={batch_size}_num_sen_per_entity={num_sen_per_entity}"
    print(args)
        
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    expan = Expan(args, class_names, eid2name=eid2name, list_eids=list_eids, eid2sents=eid2sents, eid2index=eid2index, len_vocab=len_vocab)
    
    if not args.test: 
        eid2dataset = dict()

        if args.only_first_img:
            train_list_dataset = []
            for eid in tqdm(list_eids):
                DATASET = dataset_name
                PIC_PATH = f"../image-crawler/images/{DATASET}/"
                path = PIC_PATH + eid2name[eid].replace('"',"").replace("/","_")
                imgs = sorted(os.listdir(path))
                if len(imgs) == 0:
                    input_image = default_input_image
                    this_dataset = Eid2Data(input_image, eid, eid2name, eid2sents, [eid2index[eid]], dataset_name, num_sen_per_entity)
                    train_list_dataset.append(this_dataset)
                else:
                    flag = True
                    for image in imgs:
                        try:
                            image = default_loader(path + "/" + image) 
                            flag = False
                            input_image = transform(image)
                            this_dataset = Eid2Data(input_image, eid, eid2name, eid2sents, [eid2index[eid]], dataset_name, num_sen_per_entity)
                            train_list_dataset.append(this_dataset)
                            break
                        except:
                            continue
                    if flag:
                        input_image = default_input_image
                        this_dataset = Eid2Data(input_image, eid, eid2name, eid2sents, [eid2index[eid]], dataset_name, num_sen_per_entity)
                        train_list_dataset.append(this_dataset)
                eid2dataset[eid] = [this_dataset]
        else:
            img_size = 3
            train_list_dataset = []
            for eid in tqdm(list_eids):
                try:
                    DATASET = dataset_name
                    PIC_PATH = f"../image-crawler/images/{DATASET}/"
                    path = PIC_PATH + eid2name[eid].replace('"',"").replace("/","_")
                    imgs = sorted(os.listdir(path))
                    # imgs = []
                    if len(imgs) == 0:
                        input_image = default_input_image
                        if args.pretrain:
                            this_dataset = Eid2Data(input_image, eid, eid2name, eid2sents, [eid2index[eid]], dataset_name, num_sen_per_entity, train=True)
                            train_list_dataset.append(this_dataset)
                        test_dataset = Eid2Data(input_image, eid, eid2name, eid2sents, [eid2index[eid]], dataset_name)
                        eid2dataset[eid] = [test_dataset]
                    elif len(imgs) == 1:
                        image = imgs[0]
                        try:
                            image = default_loader(path + "/" + image)
                            input_image = transform(image)
                        except:
                            input_image = default_input_image
                        if args.pretrain:
                            this_dataset = Eid2Data(input_image, eid, eid2name, eid2sents, [eid2index[eid]], dataset_name, num_sen_per_entity, train=True)
                            train_list_dataset.append(this_dataset)
                        test_dataset = Eid2Data(input_image, eid, eid2name, eid2sents, [eid2index[eid]], dataset_name)
                        eid2dataset[eid] = [test_dataset]
                    else:
                        flag = True
                        tmp = 0
                        for image in imgs:
                            if tmp < img_size:
                                try:
                                    image = default_loader(path + "/" + image) 
                                    flag = False
                                    input_image = transform(image)
                                    if args.pretrain:
                                        this_dataset = Eid2Data(input_image, eid, eid2name, eid2sents, [eid2index[eid]], dataset_name, num_sen_per_entity, train=True)
                                        train_list_dataset.append(this_dataset)
                                    test_dataset = Eid2Data(input_image, eid, eid2name, eid2sents, [eid2index[eid]], dataset_name)
                                    if tmp == 0:
                                        eid2dataset[eid] = [test_dataset]
                                    tmp += 1
                                except:
                                    continue
                            else:
                                break
                        if flag:
                            input_image = default_input_image
                            if args.pretrain:
                                this_dataset = Eid2Data(input_image, eid, eid2name, eid2sents, [eid2index[eid]], dataset_name, num_sen_per_entity, train=True)
                                train_list_dataset.append(this_dataset)
                            test_dataset = Eid2Data(input_image, eid, eid2name, eid2sents, [eid2index[eid]], dataset_name)
                            eid2dataset[eid] = [test_dataset]
                except:
                    input_image = default_input_image
                    this_dataset = Eid2Data(input_image, eid, eid2name, eid2sents, [eid2index[eid]], dataset_name, num_sen_per_entity, train=True)
                    test_dataset = Eid2Data(input_image, eid, eid2name, eid2sents, [eid2index[eid]], dataset_name, num_sen_per_entity)
                    train_list_dataset.append(this_dataset)
                    eid2dataset[eid] = [test_dataset]
                    
        if args.pretrain:
            if not os.path.exists(args.save_path):
                os.mkdir(args.save_path)

            print("batch_size:", batch_size)
            print("epoch:", epoch)
            expan.pretrain(save_path = args.save_path, list_dataset=train_list_dataset, lr=learn_rate, epoch=epoch, batchsize=batch_size, num_sen_per_entity=8, smoothing=0.1)
        
        if args.pretrained_model is not None and args.make_dist:
            print("make_dist")
            expan.load_model(os.path.join(args.save_path, args.pretrained_model))
            expan.make_eindex2dist_beit3(batchsize=8, eid2dataset=eid2dataset)

    '''
    Expanding and Evalutation
    '''
    if args.pretrained_model is not None and args.test:
        mode = args.mode
        print(f"start expand {mode}")
        MAPs = [0, 0, 0, 0]
        num_class = len(class_names)
        target_size = 433
        query_set = []

        print("num_query_per_class:", num_query_per_class)
        with open(os.path.join(args.output, f'{mode}_summary.txt'), 'w') as file_summary:
            for i in range(0, num_query_per_class):
                print('\n[Test %d]' % (i+1))
                query_set = [query_sets[cls_name][i] for cls_name in class_names]
                origin_query_set = copy.deepcopy(query_set)

                expanded = expan.expand(query_set, target_size=target_size, ranking=False, mu=9, win_grow_rate=2.5, win_grow_step=20, q_len=len(query_set[0]), mode=mode)
                AP10s, AP20s, AP50s, AP100s = [[], [], [], []]
                for j, cls in enumerate(class_names):
                    with open(os.path.join(args.output, f'{i}_{cls}_{mode}'), 'w') as f:
                        AP10, AP20, AP50 = [apk(set(gt[cls])-set(origin_query_set[j]), expanded[j], n) for n in [10, 20, 50]]
                        AP10s.append(AP10)
                        AP20s.append(AP20)
                        AP50s.append(AP50)

                        print(AP10, AP20, AP50, file=f)
                        print('', file=f)
                        for eid in expanded[j]:
                            print(f'{eid}\t{expan.eid2name[eid]}', file=f)

                MAPs[0] += sum(AP10s) / num_class
                MAPs[1] += sum(AP20s) / num_class
                MAPs[2] += sum(AP50s) / num_class
                for j, cls in enumerate(class_names):
                    print('[%s]\t %.6f %.6f %.6f %.6f' % (cls, AP10s[j], AP20s[j], AP50s[j]))
                print('[TEST %d]' % (i + 1), file=file_summary)
                print('MAP %.6f %.6f %.6f %.6f\n' %
                    (sum(AP10s) / num_class, sum(AP20s) / num_class, sum(AP50s) / num_class),
                    file=file_summary)
            print('\nTotal MAP %.6f %.6f %.6f %.6f\n' %
                (MAPs[0] / num_query_per_class, MAPs[1] / num_query_per_class,
                MAPs[2] / num_query_per_class), file=file_summary)
            
                


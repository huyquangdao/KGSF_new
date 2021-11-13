#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The standard way to train a model. After training, also computes validation
and test error.

The user must provide a model (with ``--model``) and a task (with ``--task`` or
``--pytorch-teacher-task``).

Examples
--------

.. code-block:: shell

  python -m parlai.scripts.train -m ir_baseline -t dialog_babi:Task:1 -mf /tmp/model
  python -m parlai.scripts.train -m seq2seq -t babi:Task10k:1 -mf '/tmp/model' -bs 32 -lr 0.5 -hs 128
  python -m parlai.scripts.train -m drqa -t babi:Task10k:1 -mf /tmp/model -bs 10

"""  # noqa: E501

# TODO List:
# * More logging (e.g. to files), make things prettier.

import numpy as np
from tqdm import tqdm
from math import exp
import os

# os.environ['CUDA_VISIBLE_DEVICES']='3'
import signal
import json
import argparse
import pickle as pkl
from dataset_copy_concept_item import dataset, CRSdataset
from model_copy_concept_item import CrossModel
import torch.nn as nn
from torch import optim
import torch
from torch_geometric.utils import negative_sampling

import pickle
import wandb

import random

try:
    import torch.version
    import torch.distributed as dist

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
from nltk.translate.bleu_score import sentence_bleu


def is_distributed():
    """
    Returns True if we are in distributed mode.
    """
    return TORCH_AVAILABLE and dist.is_available() and dist.is_initialized()


def save_logs(dic, file_name):
    with open(file_name, "w") as f:
        json.dump(dic, f)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def setup_args():
    train = argparse.ArgumentParser()
    train.add_argument("-random_seed", "--random_seed", type=int, default=234)
    train.add_argument("-max_c_length", "--max_c_length", type=int, default=256)
    train.add_argument("-max_r_length", "--max_r_length", type=int, default=30)
    train.add_argument("-batch_size", "--batch_size", type=int, default=32)
    train.add_argument("-max_count", "--max_count", type=int, default=5)
    train.add_argument("-use_cuda", "--use_cuda", type=bool, default=True)
    train.add_argument("-load_dict", "--load_dict", type=str, default=None)
    train.add_argument("-learningrate", "--learningrate", type=float, default=1e-3)
    train.add_argument("-optimizer", "--optimizer", type=str, default="adam")
    train.add_argument("-momentum", "--momentum", type=float, default=0)
    train.add_argument("-is_finetune", "--is_finetune", type=bool, default=False)
    train.add_argument("--error_analysis", "--error_analysis", type=bool, default=True)
    train.add_argument(
        "-embedding_type", "--embedding_type", type=str, default="random"
    )
    train.add_argument("-epoch", "--epoch", type=int, default=2)
    train.add_argument("-gpu", "--gpu", type=str, default="0,1")
    train.add_argument("-gradient_clip", "--gradient_clip", type=float, default=0.1)
    train.add_argument("-embedding_size", "--embedding_size", type=int, default=300)

    train.add_argument("-n_heads", "--n_heads", type=int, default=2)
    train.add_argument("-n_layers", "--n_layers", type=int, default=2)
    train.add_argument("-ffn_size", "--ffn_size", type=int, default=300)

    train.add_argument("-dropout", "--dropout", type=float, default=0.1)
    train.add_argument(
        "-attention_dropout", "--attention_dropout", type=float, default=0.0
    )
    train.add_argument("-relu_dropout", "--relu_dropout", type=float, default=0.1)

    train.add_argument(
        "-learn_positional_embeddings",
        "--learn_positional_embeddings",
        type=bool,
        default=False,
    )
    train.add_argument(
        "-embeddings_scale", "--embeddings_scale", type=bool, default=True
    )

    train.add_argument("-n_entity", "--n_entity", type=int, default=40000)
    train.add_argument("-n_relation", "--n_relation", type=int, default=26)
    train.add_argument("-n_concept", "--n_concept", type=int, default=29308)
    train.add_argument("-n_con_relation", "--n_con_relation", type=int, default=48)
    train.add_argument("-dim", "--dim", type=int, default=128)
    train.add_argument("-n_hop", "--n_hop", type=int, default=2)
    train.add_argument("-kge_weight", "--kge_weight", type=float, default=1)
    train.add_argument("-l2_weight", "--l2_weight", type=float, default=2.5e-6)
    train.add_argument("-n_memory", "--n_memory", type=float, default=32)
    train.add_argument(
        "-item_update_mode", "--item_update_mode", type=str, default="0,1"
    )
    train.add_argument("-using_all_hops", "--using_all_hops", type=bool, default=True)
    train.add_argument("-num_bases", "--num_bases", type=int, default=8)
    train.add_argument("-max_neighbors", "--max_neighbors", type=int, default=10)

    train.add_argument("-train_mim", "--train_mim", type=int, default=1)

    train.add_argument(
        "-info_loss_ratio", "--info_loss_ratio", type=float, default=0.025
    )
    
    train.add_argument(
        "-num_negative", "--num_negative", type=int, default=10
    )
    

    return train


class TrainLoop_fusion_rec:
    def __init__(self, opt, is_finetune):
        self.opt = opt
        self.train_dataset = dataset("data/train_data.jsonl", opt)

        self.dict = self.train_dataset.word2index
        self.index2word = {self.dict[key]: key for key in self.dict}

        self.batch_size = self.opt["batch_size"]
        self.epoch = self.opt["epoch"]

        self.log_file_name = f"logs/log_for_{opt['max_neighbors']}_neighbors.json"
        self.logs = {}
        self.train_MIM = self.opt["train_mim"]

        self.use_cuda = opt["use_cuda"]
        if opt["load_dict"] != None:
            self.load_data = True
        else:
            self.load_data = False
        self.is_finetune = False

        self.info_loss_ratio = self.opt["info_loss_ratio"]
        self.movie_ids = pkl.load(open("generated_data/final_movie_ids.pkl", "rb"))
        # Note: we cannot change the type of metrics ahead of time, so you
        # should correctly initialize to floats or ints here

        self.metrics_rec = {
            "recall@1": 0,
            "recall@10": 0,
            "recall@50": 0,
            "loss": 0,
            "count": 0,
        }
        self.metrics_gen = {
            "dist1": 0,
            "dist2": 0,
            "dist3": 0,
            "dist4": 0,
            "bleu1": 0,
            "bleu2": 0,
            "bleu3": 0,
            "bleu4": 0,
            "count": 0,
        }

        self.build_model(is_finetune)

        if opt["load_dict"] is not None:
            # load model parameters if available
            print(
                "[ Loading existing model params from {} ]" "".format(opt["load_dict"])
            )
            states = self.model.load(opt["load_dict"])
        else:
            states = {}

        self.init_optim(
            [p for p in self.model.parameters() if p.requires_grad],
            optim_states=states.get("optimizer"),
            saved_optim_type=states.get("optimizer_type"),
        )

    def build_model(self, is_finetune):
        self.model = CrossModel(self.opt, self.dict, is_finetune)
        if self.opt["embedding_type"] != "random":
            pass
        if self.use_cuda:
            self.model.cuda()

    def train(self):
        # self.model.load_model()
        losses = []
        best_val_rec = 0
        rec_stop = False

        if self.train_MIM:
            print("Pretraining MIM objective ... ")
            for i in range(3):
                #init the current edge sets for model
#                 print('number of sample edges: ', len(set(all_sample_edges)))
#                 self.model.current_edges = set(self.model.word_item_edge_sets)
                
                train_set = CRSdataset(
                    self.train_dataset.data_process(),
                    self.opt["n_entity"],
                    self.opt["n_concept"],
                )
                train_dataset_loader = torch.utils.data.DataLoader(
                    dataset=train_set, batch_size=self.batch_size, shuffle=False
                )
                num = 0
                for (
                    context,
                    c_lengths,
                    response,
                    r_length,
                    mask_response,
                    mask_r_length,
                    entity,
                    entity_vector,
                    movie,
                    concept_mask,
                    dbpedia_mask,
                    concept_vec,
                    db_vec,
                    rec,
                ) in tqdm(train_dataset_loader):
                    seed_sets = []
                    batch_words = []
                    batch_entities = []
#                     iterations += 1
                    batch_size = context.shape[0]
                    lookup_words = []
                
                    for b in range(batch_size):
                        seed_set = entity[b].nonzero().view(-1).tolist()
                        words = concept_vec[b].nonzero().view(-1).tolist()
                        seed_sets.append(seed_set)
                        batch_words.extend(words)
                        batch_entities.extend(seed_set)
                        lookup_words.append(words)

    #                 batch_entities.extend(movie.detach().cpu().numpy().tolist())
                    word_item_pairs = [(x,y) for x in batch_words for y in batch_entities]
                    word_item_pairs.extend([(y,x) for x in batch_words for y in batch_entities])
            
                    self.model.train()
                    self.zero_grad()

                    (
                        scores,
                        preds,
                        rec_scores,
                        rec_loss,
                        gen_loss,
                        mask_loss,
                        info_db_loss,
                        info_con_loss,
                    ) = self.model(
                        context.cuda(),
                        response.cuda(),
                        mask_response.cuda(),
                        concept_mask,
                        dbpedia_mask,
                        seed_sets,
                        movie,
                        concept_vec,
                        db_vec,
                        entity_vector.cuda(),
                        rec,
                        word_item_pairs,
                        lookup_words,
                        test=False,
                    )

                    joint_loss = info_db_loss
                    losses.append([info_db_loss])
                    self.backward(joint_loss)
                    self.update_params(pretrain = True)
                    if num % 50 == 0:
                        print(
                            "info db loss is %f"
                            % (sum([l[0] for l in losses]) / len(losses))
                        )
                        # print('info con loss is %f'%(sum([l[1] for l in losses])/len(losses)))
                        losses = []
                    num += 1
                    
            self.model.save_model(name = 'pretrained_mim_model.pkl')
            # print("masked loss pre-trained")
        
        if self.train_MIM:
            self.model.load_model(name = 'pretrained_mim_model.pkl')

        print("Recommendation training ......")
#         wandb.watch(self.model)
        losses = []
        iterations = 0
        for i in range(self.epoch):
            neg_edge_index = negative_sampling(
                    edge_index=self.model.word_item_edge_sets, num_nodes=69000,
                    num_neg_samples = self.opt['num_negative'] * self.model.word_item_edge_sets.size(1) , method='sparse').cuda()
                        
            self.model.neg_edge_index = neg_edge_index
            train_set = CRSdataset(
                self.train_dataset.data_process(),
                self.opt["n_entity"],
                self.opt["n_concept"],
            )
            train_dataset_loader = torch.utils.data.DataLoader(
                dataset=train_set, batch_size=self.batch_size, shuffle=False
            )
            num = 0
            for (
                context,
                c_lengths,
                response,
                r_length,
                mask_response,
                mask_r_length,
                entity,
                entity_vector,
                movie,
                concept_mask,
                dbpedia_mask,
                concept_vec,
                db_vec,
                rec,
            ) in tqdm(train_dataset_loader):
                seed_sets = []
                batch_words = []
                batch_entities = []
                batch_labels = []
                iterations += 1
                batch_size = context.shape[0]
                lookup_words = []
                for b in range(batch_size):
                    
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    words = concept_vec[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)
                    batch_words.extend(words)
                    batch_entities.extend(seed_set)
                    lookup_words.append(words)
                    
                    temp = [(x,y) for x in words for y in seed_set]
                    batch_labels.append(temp)
                
#                 batch_entities.extend(movie.detach().cpu().numpy().tolist())
                word_item_pairs = [(x,y) for x in batch_words for y in batch_entities]
                word_item_pairs.extend([(y,x) for x in batch_words for y in batch_entities])
                    
                self.model.train()
                self.zero_grad()

                (
                    scores,
                    preds,
                    rec_scores,
                    rec_loss,
                    gen_loss,
                    mask_loss,
                    info_db_loss,
                    info_con_loss,
                ) = self.model(
                    context.cuda(),
                    response.cuda(),
                    mask_response.cuda(),
                    concept_mask,
                    dbpedia_mask,
                    seed_sets,
                    movie,
                    concept_vec,
                    db_vec,
                    entity_vector.cuda(),
                    rec,
                    word_item_pairs,
                    lookup_words,
                    batch_labels,
                    test=False,
                    pretrain = False
                )

                joint_loss = (
                    rec_loss + self.opt['info_loss_ratio'] * info_db_loss
                )  # +0.0*info_con_loss#+mask_loss*0.05
                
#                 wandb.log({"train_rec_loss": rec_loss})
#                 wandb.log({"train_info_loss": info_db_loss})

                losses.append([rec_loss, info_db_loss])
                self.backward(joint_loss)
                self.update_params()
                if num % 50 == 0:
                    print(
                        "rec loss is %f" % (sum([l[0] for l in losses]) / len(losses))
                    )
                    print(
                        "info db loss is %f"
                        % (sum([l[1] for l in losses]) / len(losses))
                    )
                    losses = []
                    
                if iterations % 400 == 0:
                    print(f"Evaluate model on test set at {iterations} step....")
                    output_metrics_rec = self.val(is_test=True)
                    self.logs[iterations] = {
                        "recall@1": output_metrics_rec["recall@1"],
                        "recall@10": output_metrics_rec["recall@10"],
                        "recall@50": output_metrics_rec["recall@50"],
                    }
                    
                num += 1
            output_metrics_rec = self.val()

            if (
                best_val_rec
                > output_metrics_rec["recall@50"] + output_metrics_rec["recall@1"]
            ):
                rec_stop = True
            else:
                best_val_rec = (
                    output_metrics_rec["recall@50"] + output_metrics_rec["recall@1"]
                )
                self.model.save_model(name = 'best_recommendation_model.pkl')
                print(
                    "recommendation model saved once------------------------------------------------"
                )
            if rec_stop == True:
                break

        _ = self.val(is_test=True)

        print("Saving logs .........")
        save_logs(self.logs, self.log_file_name)
        print('Saving latent vectors ......')
        self.save_latent_vectors()
        self.model.save_model(name = 'concept_final_recommendation_model.pkl')

    def metrics_cal_rec(self, rec_loss, scores, labels):
        batch_size = len(labels.view(-1).tolist())
        self.metrics_rec["loss"] += rec_loss
        outputs = scores.cpu()
        
        #get all prediction scores of movies
        #movie_ids is a list of indexes of movies in the entity2entityid_dict
        outputs = outputs[:, torch.LongTensor(self.movie_ids)]
        #pred_idx = top_k indices of preidction scores
        _, pred_idx = torch.topk(outputs, k=100, dim=1)
        
        #for all batch
        for b in range(batch_size):
#             if label is a pad movie, then we skip ??
            if labels[b].item() == 0:
                continue
            #get target movie idx by finding the label in the movie_set ids.
            target_idx = self.movie_ids.index(labels[b].item())
#             print(f'target: {target_idx}, label:{labels[b]}, pred: {pred_idx[b][:50]}')
            #if target idx in 
            self.metrics_rec["recall@1"] += int(target_idx in pred_idx[b][:1].tolist())
            self.metrics_rec["recall@10"] += int(
                target_idx in pred_idx[b][:10].tolist()
            )
            self.metrics_rec["recall@50"] += int(
                target_idx in pred_idx[b][:50].tolist()
            )
            self.metrics_rec["count"] += 1

    def val(self, is_test=False):
        self.metrics_gen = {
            "ppl": 0,
            "dist1": 0,
            "dist2": 0,
            "dist3": 0,
            "dist4": 0,
            "bleu1": 0,
            "bleu2": 0,
            "bleu3": 0,
            "bleu4": 0,
            "count": 0,
        }
        self.metrics_rec = {
            "recall@1": 0,
            "recall@10": 0,
            "recall@50": 0,
            "loss": 0,
            "gate": 0,
            "count": 0,
            "gate_count": 0,
        }
        self.model.eval()
        if is_test:
            val_dataset = dataset("data/test_data.jsonl", self.opt)
        else:
            val_dataset = dataset("data/valid_data.jsonl", self.opt)
        val_set = CRSdataset(
            val_dataset.data_process(), self.opt["n_entity"], self.opt["n_concept"]
        )
        val_dataset_loader = torch.utils.data.DataLoader(
            dataset=val_set, batch_size=self.batch_size, shuffle=False
        )
        recs = []
        for (
            context,
            c_lengths,
            response,
            r_length,
            mask_response,
            mask_r_length,
            entity,
            entity_vector,
            movie,
            concept_mask,
            dbpedia_mask,
            concept_vec,
            db_vec,
            rec,
        ) in val_dataset_loader:
            with torch.no_grad():
                seed_sets = []
                batch_words = []
                batch_entities = []
                lookup_words = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    words = concept_vec[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)
                    batch_words.extend(words)
                    lookup_words.append(words)
                    batch_entities.extend(seed_set)

                word_item_pairs = [(x,y) for x in batch_words for y in batch_entities]
                word_item_pairs.extend([(y,x) for x in batch_words for y in batch_entities])
                
                (
                    scores,
                    preds,
                    rec_scores,
                    rec_loss,
                    _,
                    mask_loss,
                    info_db_loss,
                    info_con_loss,
                ) = self.model(
                    context.cuda(),
                    response.cuda(),
                    mask_response.cuda(),
                    concept_mask,
                    dbpedia_mask,
                    seed_sets,
                    movie,
                    concept_vec,
                    db_vec,
                    entity_vector.cuda(),
                    rec,
                    word_item_pairs,
                    lookup_words,
                    test=True,
                    maxlen=20,
                    bsz=batch_size,
                    pretrain = False
                )
                
#             wandb.log({"test_rec_loss": rec_loss})
#             wandb.log({"test_info_loss": info_db_loss})

            recs.extend(rec.cpu())
            # print(losses)
            # exit()
            self.metrics_cal_rec(rec_loss, rec_scores, movie)

        output_dict_rec = {
            key: self.metrics_rec[key] / self.metrics_rec["count"]
            for key in self.metrics_rec
        }
        print(output_dict_rec)

        return output_dict_rec
    
    def error_analysis(self, is_test=True):
        
        print('load model ........')
        self.model.load_model(name = 'recommendation_model.pkl')
        
        if is_test:
            val_dataset = dataset("data/test_data.jsonl", self.opt)
        else:
            val_dataset = dataset("data/valid_data.jsonl", self.opt)
        val_set = CRSdataset(
            val_dataset.data_process(), self.opt["n_entity"], self.opt["n_concept"]
        )
        val_dataset_loader = torch.utils.data.DataLoader(
            dataset=val_set, batch_size=self.batch_size, shuffle=False
        )
        recs = []
        
        results = []
        for (
            context,
            c_lengths,
            response,
            r_length,
            mask_response,
            mask_r_length,
            entity,
            entity_vector,
            movie,
            concept_mask,
            dbpedia_mask,
            concept_vec,
            db_vec,
            one_hop_vec,
            two_hop_vec,
            rec,
        ) in tqdm(val_dataset_loader):
            with torch.no_grad():
                seed_sets = []
                batch_words = []
                batch_entities = []
                lookup_words = []
                batch_size = context.shape[0]
                one_hop_seed_sets = []
                two_hop_seed_sets = []
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    words = concept_vec[b].nonzero().view(-1).tolist()
                    one_hop_set = one_hop_vec[b].nonzero().view(-1).tolist()
                    two_hop_set = two_hop_vec[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)
                    batch_words.extend(words)
                    lookup_words.append(words)
                    batch_entities.extend(seed_set)
                    one_hop_seed_sets.append(one_hop_set)
                    two_hop_seed_sets.append(two_hop_set)         
                    
                word_item_pairs = [(x,y) for x in batch_words for y in batch_entities]
                word_item_pairs.extend([(y,x) for x in batch_words for y in batch_entities])
                
                (
                    scores,
                    preds,
                    rec_scores,
                    rec_loss,
                    _,
                    mask_loss,
                    info_db_loss,
                    info_con_loss,
                ) = self.model(
                    context.cuda(),
                    response.cuda(),
                    mask_response.cuda(),
                    concept_mask,
                    dbpedia_mask,
                    seed_sets,
                    movie,
                    concept_vec,
                    db_vec,
                    entity_vector.cuda(),
                    rec,
                    word_item_pairs,
                    lookup_words,
                    one_hop_seed_sets,
                    two_hop_seed_sets,
                    test=True,
                    maxlen=20,
                    bsz=batch_size,
                    pretrain = False
                )
                
#             wandb.log({"test_rec_loss": rec_loss})
#             wandb.log({"test_info_loss": info_db_loss})

            outputs = rec_scores.cpu()
            #get all prediction scores of movies
            #movie_ids is a list of indexes of movies in the entity2entityid_dict
            outputs = outputs[:, torch.LongTensor(self.movie_ids)]
            #pred_idx = top_k indices of preidction scores
            _, pred_idx = torch.topk(outputs, k=100, dim=1)

            #for all batch
            for b in range(batch_size):
    #             if label is a pad movie, then we skip ??
                if movie[b].item() == 0:
                    continue
                #get target movie idx by finding the label in the movie_set ids.
                target_idx = self.movie_ids.index(movie[b].item())
                
                result_dict = {}
                result_dict['context'] = concept_mask[b].cpu().detach().numpy().tolist()
                result_dict['seed_sets'] = seed_sets[b]
                
                #get top-50 prediction
                result_dict['prediction'] = pred_idx[b][:10].tolist()
                result_dict['label'] = target_idx
                result_dict['movie'] = movie[b]
                
                results.append(result_dict)
        
        with open('results.pkl','wb') as f:
            pickle.dump(results, f)


    @classmethod
    def optim_opts(self):
        """
        Fetch optimizer selection.

        By default, collects everything in torch.optim, as well as importing:
        - qhm / qhmadam if installed from github.com/facebookresearch/qhoptim

        Override this (and probably call super()) to add your own optimizers.
        """
        # first pull torch.optim in
        optims = {
            k.lower(): v
            for k, v in optim.__dict__.items()
            if not k.startswith("__") and k[0].isupper()
        }
        try:
            import apex.optimizers.fused_adam as fused_adam

            optims["fused_adam"] = fused_adam.FusedAdam
        except ImportError:
            pass

        try:
            # https://openreview.net/pdf?id=S1fUpoR5FQ
            from qhoptim.pyt import QHM, QHAdam

            optims["qhm"] = QHM
            optims["qhadam"] = QHAdam
        except ImportError:
            # no QHM installed
            pass

        return optims

    def init_optim(self, params, optim_states=None, saved_optim_type=None):
        """
        Initialize optimizer with model parameters.

        :param params:
            parameters from the model

        :param optim_states:
            optional argument providing states of optimizer to load

        :param saved_optim_type:
            type of optimizer being loaded, if changed will skip loading
            optimizer states
        """

        opt = self.opt

        # set up optimizer args
        lr = opt["learningrate"]
        kwargs = {"lr": lr}
        kwargs["amsgrad"] = True
        kwargs["betas"] = (0.9, 0.999)

        optim_class = self.optim_opts()[opt["optimizer"]]
        self.optimizer = optim_class(params, **kwargs)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1600, gamma=0.4)

    def backward(self, loss):
        """
        Perform a backward pass. It is recommended you use this instead of
        loss.backward(), for integration with distributed training and FP16
        training.
        """
        loss.backward()

    def update_params(self, pretrain = False):
        """
        Perform step of optimization, clipping gradients and adjusting LR
        schedule if needed. Gradient accumulation is also performed if agent
        is called with --update-freq.

        It is recommended (but not forced) that you call this in train_step.
        """
        update_freq = 1
        if update_freq > 1:
            # we're doing gradient accumulation, so we don't only want to step
            # every N updates instead
            self._number_grad_accum = (self._number_grad_accum + 1) % update_freq
            if self._number_grad_accum != 0:
                return

        if self.opt["gradient_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.opt["gradient_clip"]
            )

        self.optimizer.step()
        if not pretrain:
            self.lr_scheduler.step()
    

    def zero_grad(self):
        """
        Zero out optimizer.

        It is recommended you call this in train_step. It automatically handles
        gradient accumulation if agent is called with --update-freq.
        """
        self.optimizer.zero_grad()
    
    def save_latent_vectors(self):
        
        db_node_vectors, concept_node_vectors = self.model.save_latent_vectors()
        
        db_node_vectors = db_node_vectors.detach().cpu().numpy()
        concept_node_vectors = concept_node_vectors.detach().cpu().numpy()
        
        with open('db_latent_vectors.pkl','wb') as f:
            pickle.dump(db_node_vectors, f)
        
        with open('concept_node_vectors.pkl','wb') as f:
            pickle.dump(concept_node_vectors, f)


class TrainLoop_fusion_gen:
    def __init__(self, opt, is_finetune):
        self.opt = opt
        self.train_dataset = dataset("data/train_data.jsonl", opt)

        self.dict = self.train_dataset.word2index
        self.index2word = {self.dict[key]: key for key in self.dict}

        self.batch_size = self.opt["batch_size"]
        self.epoch = self.opt["epoch"]

        self.use_cuda = opt["use_cuda"]
        if opt["load_dict"] != None:
            self.load_data = True
        else:
            self.load_data = False
        self.is_finetune = False

        self.movie_ids = pkl.load(open("generated_data/final_movie_ids.pkl", "rb"))
        # Note: we cannot change the type of metrics ahead of time, so you
        # should correctly initialize to floats or ints here

        self.metrics_rec = {
            "recall@1": 0,
            "recall@10": 0,
            "recall@50": 0,
            "loss": 0,
            "count": 0,
        }
        self.metrics_gen = {
            "dist1": 0,
            "dist2": 0,
            "dist3": 0,
            "dist4": 0,
            "bleu1": 0,
            "bleu2": 0,
            "bleu3": 0,
            "bleu4": 0,
            "count": 0,
        }

        self.build_model(is_finetune=True)

        if opt["load_dict"] is not None:
            # load model parameters if available
            print(
                "[ Loading existing model params from {} ]" "".format(opt["load_dict"])
            )
            states = self.model.load(opt["load_dict"])
        else:
            states = {}

        self.init_optim(
            [p for p in self.model.parameters() if p.requires_grad],
            optim_states=states.get("optimizer"),
            saved_optim_type=states.get("optimizer_type"),
        )

    def build_model(self, is_finetune):
        self.model = CrossModel(self.opt, self.dict, is_finetune)
        if self.opt["embedding_type"] != "random":
            pass
        if self.use_cuda:
            self.model.cuda()

    def train(self):
        self.model.load_model(name='concept_final_recommendation_model.pkl')
        losses = []
        best_val_gen = -1
        gen_stop = False
        for i in range(self.epoch * 3):

            print(f'Epoch: {i+ 1}')

            train_set = CRSdataset(
                self.train_dataset.data_process(True),
                self.opt["n_entity"],
                self.opt["n_concept"],
            )
            train_dataset_loader = torch.utils.data.DataLoader(
                dataset=train_set, batch_size=self.batch_size, shuffle=False
            )
            num = 0
            for (
                context,
                c_lengths,
                response,
                r_length,
                mask_response,
                mask_r_length,
                entity,
                entity_vector,
                movie,
                concept_mask,
                dbpedia_mask,
                concept_vec,
                db_vec,
                rec,
            ) in tqdm(train_dataset_loader):
                seed_sets = []
                batch_words = []
                batch_entities = []
                lookup_words = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    words = concept_vec[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)
                    batch_words.extend(words)
                    lookup_words.append(words)
                    batch_entities.extend(seed_set)
                
#                 batch_entities.extend(movie.detach().cpu().numpy().tolist())
                    
                word_item_pairs = [(x,y) for x in batch_words for y in batch_entities]
                word_item_pairs.extend([(y,x) for x in batch_words for y in batch_entities])
                self.model.train()
                self.zero_grad()

                (
                    scores,
                    preds,
                    rec_scores,
                    rec_loss,
                    gen_loss,
                    mask_loss,
                    info_db_loss,
                    info_con_loss,
                ) = self.model(
                    context.cuda(),
                    response.cuda(),
                    mask_response.cuda(),
                    concept_mask,
                    dbpedia_mask,
                    seed_sets,
                    movie,
                    concept_vec,
                    db_vec,
                    entity_vector.cuda(),
                    rec,
                    word_item_pairs,
                    lookup_words,
                    test=False,
                )

                joint_loss = gen_loss
                losses.append([gen_loss])
                self.backward(joint_loss)
                self.update_params()
                if num % 50 == 0:
                    print(
                        "gen loss is %f" % (sum([l[0] for l in losses]) / len(losses))
                    )
                    losses = []
                num += 1

            output_metrics_gen = self.val(is_test=False)
            
            if best_val_gen > output_metrics_gen["dist4"]:
                pass
            else:
                best_val_gen = output_metrics_gen["dist4"]
#                 self.model.save_model()
                print(
                    "generator model saved once------------------------------------------------"
                )

        _ = self.val(is_test=True)

    def val(self, is_test=False):
        self.metrics_gen = {
            "ppl": 0,
            "dist1": 0,
            "dist2": 0,
            "dist3": 0,
            "dist4": 0,
            "bleu1": 0,
            "bleu2": 0,
            "bleu3": 0,
            "bleu4": 0,
            "count": 0,
        }
        self.metrics_rec = {
            "recall@1": 0,
            "recall@10": 0,
            "recall@50": 0,
            "loss": 0,
            "gate": 0,
            "count": 0,
            "gate_count": 0,
        }
        self.model.eval()
        if is_test:
            val_dataset = dataset("data/test_data.jsonl", self.opt)
        else:
            val_dataset = dataset("data/valid_data.jsonl", self.opt)
        val_set = CRSdataset(
            val_dataset.data_process(True), self.opt["n_entity"], self.opt["n_concept"]
        )
        val_dataset_loader = torch.utils.data.DataLoader(
            dataset=val_set, batch_size=self.batch_size, shuffle=False
        )
        inference_sum = []
        golden_sum = []
        context_sum = []
        losses = []
        recs = []
        for (
            context,
            c_lengths,
            response,
            r_length,
            mask_response,
            mask_r_length,
            entity,
            entity_vector,
            movie,
            concept_mask,
            dbpedia_mask,
            concept_vec,
            db_vec,
            rec,
        ) in tqdm(val_dataset_loader):
            with torch.no_grad():
                seed_sets = []
                batch_words = []
                batch_entities = []
                lookup_words = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    words = concept_vec[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)
                    batch_words.extend(words)
                    lookup_words.append(words)
                    batch_entities.extend(seed_set)
                
                word_item_pairs = [(x,y) for x in batch_words for y in batch_entities]
                word_item_pairs.extend([(y,x) for x in batch_words for y in batch_entities])
                
                (
                    _,
                    _,
                    _,
                    _,
                    gen_loss,
                    mask_loss,
                    info_db_loss,
                    info_con_loss,
                ) = self.model(
                    context.cuda(),
                    response.cuda(),
                    mask_response.cuda(),
                    concept_mask,
                    dbpedia_mask,
                    seed_sets,
                    movie,
                    concept_vec,
                    db_vec,
                    entity_vector.cuda(),
                    rec,
                    word_item_pairs,
                    lookup_words,
                    test=False,
                )
                (
                    scores,
                    preds,
                    rec_scores,
                    rec_loss,
                    _,
                    mask_loss,
                    info_db_loss,
                    info_con_loss,
                ) = self.model(
                    context.cuda(),
                    response.cuda(),
                    mask_response.cuda(),
                    concept_mask,
                    dbpedia_mask,
                    seed_sets,
                    movie,
                    concept_vec,
                    db_vec,
                    entity_vector.cuda(),
                    rec,
                    word_item_pairs,
                    lookup_words,
                    test=True,
                    maxlen=20,
                    bsz=batch_size,
                )

            golden_sum.extend(self.vector2sentence(response.cpu()))
            inference_sum.extend(self.vector2sentence(preds.cpu()))
            context_sum.extend(self.vector2sentence(context.cpu()))
            recs.extend(rec.cpu())
            losses.append(torch.mean(gen_loss))
            # print(losses)
            # exit()

        self.metrics_cal_gen(losses, inference_sum, golden_sum, recs)

        output_dict_gen = {}
        for key in self.metrics_gen:
            if "bleu" in key:
                output_dict_gen[key] = self.metrics_gen[key] / self.metrics_gen["count"]
            else:
                output_dict_gen[key] = self.metrics_gen[key]
        print(output_dict_gen)

        f = open("context_test.txt", "w", encoding="utf-8")
        f.writelines([" ".join(sen) + "\n" for sen in context_sum])
        f.close()

        f = open("output_test.txt", "w", encoding="utf-8")
        f.writelines([" ".join(sen) + "\n" for sen in inference_sum])
        f.close()
        return output_dict_gen

    def metrics_cal_gen(self, rec_loss, preds, responses, recs):
        def bleu_cal(sen1, tar1):
            bleu1 = sentence_bleu([tar1], sen1, weights=(1, 0, 0, 0))
            bleu2 = sentence_bleu([tar1], sen1, weights=(0, 1, 0, 0))
            bleu3 = sentence_bleu([tar1], sen1, weights=(0, 0, 1, 0))
            bleu4 = sentence_bleu([tar1], sen1, weights=(0, 0, 0, 1))
            return bleu1, bleu2, bleu3, bleu4

        def distinct_metrics(outs):
            # outputs is a list which contains several sentences, each sentence contains several words
            unigram_count = 0
            bigram_count = 0
            trigram_count = 0
            quagram_count = 0
            unigram_set = set()
            bigram_set = set()
            trigram_set = set()
            quagram_set = set()
            for sen in outs:
                for word in sen:
                    unigram_count += 1
                    unigram_set.add(word)
                for start in range(len(sen) - 1):
                    bg = str(sen[start]) + " " + str(sen[start + 1])
                    bigram_count += 1
                    bigram_set.add(bg)
                for start in range(len(sen) - 2):
                    trg = (
                        str(sen[start])
                        + " "
                        + str(sen[start + 1])
                        + " "
                        + str(sen[start + 2])
                    )
                    trigram_count += 1
                    trigram_set.add(trg)
                for start in range(len(sen) - 3):
                    quag = (
                        str(sen[start])
                        + " "
                        + str(sen[start + 1])
                        + " "
                        + str(sen[start + 2])
                        + " "
                        + str(sen[start + 3])
                    )
                    quagram_count += 1
                    quagram_set.add(quag)
            dis1 = len(unigram_set) / len(outs)  # unigram_count
            dis2 = len(bigram_set) / len(outs)  # bigram_count
            dis3 = len(trigram_set) / len(outs)  # trigram_count
            dis4 = len(quagram_set) / len(outs)  # quagram_count
            return dis1, dis2, dis3, dis4

        predict_s = preds
        golden_s = responses
        # print(rec_loss[0])
        # self.metrics_gen["ppl"]+=sum([exp(ppl) for ppl in rec_loss])/len(rec_loss)
        generated = []

        for out, tar, rec in zip(predict_s, golden_s, recs):
            bleu1, bleu2, bleu3, bleu4 = bleu_cal(out, tar)
            generated.append(out)
            self.metrics_gen["bleu1"] += bleu1
            self.metrics_gen["bleu2"] += bleu2
            self.metrics_gen["bleu3"] += bleu3
            self.metrics_gen["bleu4"] += bleu4
            self.metrics_gen["count"] += 1

        dis1, dis2, dis3, dis4 = distinct_metrics(generated)
        self.metrics_gen["dist1"] = dis1
        self.metrics_gen["dist2"] = dis2
        self.metrics_gen["dist3"] = dis3
        self.metrics_gen["dist4"] = dis4

    def vector2sentence(self, batch_sen):
        sentences = []
        for sen in batch_sen.numpy().tolist():
            sentence = []
            for word in sen:
                if word > 3:
                    sentence.append(self.index2word[word])
                elif word == 3:
                    sentence.append("_UNK_")
            sentences.append(sentence)
        return sentences

    @classmethod
    def optim_opts(self):
        """
        Fetch optimizer selection.

        By default, collects everything in torch.optim, as well as importing:
        - qhm / qhmadam if installed from github.com/facebookresearch/qhoptim

        Override this (and probably call super()) to add your own optimizers.
        """
        # first pull torch.optim in
        optims = {
            k.lower(): v
            for k, v in optim.__dict__.items()
            if not k.startswith("__") and k[0].isupper()
        }
        try:
            import apex.optimizers.fused_adam as fused_adam

            optims["fused_adam"] = fused_adam.FusedAdam
        except ImportError:
            pass

        try:
            # https://openreview.net/pdf?id=S1fUpoR5FQ
            from qhoptim.pyt import QHM, QHAdam

            optims["qhm"] = QHM
            optims["qhadam"] = QHAdam
        except ImportError:
            # no QHM installed
            pass

        return optims

    def init_optim(self, params, optim_states=None, saved_optim_type=None):
        """
        Initialize optimizer with model parameters.

        :param params:
            parameters from the model

        :param optim_states:
            optional argument providing states of optimizer to load

        :param saved_optim_type:
            type of optimizer being loaded, if changed will skip loading
            optimizer states
        """

        opt = self.opt

        # set up optimizer args
        lr = opt["learningrate"]
        kwargs = {"lr": lr}
        kwargs["amsgrad"] = True
        kwargs["betas"] = (0.9, 0.999)

        optim_class = self.optim_opts()[opt["optimizer"]]
        self.optimizer = optim_class(params, **kwargs)

    def backward(self, loss):
        """
        Perform a backward pass. It is recommended you use this instead of
        loss.backward(), for integration with distributed training and FP16
        training.
        """
        loss.backward()

    def update_params(self):
        """
        Perform step of optimization, clipping gradients and adjusting LR
        schedule if needed. Gradient accumulation is also performed if agent
        is called with --update-freq.

        It is recommended (but not forced) that you call this in train_step.
        """
        update_freq = 1
        if update_freq > 1:
            # we're doing gradient accumulation, so we don't only want to step
            # every N updates instead
            self._number_grad_accum = (self._number_grad_accum + 1) % update_freq
            if self._number_grad_accum != 0:
                return

        if self.opt["gradient_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.opt["gradient_clip"]
            )

        self.optimizer.step()

    def zero_grad(self):
        """
        Zero out optimizer.

        It is recommended you call this in train_step. It automatically handles
        gradient accumulation if agent is called with --update-freq.
        """
        self.optimizer.zero_grad()


if __name__ == "__main__":
    args = setup_args().parse_args()
    print(vars(args))
    seed_all(vars(args)["random_seed"])
#     wandb.config.update(args)
    if args.is_finetune == False:
        loop = TrainLoop_fusion_rec(vars(args), is_finetune=False)
        # loop.model.load_model()
        loop.train()
    else:
        loop = TrainLoop_fusion_gen(vars(args), is_finetune=True)
        # loop.train()
        loop.model.load_model(name='final_recommendation_model.pkl')
        # met = loop.val(True)
        loop.train()
    
#     if args.error_analysis:
#         loop = TrainLoop_fusion_rec(vars(args), is_finetune=False)
#         loop.error_analysis()
    
    met = loop.val(True)
    # print(met)
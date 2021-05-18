import sys
import os
import json
import pickle
import datetime

import torch
from torch.utils.data import DataLoader

from dataset import load_data
from model import ContextModel
from loss import SmoothLoss

from evaluate import ANETcaptions


def hyper_parameter():
    # pre-defined hyper parameters for the experiment
    hyper_p = {
        'train_path': './data/train.pt',
        'val_path': './data/val/pt',
        'reference_path': './data/reference.json',
        'log_path': f'./log/{datetime.datetime.now().strftime("%Y/%m/%d-%H:%M:%S")}',
        'lr': 1e-4,
        'epoch': 30,
        'batch_size': 32,
        'smooth': 0.4,
        'dim_hidden': 1024,
        'dim_ff': 128,
        'n_head': 4,
        'n_layer': 2,
        'dropout': 0.4
    }

    return hyper_p


def evaluate(hypothesis_path, reference_path):
    # block verbose output
    text_trap = io.StringIO()
    sys.stdout = text_trap

    metrics = {}
    prediction_fields = ['results']
    evaluator = ANETcaptions(
        reference_path, hypothesis_path, [1.0],
        1000, prediction_fields, False)
    evaluator.evaluate()

    for i, tiou in enumerate([1.0]):
        metrics[tiou] = {}

        for metric in evaluator.scores:
            score = evaluator.scores[metric][i]
            metrics[tiou][metric] = score

    metrics['Average across tIoUs'] = {}
    for metric in evaluator.scores:
        score = evaluator.scores[metric]
        metrics['Average across tIoUs'][metric] = sum(score) / float(len(score))

    sys.stdout = sys.__stdout__
    return metrics


def decode(model, v, t, v_mask, t_mask, tokenizer, max_len):
    model.eval()

    # construct the init sentences with only bos token_ids
    batch_size = v[0].size(0)
    complete = torch.zeros(curr_batch_size, 1).byte().cuda()
    target = (torch.ones(curr_batch_size, 1) * tokenizer.cls_token_id).long().cuda()
    with torch.no_grad():
        while target.size(1) < max_len and not all(complete):
            target_mask = (label != 0).unsqueeze(-2) \
                         & torch.tril(torch.ones(1, label.size(1), label.size(1)), 0)
            model(v, t, target, v_mask, t_mask, target_mask)

            # use greedy search as the generation strategy
            next_word = output[:, -1].max(dim=-1)[1].unsqueeze(1)
            target = torch.cat([target, next_word], dim=-1)
            complete = complete | torch.eq(next_word, self.eos_idx).byte()
    return target


def generate(model, val_data, tokenizer, logging_path):
    predictions = {}

    for i, batch in enumerate(val_data):

        v = [batch['prev_v'], batch['v'], batch['next_v']]
        t = [batch['prev_t'], batch['t'], batch['next_t']]
        v_mask = [batch['prev_v_mask'], batch['v_mask'], batch['next_v_mask']]
        t_mask = [batch['prev_t_mask'], batch['t_mask'], batch['next_t_mask']]

        # generate output tokens
        predict_ids = decode(model, v, t, v_mask, t_mask, tokenizer).long().cpu()
        predict_sentences = tokenizer.batch_decode(predict_ids)
        predict_words = [sentence.split() for sentence in predict_sentences]

        # save output captions
        for idx, each_video in enumerate(batch['video_id']):
            curr_start = batch['start'][idx]
            curr_end = batch['end'][idx]
            sent = predict_words[idx]
            sent = ' '.join([word for word in sent if word not in tokenizer.all_special_tokens]).capitalize()
            if each_video not in predictions:
                predictions[each_video] = []
            predictions[each_video].append(
                {"sentence": sent, "timestamp": [curr_start.item(), curr_end.item()]}
            )

    # save result file
    result_path = os.path.join(logging_path, 'result.json')
    json.dump(predictions, open(result_path, 'w'))
    return result_path, predictions


def train_loop(model, train_data, loss_fn, optimizer):
    # train the model
    total_loss = []

    model.train()
    for i, batch in enumerate(train_data):

        # pack input data for the model
        v = [batch['prev_v'], batch['v'], batch['next_v']]
        t = [batch['prev_t'], batch['t'], batch['next_t']]
        v_mask = [batch['prev_v_mask'], batch['v_mask'], batch['next_v_mask']]
        t_mask = [batch['prev_t_mask'], batch['t_mask'], batch['next_t_mask']]

        output = model(v, t, batch['label'][:, :-1], v_mask, t_mask, batch['label_mask'][:, :-1])

        y = batch['label'][:, 1:]
        n_tokens = (y != 0).sum()
        curr_loss = loss_fn(output, y, n_tokens)

        total_loss.append(curr_loss.item())

        optimizer.zero_grad()
        curr_loss.backward()
        optimizer.step()

    avg_loss = mean(total_loss)

    return avg_loss


def eval_loop(model, val_data, loss_fn, tokenizer, logging_path, reference):
    # evaluate the model
    total_loss = []
    metrics = {}

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_data):
            # pack input data for the model
            v = [batch['prev_v'], batch['v'], batch['next_v']]
            t = [batch['prev_t'], batch['t'], batch['next_t']]
            v_mask = [batch['prev_v_mask'], batch['v_mask'], batch['next_v_mask']]
            t_mask = [batch['prev_t_mask'], batch['t_mask'], batch['next_t_mask']]

            output = model(v, t, batch['label'][:, :-1], v_mask, t_mask, batch['label_mask'][:, :-1])

            y = batch['label'][:, 1:]
            n_tokens = (y != 0).sum()
            curr_loss = loss_fn(output, y, n_tokens)

            total_loss.append(curr_loss.item())

    avg_loss = mean(total_loss)

    hypothesis_path, output = generate(model, val_data, tokenizer, logging_path)
    scores = evaluate(hypothsis_path, reference)

    metrics['b4'] = scores['Average across tIoUs']['Bleu_4'] * 100
    metrics['m'] = scores['Average across tIoUs']['METEOR'] * 100
    metrics['r'] = scores['Average across tIoUs']['ROUGE_L'] * 100
    metrics['c'] = scores['Average across tIoUs']['CIDEr'] * 100


    return avg_loss, metrics, output


def main():

    # get hyper parameters
    config = hyper_parameter()

    # prepare data
    train_data = load_data(config['train_path'])
    val_data = load_data(config['val_path'])
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], drop_last=True)

    # prepare model
    model = ContextModel(dim_text=train_data.dim_text, dim_video=train_data.dim_video, **config).cuda()

    # prepare loss and optimizer
    loss_fn = SmoothLoss(config['smooth'])
    optimizer = torch.optim.Adam(model.parameters(), config['lr'])

    # record the result
    loss_record = {
        'train_loss': [],
        'val_loss': []
    }
    metrics_record = {
        'b4': [],
        'm': [],
        'r': [],
        'c': []
    }

    # run experiment
    for epoch in range(config['epochs']):
        train_loss = train_loop(model, train_loader, loss_fn, optimizer)
        val_loss, metrics, output = eval_loop(model, val_loader, loss_fn, config['reference_path'])

        # record the loss and metrics scores
        b4, meteor, rough, cider = metrics.values()
        loss_record['train_loss'].append(train_loss)
        loss_record['val_loss'].append(val_loss)
        metrics_record['b4'].append(b4)
        metrics_record['m'].append(meteor)
        metrics_record['r'].append(rough)
        metrics_record['c'].append(cider)

        # save the generated output
        if not os.path.exists(config['log_path']):
            os.makedirs(config['log_path'])
        json.dump(output, open(os.path.join(config['log_path'], f'output_{epoch:02}.json')))

    json.dump(loss_record, open(os.path.join(config['log_path'], 'loss.json')))
    json.dump(metrics_record, open(os.path.join(config['log_path'], 'metrics.json')))


if __name__ == '__main__':
    main()

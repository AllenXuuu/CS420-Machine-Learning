from dataloader import build_dataloader
from losses import build_loss
from models import build_model
from utils import parse_args, evaluate
import tqdm
import torch
import os
import numpy as np
import time


def main():
    args = parse_args()
    print(args)
    assert args.pretrained is not None, 'Please give a path of pre-trained weight'

    valLoader = build_dataloader('test', args.bz)

    device = torch.device('cuda')
    model = build_model(args, 1, 2).to(device)
    print('Load model from <== %s' % args.pretrained)
    model.load_state_dict(torch.load(args.pretrained))
    loss_func = build_loss(args.loss)

    ######################################## Eval
    model.eval()
    val_score = []
    val_label = []
    val_loss = []
    start_time = time.time()
    with torch.no_grad():
        for step, (img, label) in enumerate(
                tqdm.tqdm(valLoader, desc='Val', ncols=75, leave=False)):
            img = img.float().unsqueeze(1).to(device)
            label = label.long().to(device)
            score = model(img)

            val_score.append(score)
            val_label.append(label)
            loss = loss_func(score, label)
            val_loss.append(loss.item())

    val_loss = np.mean(val_loss)
    val_label = torch.cat(val_label, 0).cpu().numpy()
    val_score = torch.cat(val_score, 0)
    val_pred = torch.argmax(val_score, 1).cpu().numpy()
    val_result = evaluate(val_pred, val_label)
    with open('pred.pkl', 'wb') as f:
        torch.save(val_pred, f)

    val_report = 'Val   | Time %.2fs | loss %.4f |' % (time.time() - start_time, val_loss)
    for metric, score in val_result.items():
        val_report += ' %s %.4f |' % (metric, score)
    print(val_report)

    print('Finish Evaluation!')


if __name__ == '__main__':
    main()

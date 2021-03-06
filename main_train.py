from dataloader import build_dataloader
from losses import build_loss
from models import build_model
from utils import parse_args, evaluate
from tensorboardX import SummaryWriter
import tqdm
import torch
import os
import numpy as np
import time


def main():
    args = parse_args()
    print(args)

    trainLoader = build_dataloader('train', args.bz)
    valLoader = build_dataloader('test', args.bz)

    device = torch.device('cuda')
    model = build_model(args, 1, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_func = build_loss(args.loss)

    ckpt_dir = os.path.join('./checkpoints', args.save_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    log_dir = os.path.join('./logs', args.save_dir)
    writer = SummaryWriter(log_dir)

    for epoch in range(1, args.epoch + 1):
        ######################################## Train
        model.train()
        train_score = []
        train_label = []
        train_loss = []
        start_time = time.time()
        for step, (img, label) in enumerate(
                tqdm.tqdm(trainLoader, desc='Train Epoch %d' % epoch, ncols=75, leave=False)):
            img = img.float().unsqueeze(1).to(device)
            label = label.long().to(device)
            score = model(img)

            train_score.append(score)
            train_label.append(label)
            loss = loss_func(score, label)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = np.mean(train_loss)
        writer.add_scalar('train/loss', train_loss, epoch)
        train_score = torch.cat(train_score, 0)
        train_pred = torch.argmax(train_score, 1).cpu().numpy()
        train_label = torch.cat(train_label, 0).cpu().numpy()
        train_result = evaluate(train_pred, train_label)

        for metric, score in train_result.items():
            writer.add_scalar('train/%s' % metric, score, epoch)

        ######################################## Eval
        model.eval()
        val_score = []
        val_label = []
        val_loss = []
        start_time = time.time()
        with torch.no_grad():
            for step, (img, label) in enumerate(
                    tqdm.tqdm(valLoader, desc='Val   Epoch %d' % epoch, ncols=75, leave=False)):
                img = img.float().unsqueeze(1).to(device)
                label = label.long().to(device)
                score = model(img)

                val_score.append(score)
                val_label.append(label)
                loss = loss_func(score, label)
                val_loss.append(loss.item())

        val_loss = np.mean(val_loss)
        writer.add_scalar('val/loss', val_loss, epoch)

        val_label = torch.cat(val_label, 0).cpu().numpy()
        val_score = torch.cat(val_score, 0)

        val_pred = torch.argmax(val_score, 1).cpu().numpy()
        val_result = evaluate(val_pred, val_label)

        val_report = 'Val   | Epoch %d | Time %.2fs | loss %.4f |' % (epoch, time.time() - start_time, val_loss)
        for metric, score in val_result.items():
            writer.add_scalar('val/%s' % metric, score, epoch)
            val_report += ' %s %.4f |' % (metric, score)
        print(val_report)

        ######################################## Save checkpoint
        if epoch % args.save_freq == 0:
            ckpt_path = os.path.join(ckpt_dir, 'epoch_%d.pth' % epoch)
            torch.save(model.state_dict(), ckpt_path)
            print('Save model into ==> %s' % ckpt_path)

    print('Finish Training!')


if __name__ == '__main__':
    main()

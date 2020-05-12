#!/usr/bin/env python3

import argparse
import torch
import torch.optim as optim
from dataloader import load_data
from terminaltables import AsciiTable
from tensorboardX import SummaryWriter
import eval_widerface
import os
import torchvision_model


def get_args():
    parser = argparse.ArgumentParser(description="Train program for retinaface.")
    parser.add_argument('--data_path', type=str, help='Path for dataset,default WIDERFACE')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Max training epochs')
    parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle dataset or not')
    parser.add_argument('--img_size', type=int, default=640, help='Input image size')
    parser.add_argument('--verbose', type=int, default=10, help='Log verbose')
    parser.add_argument('--save_step', type=int, default=2, help='Save every save_step epochs')
    parser.add_argument('--eval_step', type=int, default=3, help='Evaluate every eval_step epochs')
    parser.add_argument('--save_path', type=str, default='./out', help='Model save path')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    args = parser.parse_args()
    print(args)

    return args


def validate(dataloader_test, model):
    total_loss = 0
    total_classification_loss = 0
    total_bbox_regression_loss = 0
    total_ldm_regression_loss = 0

    amount_of_elems = 0
    for data in dataloader_test:
        classification_loss, bbox_regression_loss, ldm_regression_loss = model(
            [data['img'].cuda().float(), data['annot']]
        )
        classification_loss = classification_loss.mean()
        bbox_regression_loss = bbox_regression_loss.mean()
        ldm_regression_loss = ldm_regression_loss.mean()

        # loss = classification_loss + 1.0 * bbox_regression_loss + 0.5 * ldm_regression_loss
        loss = classification_loss + bbox_regression_loss + 0.5 * ldm_regression_loss

        total_loss += loss.item()
        total_classification_loss += classification_loss.item()
        total_bbox_regression_loss += bbox_regression_loss.item()
        total_ldm_regression_loss += ldm_regression_loss.item()
        amount_of_elems += 1

    total_loss /= amount_of_elems
    total_classification_loss /= amount_of_elems
    total_bbox_regression_loss /= amount_of_elems
    total_ldm_regression_loss /= amount_of_elems

    table_data = [
        ['loss name', 'value'],
        ['total_loss', str(total_loss)],
        ['classification', str(total_classification_loss)],
        ['bbox', str(total_bbox_regression_loss)],
        ['landmarks', str(total_ldm_regression_loss)]
    ]
    table = AsciiTable(table_data)
    log_str = table.table
    print("test loses:")
    print(log_str)


def main():
    args = get_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    log_path = os.path.join(args.save_path, 'log')
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    writer = SummaryWriter(log_dir=log_path)

    data_path = args.data_path
    train_path = os.path.join(data_path, 'train/label.txt')
    val_path = os.path.join(data_path, 'val/label.txt')
    dataloader_train, dataloader_test = load_data(train_path, args.batch_size, split_train_test=True)
    dataloader_val = load_data(val_path, args.batch_size)

    total_batch = len(dataloader_train)

    # Create torchvision model
    retinaface = torchvision_model.create_retinaface().cuda()
    retinaface = torch.nn.DataParallel(retinaface).cuda()
    retinaface.training = True

    optimizer = optim.Adam(retinaface.parameters(), lr=1e-3)

    print('Start to train.')

    epoch_loss = []
    iteration = 0

    for epoch in range(args.epochs):
        retinaface.train()

        # Training
        for iter_num, data in enumerate(dataloader_train):
            optimizer.zero_grad()
            classification_loss, bbox_regression_loss, ldm_regression_loss = retinaface(
                [data['img'].cuda().float(), data['annot']]
            )
            classification_loss = classification_loss.mean()
            bbox_regression_loss = bbox_regression_loss.mean()
            ldm_regression_loss = ldm_regression_loss.mean()

            # loss = classification_loss + 1.0 * bbox_regression_loss + 0.5 * ldm_regression_loss
            loss = classification_loss + bbox_regression_loss + 0.5 * ldm_regression_loss

            loss.backward()
            optimizer.step()

            if iter_num % args.verbose == 0:
                log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, args.epochs, iter_num, total_batch)
                table_data = [
                    ['loss name', 'value'],
                    ['total_loss', str(loss.item())],
                    ['classification', str(classification_loss.item())],
                    ['bbox', str(bbox_regression_loss.item())],
                    ['landmarks', str(ldm_regression_loss.item())]
                ]
                table = AsciiTable(table_data)
                log_str += table.table
                print("train loses:")
                print(log_str)

                # write the log to tensorboard
                writer.add_scalar('losses:', loss.item(), iteration * args.verbose)
                writer.add_scalar('class losses:', classification_loss.item(), iteration * args.verbose)
                writer.add_scalar('box losses:', bbox_regression_loss.item(), iteration * args.verbose)
                writer.add_scalar('landmark losses:', ldm_regression_loss.item(), iteration * args.verbose)
                iteration += 1
                validate(dataloader_test, retinaface)

        # Eval
        if epoch % args.eval_step == 0:
            print('-------- RetinaFace --------')
            print('Evaluating epoch {}'.format(epoch))
            recall, precision = eval_widerface.evaluate(dataloader_val, retinaface)
            print('Recall:', recall)
            print('Precision:', precision)

            writer.add_scalar('Recall:', recall, epoch * args.eval_step)
            writer.add_scalar('Precision:', precision, epoch * args.eval_step)

        # Save model
        if (epoch + 1) % args.save_step == 0:
            torch.save(retinaface.state_dict(), args.save_path + '/model_epoch_{}.pt'.format(epoch + 1))

    writer.close()


if __name__ == '__main__':
    main()

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from dataset import get_loader
import math
from models.ImageDepthNet import ImageDepthNet
import os
import torch.nn.functional as F



def att_loss(pred,gt,rgb_pred,depth_pred):
    rgb_pred = torch.sigmoid(rgb_pred.detach())
    depth_pred = torch.sigmoid(depth_pred.detach())
    w1 = torch.abs(gt-rgb_pred)
    w2 = torch.abs(gt-depth_pred)
    w = (w1+w2)*0.5+1
    attbce=F.binary_cross_entropy_with_logits(pred, gt,weight =w*1.0,reduction='mean')
    return attbce

def save_loss(save_dir, whole_iter_num, epoch_total_loss, epoch_loss, epoch):
    fh = open(save_dir, 'a')
    epoch_total_loss = str(epoch_total_loss)
    epoch_loss = str(epoch_loss)
    fh.write('until_' + str(epoch) + '_run_iter_num' + str(whole_iter_num) + '\n')
    fh.write(str(epoch) + '_epoch_total_loss' + epoch_total_loss + '\n')
    fh.write(str(epoch) + '_epoch_loss' + epoch_loss + '\n')
    fh.write('\n')
    fh.close()


def adjust_learning_rate(optimizer, decay_rate=.1):
    update_lr_group = optimizer.param_groups
    for param_group in update_lr_group:
        print('before lr: ', param_group['lr'])
        param_group['lr'] = param_group['lr'] * decay_rate
        print('after lr: ', param_group['lr'])
    return optimizer


def save_lr(save_dir, optimizer):
    update_lr_group = optimizer.param_groups[0]
    fh = open(save_dir, 'a')
    fh.write('encode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('decode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('\n')
    fh.close()


def train_net(num_gpus, args):
    main(args=args)
    #mp.spawn(main, nprocs=num_gpus, args=(num_gpus, args))


def main(args,num_gpus=1,local_rank=0):
     
    cudnn.benchmark = True

    #dist.init_process_group(backend='nccl', init_method=args.init_method, world_size=num_gpus, rank=local_rank)

    #torch.cuda.set_device(local_rank)
    
    #get_local.activate()
    
    net = ImageDepthNet(args)
    net.train()
    net.cuda()
    #torch.nn.DataParallel(net, device_ids=[0,1])
    #net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    '''
    net = torch.nn.parallel.DistributedDataParallel(
        net,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True)
    '''
    base_params = [params for name, params in net.named_parameters() if ("backbone" in name)]
    other_params = [params for name, params in net.named_parameters() if ("backbone" not in name)]

    optimizer = optim.Adam([{'params': base_params, 'lr': args.lr * 0.1},
                            {'params': other_params, 'lr': args.lr}])
    train_dataset = get_loader(args.trainset, args.data_root, args.img_size, mode='train')
    
    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=num_gpus,
        rank=local_rank,
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4,
                                               pin_memory=True,
                                               sampler=sampler,
                                               drop_last=True,
                                               )

    print('''
        Starting training:
            Train steps: {}
            Batch size: {}
            Learning rate: {}
            Training size: {}
        '''.format(args.train_steps, args.batch_size, args.lr, len(train_loader.dataset)))

    N_train = len(train_loader) * args.batch_size

    loss_weights = [1.5, 0.8, 0.8, 0.5, 0.5, 0.5]
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    criterion = nn.BCEWithLogitsLoss()
    whole_iter_num = 0
    iter_num = math.ceil(len(train_loader.dataset) / args.batch_size)
    
    #consistency_criterion = nn.CrossEntropyLoss()
    
    for epoch in range(args.epochs):

        print('Starting epoch {}/{}.'.format(epoch + 1, args.epochs))
        print('epoch:{0}-------lr:{1}'.format(epoch + 1, args.lr))

        epoch_total_loss = 0
        epoch_loss = 0
        
        for i, data_batch in enumerate(train_loader):
            if (i + 1) > iter_num: break
            
            #get_local.clear()
        
            images, depths, label_224, label_14, label_28, label_56, label_112 = data_batch

            images, depths, label_224 = Variable(images.cuda(local_rank, non_blocking=True)), \
                                        Variable(depths.cuda(local_rank, non_blocking=True)), \
                                        Variable(label_224.cuda(local_rank, non_blocking=True))
                                    

            label_14, label_28, label_56, label_112 = Variable(label_14.cuda()), Variable(label_28.cuda()),\
                                                      Variable(label_56.cuda()), Variable(label_112.cuda())

           

            #outputs_saliency = net(images, depths)
            outputs_saliency,complementary_loss,consistency = net(images, depths)
            
            #mask_coarse_1_16, mask_coarse_1_8, mask_coarse_1_4 = mask_coarse

            mask_1_16, mask_1_8, mask_1_4, mask_1_1 = outputs_saliency
            # loss
            loss5 = criterion(mask_1_16, label_14)
            loss4 = criterion(mask_1_8, label_28)
            loss3 = criterion(mask_1_4, label_56)
            loss1 = criterion(mask_1_1, label_224)
            consistency_loss = criterion(consistency[0],label_14)
            consistency_loss += criterion(consistency[1],label_14)
            consistency_loss += criterion(consistency[2],label_28)
            consistency_loss += criterion(consistency[3],label_28)
            consistency_loss += criterion(consistency[4],label_56)
            consistency_loss += criterion(consistency[5],label_56)
            
            
            img_total_loss = loss_weights[0] * loss1 + loss_weights[2] * loss3 + loss_weights[2] * loss4 + loss_weights[2] * loss5 + complementary_loss[0]*0.2 + complementary_loss[1]*0.2 + complementary_loss[2]*0.2 + consistency_loss*0.15   
            #print(loss_weights[0] * loss1,loss_weights[2] * loss3,loss_weights[2] * loss4,loss_weights[2] * loss5,loss6*0.2)
            #img_total_loss = loss_weights[0] * loss1 + loss_weights[2] * loss3 + loss_weights[3] * loss4 + loss_weights[4] * loss5

            total_loss = img_total_loss

            epoch_total_loss += total_loss.cpu().data.item()
            epoch_loss += loss1.cpu().data.item()

            print(
                'whole_iter_num: {0} --- {1:.4f} --- total_loss: {2:.6f} --- saliency loss: {3:.6f}'.format(
                    (whole_iter_num + 1),
                    (i + 1) * args.batch_size / N_train, total_loss.item(), loss1.item()))

            optimizer.zero_grad()

            total_loss.backward()

            optimizer.step()
            whole_iter_num += 1

            if (local_rank == 0) and (whole_iter_num == args.train_steps) or epoch == args.epochs-1:
                torch.save(net.state_dict(),
                           args.save_model_dir + 'HCT.pth')

            if whole_iter_num == args.train_steps:
                return 0

            if whole_iter_num == args.stepvalue1 or whole_iter_num == args.stepvalue2 or whole_iter_num == args.stepvalue3:
                optimizer = adjust_learning_rate(optimizer, decay_rate=args.lr_decay_gamma)
                save_dir = './loss.txt'
                save_lr(save_dir, optimizer)
                print('have updated lr!!')
            

        print('Epoch finished ! Loss: {}'.format(epoch_total_loss / iter_num))
        save_lossdir = './loss.txt'
        save_loss(save_lossdir, whole_iter_num, epoch_total_loss / iter_num, epoch_loss/iter_num, epoch+1)
        #save1=0
        #save2=0
        if (epoch >= 39 and epoch%5==0):
            torch.save(net.state_dict(),
                           args.save_model_dir + 'HCT_{}.pth'.format(epoch+1))
            with open('ablation_result.txt', 'a') as f:
                f.write('epoch:{}\n'.format(epoch+1))
            eval_net(args, epoch+1)
            eval_result(args)

        
        if epoch == args.epochs-1:
                return 0
            

from torch.utils import data
import time
import transforms as trans
from torchvision import transforms
import numpy as np

def eval_net(args,epoch):

    cudnn.benchmark = True
    
    #get_local.activate()
    
    net = ImageDepthNet(args)
    net.cuda()
    net.eval()
    
    # load model (multi-gpu)
    
    model_path = args.save_model_dir + 'HCT_{}.pth'.format(epoch)
    '''
    state_dict = torch.load(model_path)
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    net.load_state_dict(new_state_dict)

    print('Model loaded from {}'.format(model_path))
    '''
    # load model
    net.load_state_dict(torch.load(model_path))
    #model_dict = net.state_dict()
    print('Model loaded from {}'.format(model_path))
    
    test_paths = args.test_paths.split('+')
    #test_paths = 'COME-E+COME-H'.split('+')
    for test_dir_img in test_paths:

        test_dataset = get_loader(test_dir_img, args.data_root, args.img_size, mode='test')

        test_loader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1)
        print('''
                   Starting testing:
                       dataset: {}
                       Testing size: {}
                   '''.format(test_dir_img.split('/')[0], len(test_loader.dataset)))

        time_list = []
        for i, data_batch in enumerate(test_loader):
            images, depths, image_w, image_h, image_path = data_batch
            images, depths = Variable(images.cuda()), Variable(depths.cuda())
            
            #get_local.clear()
            
            starts = time.time()
            outputs_saliency,_,_ = net(images, depths)
            ends = time.time()
            '''
            cache = get_local.cache
            attention_maps = cache['Attention.forward']
            print(len(attention_maps))
            '''
            time_use = ends - starts
            time_list.append(time_use)

            mask_1_16, mask_1_8, mask_1_4, mask_1_1 = outputs_saliency

            image_w, image_h = int(image_w[0]), int(image_h[0])

            output_s = F.sigmoid(mask_1_1)
            
            output_s = output_s.data.cpu().squeeze(0)

            transform = trans.Compose([
                transforms.ToPILImage(),
                trans.Scale((image_w, image_h))
            ])
            output_s = transform(output_s)

            dataset = test_dir_img.split('/')[0]
            filename = image_path[0].split('/')[-1].split('.')[0]

            # save saliency maps
            save_test_path = args.save_test_path_root + dataset + '/HCT/'
            if not os.path.exists(save_test_path):
                os.makedirs(save_test_path)
            output_s.save(os.path.join(save_test_path, filename + '.png'))

        print('dataset:{}, cost:{}'.format(test_dir_img.split('/')[0], np.mean(time_list)*1000))


import os.path as osp
from Evaluation.evaluator import Eval_thread
from Evaluation.dataloader import EvalDataset


def eval_result(args):

    pred_dir = args.save_test_path_root
    output_dir = args.save_dir
    gt_dir = args.data_root

    method_names = args.methods.split('+')

    threads = []
    test_paths = args.test_paths.split('+')
    for dataset_setname in test_paths:

        dataset_name = dataset_setname.split('/')[0]

        for method in method_names:

            pred_dir_all = osp.join(pred_dir, dataset_name, method)
            
            if dataset_name in ['NJUD', 'NLPR', 'DUTLF-Depth', 'ReDWeb-S']:
                gt_dir_all = osp.join(osp.join(gt_dir, dataset_setname), 'testset/GT')
            else:
                gt_dir_all = osp.join(osp.join(gt_dir, dataset_setname), 'GT')
            
            loader = EvalDataset(pred_dir_all, gt_dir_all)
            thread = Eval_thread(loader, method, dataset_setname, output_dir, cuda=True)
            threads.append(thread)
    for thread in threads:
        print(thread.run())

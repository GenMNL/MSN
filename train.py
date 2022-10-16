import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
import sys
import datetime
from data import *
from options import make_parser
from model import MSN
sys.path.append("./emd")
import emd_module as emd

# ----------------------------------------------------------------------------------------
# prepare subroutine for training one epoch
def train_one_epoch(model, dataloader, alpha, optim):
    model.train()
    # params for loss
    emd_loss = emd.emdModule()
    eps = 0.005
    iters = 50

    train_loss = 0.0
    count = 0

    for i, points in enumerate(tqdm(dataloader, desc="train")):
        comp = points[0]
        partial = points[1]

        # prediction
        partial = partial.permute(0, 2, 1) # [B, 3, N]
        coarse, fine, loss_expantion= model(partial)
        coarse = coarse.permute(0, 2, 1) # [B, N, 3]
        fine = fine.permute(0, 2, 1) # [B, N, 3]
        # get earth mover distance loss
        emd_coarse, _ = emd_loss(coarse, comp, eps, iters)
        emd_coarse = torch.sqrt(emd_coarse).mean()
        emd_fine, _ = emd_loss(fine, comp, eps, iters)
        emd_fine = torch.sqrt(emd_fine).mean()

        batch_loss = emd_coarse + alpha*emd_fine + 0.1*loss_expantion

        # backward + optim
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        train_loss += batch_loss
        count += 1

    train_loss = float(train_loss)/count
    return train_loss

def val_one_epoch(model, dataloader):
    model.eval()
    # params for loss
    emd_loss = emd.emdModule()
    eps = 0.004
    iters = 3000

    val_loss = 0.0
    count = 0

    with torch.no_grad():
        for i, points in enumerate(tqdm(dataloader, desc="validation")):
            comp = points[0]
            comp = comp.permute(0, 2, 1)
            partial = points[1]
            partial = partial.permute(0, 2, 1)
            # prediction
            coarse, fine, loss_expantion= model(partial)
            # get chamfer distance loss
            emd_fine, _ = emd_loss(fine, comp, eps, iters)
            emd_fine = torch.sqrt(emd_fine).mean(dim=1)

            val_loss += emd_fine
            count += 1

    val_loss = float(val_loss)/count
    return val_loss

# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    #  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # get options
    parser = make_parser()
    args = parser.parse_args()

    # make path of save params
    dt_now = datetime.datetime.now()
    save_date = str(dt_now.month) + str(dt_now.day) + "-" + str(dt_now.hour) + "-" + str(dt_now.minute)
    save_dir = os.path.join(args.save_dir, args.subset, str(dt_now.year), save_date)
    save_normal_path = os.path.join(save_dir, "normal_weight.tar")
    save_best_path = os.path.join(save_dir, "best_weight.tar")
    os.mkdir(save_dir)
    # make condition file
    with open(os.path.join(save_dir, "conditions.txt"), 'w') as f:
        f.write('')

    writter = SummaryWriter()
    #  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    #  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # make dataloader
    # data_dir = os.path.join(args.dataset_dir)
    train_dataset = MakeDataset(dataset_path=args.dataset_dir, subset=args.subset,
                                eval="train", num_partial_pattern=4, device=args.device)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                  shuffle=True, drop_last=True,
                                  collate_fn=OriginalCollate(args.num_partial, args.num_comp, args.device)) # DataLoader is iterable object.

    # validation data
    val_dataset = MakeDataset(dataset_path=args.dataset_dir, subset=args.subset,
                              eval="val", num_partial_pattern=4,device=args.device)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=2,
                                shuffle=True, drop_last=True,
                                collate_fn=OriginalCollate(args.num_partial, args.num_comp, args.device))

    # check of data in dataloader
    # for i, points in enumerate(tqdm(train_dataloader)):
        # print(f"complete points:{points[0].shape},  partial points:{points[1].shape}")
    #  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    #  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # prepare model and optimaizer
    model = MSN(args.emb_dim, args.num_output_points, args.num_surfaces, args.sampling_method).to(args.device)
    if args.optimizer == "Adam":
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=[0.9, 0.999])
    elif args.optimizer == "SGD":
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.6)

    # lr_schdual = torch.optim.lr_scheduler.StepLR(optim, step_size=int(args.epochs/4), gamma=0.7)
    #  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    #  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # main loop
    best_loss = np.inf
    for epoch in tqdm(range(1, args.epochs+1), desc="main loop"):

        # determin the ration of loss
        if epoch < 50:
            alpha = 0.01
        elif epoch < 100:
            alpha = 0.1
        elif epoch < 200:
            alpha = 0.5
        else:
            alpha = 1.0

        # get loss of one epoch
        train_loss = train_one_epoch(model, train_dataloader, alpha, optim)
        val_loss = val_one_epoch(model, val_dataloader)

        writter.add_scalar("train_loss", train_loss, epoch)
        writter.add_scalar("validation_loss", val_loss, epoch)

        # if val loss is better than best loss, update best loss to val loss
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                        'epoch':epoch,
                        'model_state_dict':model.state_dict(), 
                        'optimizer_state_dict':optim.state_dict(),
                        'loss':best_loss
                        }, save_best_path)
        # save normal weight 
        torch.save({
                    'epoch':epoch,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optim.state_dict(),
                    'loss':val_loss
                    }, save_normal_path)
        # lr_schdual.step()

    # close writter
    writter.close()

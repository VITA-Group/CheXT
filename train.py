"""
Author: Greg Holste
Last Modified: 12/9/21
Description: Script to train model and evaluate model on NIH ChestXRay14 dataset.
Usage: python train.py --data_dir <path_to_data> \
    --out_dir <desired_path_to_output> \
    --model_name resnet50 \
    --lr 1e-4 \
    --use_class_weights \
    --augment \
    --pretrained \
    --dropout_fc \
    --label_smoothing 0.1 \
    --n_TTA 10
"""

import sys

from dataset import ChestXRay14
# from models import pmga_resnet50, ShallowFFNN, ConcatFusion, MutualCrossAttention
from utils import * 

import argparse

def main(args):
    # Set model/output directory name
    MODEL_NAME = args.model_name
    MODEL_NAME += f'_{args.fusion_mode}-fusion' if args.fusion else ''
    MODEL_NAME += '_ch' if args.ch_att else ''
    MODEL_NAME += '_sp' if args.sp_att else ''
    MODEL_NAME += '+img' if args.img_ft else ''
    MODEL_NAME += '_aug' if args.augment else ''
    MODEL_NAME += '_pretr' if args.pretrained else ''
    MODEL_NAME += '_cw' if args.use_class_weights else ''
    MODEL_NAME += f'_lr-{args.lr}'
    MODEL_NAME += f'_bs-{args.batch_size}'
    MODEL_NAME += f'_ls-{args.label_smoothing}' if args.label_smoothing != 0.0 else ''
    MODEL_NAME += '_wd1e-5'
    MODEL_NAME += '_drp-fc' if args.dropout_fc else ''

    # Create output directory for model (and delete if already exists)
    model_dir = os.path.join(args.out_dir, MODEL_NAME)
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)

    # Set all seeds for reproducibility
    set_seed(args.seed)

    # Create datasets
    train_dataset = ChestXRay14(data_dir=args.data_dir, split="train", augment=args.augment)
    val_dataset   = ChestXRay14(data_dir=args.data_dir, split="val", augment=False)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, worker_init_fn=worker_init_fn)
    val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size*2, shuffle=False, num_workers=4, worker_init_fn=val_worker_init_fn)

    # Create csv documenting training history
    if args.unsupervised:
        history = pd.DataFrame(columns=['epoch', 'phase', 'loss'])
        history.to_csv(os.path.join(model_dir, 'history.csv'), index=False)
    else:
        history = pd.DataFrame(columns=['epoch', 'phase', 'loss', 'mean_auc'] + train_dataset.CLASSES)
        history.to_csv(os.path.join(model_dir, 'history.csv'), index=False)

    # Set device
    device = torch.device('cuda:0')

    # Instantiate model
    if args.fusion:
        # Late (concatenation-based) fusion
        if args.fusion_mode == 'concat':
            backbone = ResNet50FeatureExtractor(pretrained=args.pretrained, frozen=False)

            model = ConcatFusion(backbone, train_dataset.meta_features, dropout_fc=args.dropout_fc).to(device)
        # PMGA fusion
        elif args.fusion_mode == 'pmga':
            model = pmga_resnet50(meta_features=train_dataset.meta_features, ch_att=args.ch_att, sp_att=args.sp_att, img_ft=args.img_ft, pretrained=args.pretrained)

            n_features = model.fc.in_features

            if args.dropout_fc:
                model.fc = torch.nn.Sequential(torch.nn.Linear(n_features, 8), torch.nn.Dropout(p=0.25))
            else:
                model.fc = torch.nn.Linear(n_features, 8)
            
            model = model.to(device)
        elif args.fusion_mode == 'mutual-cross-attn':
            model = MutualCrossAttention(meta_features=train_dataset.meta_features, pretrained=args.pretrained).to(device)
        else:
            sys.exit('Invalid --fusion_mode requested.')
    # Metadata-only MLP model
    elif args.meta_only:
        model = ShallowFFNN(meta_features=train_dataset.meta_features).to(device)
    # Image-only ResNet50 model
    elif args.model_name == 'dino_vit_small_patch16':
        from models.vision_transformer import vit_small
        model = vit_small(patch_size=16, num_classes=0)

        state_dict = torch.load(args.weights)['student']
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

        msg = model.load_state_dict(state_dict, strict=False)
        print(
            "Pretrained weights found at {} and loaded with msg: {}".format(
                args.weights, msg
            )
        )

        model.head = torch.nn.Linear(384, 8)

        model = model.to(device)
    elif args.model_name == 'chext':
        from models.loss import FocalLoss, NTXentLoss
        from models.radiomic import extract_radiomic_features
        from models.radiomic_mlp import RaMLP
        from models.chext import CheXT
        from models.vision_transformer import vit_small

        model = vit_small(patch_size=16, num_classes=0)

        ## INSERT CODE TO LOAD WEIGHTS ## 
        model_weight_path = f'/home/gh23476/012222_dino_imgnet-to-nih_vit-s/checkpoint0250.pth'
        state_dict = torch.load(model_weight_path)['student']
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

        msg = model.load_state_dict(state_dict, strict=False)
        print(
            "Pretrained weights found at {} and loaded with msg: {}".format(
                args.weights, msg
            )
        )

        ramlp = RaMLP()
        model = CheXT(model, ramlp, 2, 8, [384, 107], unsupervised=args.unsupervised)

        model = model.to(device)

    else:
        model = torchvision.models.resnet50(pretrained=args.pretrained)

        n_features = model.fc.in_features

        if args.dropout_fc:
            model.fc = torch.nn.Sequential(torch.nn.Linear(n_features, 8), torch.nn.Dropout(p=0.25))
        else:
            model.fc = torch.nn.Linear(n_features, 8)
        
        model = model.to(device)
    print(model)

    print('Positive class weights:')
    for i in range(8):
        print(f'{train_dataset.CLASSES[i]}: {train_dataset.pos_weight[i]}')

    if args.unsupervised:
        epoch = 1
        loss_fxn = NTXentLoss(device, 0.05).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        while epoch <= args.max_epochs:
            history = unsupervised_train(model=model, device=device, loss_fxn=loss_fxn, ls=args.label_smoothing, optimizer=optimizer, data_loader=train_loader, history=history, epoch=epoch, model_dir=model_dir)
            epoch += 1
        return

    # Set class weights
    if args.use_class_weights:
        loss_fxn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(train_dataset.pos_weight).to(device))
    else:
        loss_fxn = torch.nn.BCEWithLogitsLoss()

    # Set optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  #, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Train with early stopping
    epoch = 1
    early_stopping_dict = {'best_loss': 1e8, 'epochs_no_improve': 0}
    best_model_wts = None
    while epoch <= args.max_epochs and early_stopping_dict['epochs_no_improve'] <= args.patience:
        history = train(model=model, device=device, loss_fxn=loss_fxn, ls=args.label_smoothing, optimizer=optimizer, data_loader=train_loader, history=history, epoch=epoch, model_dir=model_dir, classes=train_dataset.CLASSES, fusion=args.fusion, meta_only=args.meta_only, chext=args.model_name == 'chext')
        history, early_stopping_dict, best_model_wts = validate(model=model, device=device, loss_fxn=loss_fxn, ls=args.label_smoothing, optimizer=optimizer, data_loader=val_loader, history=history, epoch=epoch, model_dir=model_dir, early_stopping_dict=early_stopping_dict, best_model_wts=best_model_wts, classes=val_dataset.CLASSES, fusion=args.fusion, meta_only=args.meta_only)

        epoch += 1
        # scheduler.step()
    
    # Evaluate on test set
    evaluate(model=model, data_dir=args.data_dir, device=device, loss_fxn=loss_fxn, ls=args.label_smoothing, batch_size=args.batch_size, history=history, model_dir=model_dir, weights=best_model_wts, n_TTA=args.n_TTA, fusion=args.fusion, meta_only=args.meta_only)

if __name__ == '__main__':
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/home/yh9442/yan/CXR14/', type=str)
    parser.add_argument('--out_dir', default='results/', type=str,
                        help="path to directory where results and model weights will be saved")        
    parser.add_argument('--model_name', default='resnet50', type=str, help="CNN backbone to use")
    parser.add_argument('--weights', default='', type=str)
    parser.add_argument('--fusion', action='store_true', default=False)
    parser.add_argument('--meta_only', action='store_true', default=False)
    parser.add_argument('--fusion_mode', default='concat', help="fusion method (one of ['concat', 'pmga', 'mutual-cross-attn'])")
    parser.add_argument('--ch_att', action="store_true", default=False, help="whether to use channel attention for PMGA fusion")
    parser.add_argument('--sp_att', action="store_true", default=False, help="whether to use spatial attention for PMGA fusion")
    parser.add_argument('--img_ft', action="store_true", default=False, help="whether to use image features for PMGA fusion")
    parser.add_argument('--max_epochs', default=50, type=int, help="maximum number of epochs to train")
    parser.add_argument('--batch_size', default=128, type=int, help="batch size for training, validation, and testing (will be lowered if TTA used)")
    parser.add_argument('--patience', default=10, type=int, help="early stopping 'patience' during training")
    parser.add_argument('--use_class_weights', default=False, action="store_true", help="whether or not to use class weights applied to loss during training")
    parser.add_argument('--augment', default=False, action="store_true", help="whether or not to use augmentation during training")
    parser.add_argument('--pretrained', default=False, action="store_true", help="whether or not to use ImageNet weight initialization for ResNet backbone")
    parser.add_argument('--label_smoothing', type=float, default=0.0, help="ratio of label smoothing to use during training")
    parser.add_argument('--dropout_fc', action="store_true", default=False)
    parser.add_argument('--unsupervised', action="store_true", default=False)
    parser.add_argument('--n_TTA', default=0, type=int, help="number of augmented copies to use during test-time augmentation (TTA), default 0")
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--seed', default=0, type=int, help="set random seed")
    parser.add_argument('--n', default=-1, type=int)

    args = parser.parse_args()

    main(args)
##########################
# This is a copy of the original tpt_classification.py file
# that was used to classify the TPT data.
# This file uses our functions from COOP to train the model.
##########################
import argparse
from TPT.utils.tools import set_random_seed
import loaders
import CLIP.clip as clip
_datasets = {
    "cifar100" : "./data/cifar100",
    "imagenet_A" : "./data/imagenet-a",
    "imagenet_v2" : "./data/imagenetv2-matched-frequency-format-val" 
}

_loaders = {
    "cifar100" : loaders.load_cifar100,
    "imagenet_A" : loaders.load_imagenet_A,
    "imagenet_v2" : loaders.load_imagenet_v2
}

def load_coop():
    pass

def main(parser):
    args = parser.parse_args()
    assert args.device in ['cpu', 'cuda', 'mps'], 'Invalid device'
    assert args.class_token_position in ['end', 'middle', 'front'], 'Invalid class token position'

    set_random_seed(args.seed)
    print("Using [", args.device.upper() , "]")


    # Get the model
    if args.coop:
        model, preprocess = load_coop()
        # Loading CoOp
    else:
        # Load backbone and train it
        model, preprocess = clip.load(args.backbone, device=args.device)
        pass

    # Get dataloaders
    loader, id2class = _loaders[args.dataset](
        _datasets[args.dataset], args.batch_size, preprocess=preprocess, 
    )

    # Define the optimizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model on the TPT dataset')
    parser.add_argument('--dataset', type=str, default='tpt', help='Name of the dataset')
    parser.add_argument('--backbone', type=str, default='RN50', help='Name of the backbone')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--device', type=str, default='mps', help='Device to use')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--run_name', type=str, default='exp1', help='Name of the run')
    parser.add_argument('--n_ctx', type=int, default=4, help='Number of context tokens')
    parser.add_argument('--ctx_init', type=str, default='', help='Initial context')
    parser.add_argument('--class_token_position', type=str, default='end', help='Position of the class token')
    parser.add_argument('--csc', type=bool, default=False, help='Use CSC')

    main(parser)
import os
import torch
import open_clip
import tqdm
from torchvision.transforms import transforms
from COOP.dataloader import get_data, Augmixer
from utils import caption_report

path = 'ice/captions/{}_captions/captions_{}{}'

captioners = ["coca", "blip", "llava"]
datasets = ['ImageNetA', 'ImageNetV2']

options = {
    "coca" : ['.a', '.a_photo_of', '.a_photo_containing'],
    "blip" : ["", ".concise", ".specific"],
    "llava" : ["", ".concise", ".specific"]
}
    
captions = {
    "coca" : {
        "ImageNetA" : {
            ".a" : {},
            ".a_photo_of" : {},
            ".a_photo_containing" : {}
        },
        "ImageNetV2" : {
            ".a" : {},
            ".a_photo_of" : {},
            ".a_photo_containing" : {}
        }
    },
    "blip" : {
        "ImageNetA" : {
            "" : {},
            ".concise" : {},
            ".specific" : {}
        },
        "ImageNetV2" : {
            "" : {},
            ".concise" : {},
            ".specific" : {}
        }
    },
    "llava" : {
        "ImageNetA" : {
            "" : {},
            ".concise" : {},
            ".specific" : {}
        },
        "ImageNetV2" : {
            "" : {},
            ".concise" : {},
            ".specific" : {}
        }
    }
}

def _tokenize(x, tokenizer):
    """
    Tokenizes the input text using the provided tokenizer.

    Args:
        x (str): The input text to tokenize.
        tokenizer: The tokenizer object.

    Returns:
        torch.Tensor: The tokenized input text.
    """
    x_tokenized = tokenizer(x).squeeze()
    start_token = 49406
    end_token = 49407
    assert x_tokenized[0] == start_token
    return x_tokenized[:list(x_tokenized).index(end_token)]

def _generate_macro(caption_model, im, prompt):
    """
    Generates captions for the given images using the caption model.

    Args:
        caption_model: The caption model.
        im: The input images.
        prompt: The prompt for caption generation.

    Returns:
        torch.Tensor: The generated captions.
    """
    text=torch.ones((im.shape[0], 1), device=device, dtype=torch.long)*prompt
    generated = caption_model.generate(
                im, 
                text=text,)
                # generation_type='top_p')
    return generated

def get_test_transform():
    """
    Returns the transformation pipeline for test images.

    Returns:
        torchvision.transforms.Compose: The transformation pipeline.
    """
    return transforms.Compose(
        [transforms.Resize(size=224, max_size=None, antialias=None),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))]
    )

def generate_captions(self, images, prompt, tokenizer, device)->list:
    """
    Generates captions for the given images.

    Args:
        images: The input images.
        prompt (str): The prompt for caption generation.
        tokenizer: The tokenizer object.
        device (str): The device to run the model on.

    Returns:
        list: The generated captions.
    """
    caption_model.eval()
    
    outputs = []
    prompt_extended = _tokenize(prompt, tokenizer).to(device)
        
    generated = _generate_macro(
        caption_model, 
        images, 
        prompt_extended)
    
    assert len(generated) == len(images)
    for i in range(len(generated)):
        outputs.append(open_clip.decode(generated[i]).split("<end_of_text>")[0].replace("<start_of_text>", ""))
    return outputs

def process_line_imagenetA(x):
    """
    Processes a line of data for ImageNetA dataset.

    Args:
        x (str): The input line.

    Returns:
        tuple: The processed path and caption.
    """
    raw_path, caption = x.split('<sep>')
    caption = '<sep>' + caption.strip(" \n")

    raw_path= raw_path.rsplit('_')[0] # Removes rightmost text after _
    raw_path = raw_path.rsplit('_')[0].strip()  # Repeats
    parts = raw_path.split('/')
    parts[1] = 'data'
    ret_path = '/'.join(parts)

    return ret_path, caption

def process_line_imagenetV2(x):
    """
    Processes a line of data for ImageNetV2 dataset.

    Args:
        x (str): The input line.

    Returns:
        tuple: The processed path and caption.
    """
    raw_path, caption = x.split('<sep>')
    caption = '<sep>' + caption.strip(" \n")

    parts = raw_path.split('/')
    parts[1] = 'data'
    ret_path = '/'.join(parts)
    
    return ret_path, caption

def get_captions()->dict:
    """
    Retrieves the captions for all datasets and options.

    Returns:
        dict: The captions dictionary.
    """
    for captioner in captioners:
        for dataset in datasets:
            for option in options[captioner]:
                file = path.format(captioner.upper(), dataset, option)
                if not os.path.exists(file):
                    print("File path does not exist\n{}".format(file))
                else:
                    print("Path checks out")

                with open(file, 'r') as f:
                    for line in f.readlines():
                        if dataset == 'ImageNetA':
                            ret_path, caption = process_line_imagenetA(line)
                        else:
                            ret_path, caption = process_line_imagenetV2(line)
                        # TODO check which part of the path we want to keep
                        captions[captioner][dataset][option][ret_path] = caption
    return captions


if __name__ == '__main__':
    # This script adjusts pre-generated captions to match the path of the images
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    cache_dir = '.dl-cache'
    prompt = "a "

    tokenizer = open_clip.get_tokenizer("coca_ViT-L-14")
    caption_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name="coca_ViT-L-14",
        pretrained="laion2B-s13B-b90k",
        cache_dir = cache_dir
    )
    caption_model.to(device)

    data_transform = Augmixer(preprocess, 32, augmix=True, severity=1)

    _, _, data, classes, id2class = get_data('imagenet_a', 1, data_transform, True, 0.0, 0.0)

    answers = []
    loop = tqdm.tqdm(enumerate(data), total=len(data))
    for idx, (images, label, path) in loop:
        images = images.to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = generate_captions(
                images, 
                caption_model, 
                prompt,
                tokenizer,
                device
            )

        caption_report(images, label, outputs, id2class, idx)
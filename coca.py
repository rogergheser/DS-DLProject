import os
import torch
import open_clip
import tqdm
from torchvision.transforms import transforms
from COOP.dataloader import get_data, Augmixer

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
    x_tokenized = tokenizer(x).squeeze()
    start_token = 49406
    end_token = 49407
    assert x_tokenized[0] == start_token
    return x_tokenized[:list(x_tokenized).index(end_token)]

def _generate_macro(caption_model, im, prompt):
    text=torch.ones((im.shape[0], 1), device=device, dtype=torch.long)*prompt
    generated = caption_model.generate(
                im, 
                text=text,
                generation_type='top_p')
    return generated

def get_test_transform():
    return transforms.Compose(
        [transforms.Resize(size=224, max_size=None, antialias=None),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))]
    )

def generate_captions(images, caption_model, prompt, tokenizer, device):
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
    raw_path, caption = x.split('<sep>')
    caption = '<sep>' + caption.strip(" \n")

    raw_path= raw_path.rsplit('_')[0] # Removes rightmost text after _
    raw_path = raw_path.rsplit('_')[0].strip()  # Repeats
    parts = raw_path.split('/')
    parts[1] = 'data'
    ret_path = '/'.join(parts)

    return ret_path, caption

def process_line_imagenetV2(x):
    raw_path, caption = x.split('<sep>')
    caption = '<sep>' + caption.strip(" \n")

    parts = raw_path.split('/')
    parts[1] = 'data'
    ret_path = '/'.join(parts)
    
    return ret_path, caption

def get_captions()->dict:
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

def caption_report(images, label, outputs, id2class, idx):
    import matplotlib.pyplot as plt

    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]

    mean = torch.tensor(clip_mean).reshape(1, 3, 1, 1)
    std = torch.tensor(clip_std).reshape(1, 3, 1, 1)

    # Denormalize the batch of images
    unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    denormalized_images = unnormalize(images)

    # Visualise the input using matplotlib
    images = [image.numpy().transpose(1, 2, 0) for image in denormalized_images.cpu()] # Convert to numpy and transpose to (H, W, C)
    label = [lab.item() for lab in label.cpu()] if label.shape[0] > 1 else label.item()

    plt.figure(figsize=(16, 16), dpi=300)
    plt.title(f"Captions generated from the {idx}th batch") if isinstance(label, list) else plt.title(f"Caption for {id2class[label]} class")
    plt.axis('off')

    for i, image in enumerate(images[:9]):
        plt.subplot(3,3, i+1)
        plt.title(id2class[label[i]]) if isinstance(label, list) else id2class[label]
        plt.xlabel(outputs[i])
        plt.imshow(image)

    plt.savefig(f"caption_reports/batch_{idx}.png")
    plt.close()


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
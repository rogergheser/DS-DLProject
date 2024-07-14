import os

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



if __name__ == '__main__':
    # This script adjusts pre-generated captions to match the path of the images
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

    print(captions.keys())
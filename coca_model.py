import torch
import open_clip
from torchvision.transforms import transforms
from typing import List

class Captioner():
    def __init__(self, model_name, version, device):
        self.caption_model, _ , self.transform = open_clip.create_model_and_transforms(
            model_name=model_name, 
            pretrained=version, 
            cache_dir='./.dl-cache'
            )
        self.caption_model.to(device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.device = device
    
    def _tokenize(self, x: str) -> torch.Tensor:
        """
        Tokenizes the input text using the tokenizer.

        Args:
            x (str): The input text.

        Returns:
            torch.Tensor: The tokenized text.
        """
        x_tokenized = self.tokenizer(x).squeeze()
        start_token = 49406
        end_token = 49407
        assert x_tokenized[0] == start_token
        return x_tokenized[:list(x_tokenized).index(end_token)]
    
    def _generate_macro(self, im: torch.Tensor, prompt: int) -> torch.Tensor:
        """
        Generates captions for the input images.

        Args:
            im (torch.Tensor): The input images.
            prompt (int): The prompt for caption generation.

        Returns:
            torch.Tensor: The generated captions.
        """
        text=torch.ones((im.shape[0], 1), device=self.device, dtype=torch.long)*prompt

        generated = self.caption_model.generate(
                    im, 
                    text=text,
                    generation_type='top_p')
        return generated
    
    def get_test_transform(self) -> transforms.Compose:
        """
        Returns the test transformation for images.

        Returns:
            transforms.Compose: The test transformation.
        """
        return transforms.Compose(
            [transforms.Resize(size=224, max_size=None, antialias=None),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))]
        )
    
    def generate_captions(self, images: torch.Tensor, prompt: int) -> List[str]:
        """
        Generates captions for the input images.

        Args:
            images (torch.Tensor): The input images.
            prompt (int): The prompt for caption generation.

        Returns:
            List[str]: The generated captions.
        """
        self.caption_model.eval()
        
        outputs = []
        prompt_extended = self._tokenize(prompt).to(self.device)
            
        generated = self._generate_macro( 
            images, 
            prompt_extended)
        
        assert len(generated) == len(images)
        for i in range(len(generated)):
            outputs.append(open_clip.decode(generated[i]).split("<end_of_text>")[0].replace("<start_of_text>", ""))
        return outputs
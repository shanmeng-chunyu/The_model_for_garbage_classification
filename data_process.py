from torchvision import transforms
import torch



class ImagePreprocessor:
      def __init__(self, mean, std):
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

      def __call__(self, img):
            return self.transform(img)

class ImageAugmentor:
      def __init__(self, mean, std):
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop((224,224), scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(degrees=15),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
      def __call__(self, img):
          return self.transform(img)
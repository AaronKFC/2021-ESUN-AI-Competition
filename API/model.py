import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import pickle

class Model:
    def __init__(self):
        self.model = None
        self.wordset = None
        self.class_names = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    def initialize(self, model_name, model_path):
        with open('./data/wordset.txt') as f:
            self.wordset = f.read().split('\n')

        with open('./idx2class.pickle', 'rb') as f:
            self.class_names = pickle.load(f)

        self.model = EfficientNet.from_pretrained(model_name)
        num_ftrs = self.model._fc.in_features
        self.model._fc  = torch.nn.Linear(num_ftrs, len(self.class_names))
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(model_path))

    def predict(self, image):
        self.model.eval()
        input = self.transform(image).unsqueeze(0)        
        input = input.to(self.device)
        with torch.no_grad():
            output = self.model(input)
            _, pred = torch.max(output, 1)
            word = self.class_names[pred[0]]
            if word in self.wordset:
                return word
            else:
                return 'isnull'
        
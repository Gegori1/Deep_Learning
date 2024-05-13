import torch
from torchvision.transforms import v2 as T
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt

class Pipeline:
    def __init__(self, model_fasterrcnn, model_facenet, image_path, device):
        self.model_fasterrcnn = model_fasterrcnn
        self.model_facenet = model_facenet
        self.image_path = image_path
        self.device = device
        
    def test_transform_fasterrcnn(self):
        return T.Compose([
            T.ToDtype(torch.float, scale=True),
            T.ToPureTensor()
        ])

        
    def test_transform_facenet(self, coordinates):
        coords = {
            'top': coordinates[0],
            'left': coordinates[1],
            'height': coordinates[3] - coordinates[1],
            'width': coordinates[3] - coordinates[1],
            'size': [160, 160]
        }

        return (
            T.Compose([
              T.ToDtype(torch.float, scale=True),
              T.ToPureTensor()
          ]), 
          coords
        )
        
        
        
    def run(self):
        image = read_image(self.image_path)
        # FasterRCNN
        transform_rcnn = self.test_transform_fasterrcnn()
        self.model_fasterrcnn.eval()
        with torch.no_grad():
            # transform image
            x = transform_rcnn(image).to(self.device)
            x = x.unsqueeze(0)
            predictions = self.model_fasterrcnn(x)
            pred = predictions[0]
            
        image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
        self.image = image[:3, ...]
        
        best_scores = torch.argmax(pred['scores'])
        best_box = pred['boxes'][best_scores]
        self.best_box = best_box.long()
        
        self.best_label = pred['labels'][best_scores]
        self.best_score = pred['scores'][best_scores]
        
        transform_facenet, coords = self.test_transform_facenet(self.best_box)
        
        # Facenet
        self.model_facenet.eval()
        with torch.no_grad():
            try:
                # transform image
                x = T.functional.resized_crop(inpt=image, **coords)
                x = transform_facenet(self.image)
            except IndexError:
                raise IndexError('IndexError: list index out of range')
            x = x.unsqueeze(0).to(self.device)
            embeddings = self.model_facenet(x)
            
        self.print_results()
            
        return embeddings

    def print_results(self):
        # pred_labels = [f"man: {score:.3f}" if label == 2 else f"woman: {score:.3f}" for label, score in zip(self.best_label, self.best_score)]
        pred_labels = f"man: {self.best_score:.3f}" if self.best_label == 2 else f"woman: {self.best_score:.3f}"
        output_image = draw_bounding_boxes(self.image, self.best_box.unsqueeze(0), [pred_labels], colors="red", width=3, font_size=3)
        
        print(f'Predictions: {pred_labels}')
        plt.figure(figsize=(5, 5))
        plt.imshow(output_image.permute(1, 2, 0))
        

    def save_embeddings(self):
        print('Embeddings saved')
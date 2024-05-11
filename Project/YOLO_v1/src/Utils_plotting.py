import torch
from matplotlib.patches import Rectangle

import matplotlib.pyplot as plt
from cv2 import resize

class BoxPlotter:
    def __init__(self, model, data, threshold=0.5, S=4, B=2, C=2, resize_size=..., device=...):
        self.model = model
        self.data = data
        self.threshold = threshold
        self.grid_size = S
        self.boxes = B
        self.classes = C
        self.resize=resize_size
        self.device = device

    def inference_one_image(self, image_idx):
        self.model.eval()
        image = self.data[image_idx]["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_box = self.model(image)
        pred_box = pred_box.reshape(self.grid_size, self.grid_size, 5 * self.boxes + self.classes)

        pred_box_1 = pred_box[..., :5]
        pred_box_2 = pred_box[..., 5:10]
        
        classes = pred_box[..., 10:12]
        
        idx = torch.where(pred_box_1[..., 4] > self.threshold)
        box_1 = pred_box_1[idx]
        idx = torch.where(pred_box_2[..., 4] > self.threshold)
        box_2 = pred_box_2[idx]
        
        classes_box = classes[idx]
        
        print(f"classes: {classes_box}")
        print(f"prob_1: {box_1}, prob_2: {box_2}")
        
        return box_1, box_2

    def draw_boxes(self, boxes, thickness=1, color="red", text=""):
        rs1, rs2 = self.resize
        boxes = boxes.cpu().numpy()
        boxes = boxes.reshape(-1, 5)
        boxes = boxes[:, :-1]

        box_lines = []
        for box in boxes:
            xc, yc, w, h = box
            x, w = (xc - w/2) * rs1, w * rs1
            y, h = (yc - h/2) * rs2, h * rs2
            dn = Rectangle(
                xy=(x, y),
                width=w,
                height=h,
                fill=False,
                color=color,
                linewidth=thickness
            )
            box_lines.append(dn)
            
        box_n = [{
            "box": box_lines,
            "coord": (xc * rs1, yc * rs2),
            "text": text,
            "color": color
        }]

        # return box_lines
        return box_n

    def plot_boxes(self, image, boxes):
        fig, ax = plt.subplots(1)
        img = resize(image, self.resize)
        ax.imshow(img)
        
        for box in boxes:
            for box_line in box["box"]:
                ax.add_patch(box_line)
            ax.text(*box["coord"], box["text"], color=box["color"])

        fig.show()
        
    def plot(self, image_idx):
        image = self.data[image_idx]["image"].cpu().numpy().transpose(1, 2, 0)
        box_1, box_2 = self.inference_one_image(image_idx)
        
        if len(box_1) == 0 and len(box_2) == 0:
            print("No boxes detected")
            image = resize(image, self.resize)
            plt.imshow(image)
            plt.show()
            return
        if len(box_1) > 0:
            boxes_1 = self.draw_boxes(box_1, color="blue", text="Box 1")
        else:
            boxes_1 = []
        if len(box_2) > 0:
            boxes_2 = self.draw_boxes(box_2, color="green", text="Box 2")
        else:
            boxes_2 = []
        
        boxes = boxes_1 + boxes_2
        
        self.plot_boxes(image, boxes)

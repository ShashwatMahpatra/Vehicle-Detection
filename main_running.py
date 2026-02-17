from tkinter import *
from tkinter import ttk, filedialog
from PIL import ImageTk, Image, ImageFilter
import cv2
import numpy as np
import time
from scipy import spatial
from input_retrieval import *
from polygon import *
from create_mask import *
import torch
from torchvision import transforms, models
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
from ultralytics import YOLO

# ========== Device Setup ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

def setup_uda_components():
    global uda_transform, resnet18
    
    # Define the image transformation pipeline
    uda_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load pre-trained ResNet18
    resnet18 = models.resnet18(pretrained=True).eval().to(device)

# Initialize UDA components early
setup_uda_components()

# ========== GAN Model Definitions ==========
class GeneratorUNet(nn.Module):
    def __init__(self):
        super().__init__()

        def down(in_c, out_c, bn=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
            if bn: layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)

        def up(in_c, out_c, drop=0.0):
            layers = [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
                      nn.BatchNorm2d(out_c),
                      nn.ReLU()]
            if drop: layers.append(nn.Dropout(drop))
            return nn.Sequential(*layers)

        self.down1 = down(3, 64, False)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)
        self.down5 = down(512, 512)
        self.down6 = down(512, 512)
        self.down7 = down(512, 512)
        self.down8 = down(512, 512, False)

        self.up1 = up(512, 512, 0.5)
        self.up2 = up(1024, 512, 0.5)
        self.up3 = up(1024, 512, 0.5)
        self.up4 = up(1024, 512)
        self.up5 = up(1024, 256)
        self.up6 = up(512, 128)
        self.up7 = up(256, 64)

        self.final = nn.ConvTranspose2d(128, 3, 4, 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        return self.tanh(self.final(torch.cat([u7, d1], 1)))

# ========== Fog Processing Functions ==========
def load_fog_removal_model(model_path=r"C:\Users\Shashwat\OneDrive\Desktop\new_gan_model\generator1.pth"):
    model = GeneratorUNet().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("✅ Fog removal model loaded successfully")
    else:
        print(f"⚠️ Warning: No fog removal model found at {model_path}")
        return None
    return model

def is_foggy(frame, threshold=0.7):
    """Improved fog detection using multiple features"""
    if frame is None:
        return False
        
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Dark channel calculation
    dark_channel = np.min(frame, axis=2)
    dark_channel = cv2.erode(dark_channel, np.ones((15,15), np.uint8))
    dark_mean = dark_channel.mean() / 255.0
    
    # Contrast check
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    contrast = gray.std() / 255.0
    
    # Saturation check
    saturation = hsv[:,:,1].mean() / 255.0
    
    # Combined fog likelihood score
    fog_score = (dark_mean + (1 - contrast) + (1 - saturation)) / 3
    
    return fog_score > threshold

def process_frame_for_fog(frame, model):
    """Process frame through fog removal GAN if fog is detected"""
    if frame is None or model is None:
        return frame
    
    try:
        # Convert to 3-channel if needed
        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
        if not is_foggy(frame):
            return frame
        
        # Preprocess for GAN
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
        
        # Post-process output
        output = output.squeeze().permute(1, 2, 0).cpu().numpy()
        output = (output * 0.5 + 0.5) * 255  # [-1,1] to [0,255]
        output = output.astype('uint8')
        output = cv2.resize(output, (frame.shape[1], frame.shape[0]))
        
        # Convert back to BGR
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output
        
    except Exception as e:
        print(f"⚠️ Fog removal failed: {str(e)}")
        return frame

def uda_preprocess_frame(frame):
    """Preprocess frame using UDA technique"""
    global uda_transform, resnet18
    
    try:
        # Convert frame to PIL Image and apply sharpening
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
        img_sharp = img_pil.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        
        # Apply transformations and move to device
        img_tensor = uda_transform(img_sharp).unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            _ = resnet18(img_tensor)
            
        return frame
    except Exception as e:
        print(f"⚠️ UDA preprocessing failed: {str(e)}")
        return frame

# ========== Vehicle Counting Variables ==========
class PolygonDrawer:
    def __init__(self, window_name):
        self.window_name = window_name
        self.done = False
        self.current = (0, 0)
        self.points = []
        
    def on_mouse(self, event, x, y, buttons, user_param):
        if self.done:
            return
            
        if event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.done = True
    
    def run(self, frame):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        
        while not self.done:
            canvas = frame.copy()
            if len(self.points) > 0:
                cv2.polylines(canvas, [np.array(self.points)], False, (0, 0, 255), 2)
                cv2.circle(canvas, self.current, 2, (0, 0, 255), -1)
            cv2.imshow(self.window_name, canvas)
            if cv2.waitKey(50) == 27:  # ESC to finish
                self.done = True
        
        cv2.destroyWindow(self.window_name)
        return self.points

class VehicleCounter:
    def __init__(self):
        self.start_counting = False
        self.vehicle_count = 0
        self.count = {
            "bicycle": 0,
            "motorbike": 0,
            "auto": 0,
            "car": 0,
            "truck": 0,
            "rickshaw": 0,
            "minitruck": 0,
            "van": 0,
            "bus": 0
        }
        self.FRAMES_BEFORE_CURRENT = 15
        self.previous_frame_detections = [{(0, 0): 0} for _ in range(self.FRAMES_BEFORE_CURRENT)]
        self.yolo_model = YOLO("yolov8x.pt")
        
        # Custom class mapping with adjusted thresholds
        self.class_mapping = {
            # Two-wheelers
            1: "bicycle",    # bicycle
            3: "motorbike",  # motorcycle (changed from default car to motorbike)
            
            # Three-wheelers
            80: "auto",      # autorickshaw (custom mapping)
            81: "rickshaw",  # cycle rickshaw (custom mapping)
            
            # Four-wheelers
            2: "car",        # car
            7: "truck",      # truck
            5: "bus",        # bus
            9: "minitruck",  # minitruck (custom mapping)
            79: "van"        # van (custom mapping)
        }
        
        # Class-specific confidence thresholds
        self.class_thresholds = {
            "bicycle": 0.4,
            "motorbike": 0.5,
            "auto": 0.6,      # Higher threshold for autorickshaws
            "car": 0.5,
            "truck": 0.6,
            "rickshaw": 0.5,
            "minitruck": 0.6,
            "van": 0.6,
            "bus": 0.7        # Highest threshold for buses
        }
        
        self.setup_gui()

    def setup_gui(self):
        self.root = Tk()
        self.root.grid()
        self.root["bg"] = "gainsboro"
        self.root.title('Count Vehicle')
        self.root.minsize(400, 500)

        # Step 1: Video Selection
        self.selection_label = Label(self.root, text="Select a video file to begin", pady=20)
        self.selection_label.grid(row=0, column=0, columnspan=3)
        
        self.select_button = Button(
            self.root, text="Select Video", bg='NavajoWhite2', width=30, 
            pady=15, command=self.select_video
        )
        self.select_button.grid(row=1, column=1, sticky="nsew")
        self.select_button.config(borderwidth=5, relief=RAISED)
        
        # Will be created after video selection
        self.start_button = None
        
        # Count display area (initially empty)
        self.count_frame = Frame(self.root)
        self.count_frame.grid(row=2, column=0, columnspan=3, sticky="nsew")
        
        # Video display area
        self.video_frame = Frame(self.root)
        self.video_frame.grid(row=3, column=0, columnspan=3, sticky="nsew")
        self.video_label = Label(self.video_frame)
        self.video_label.pack()
        
        # Configure grid weights
        for i in range(4):
            self.root.rowconfigure(i, weight=1)
        for i in range(3):
            self.root.columnconfigure(i, weight=1)
            
        self.root.mainloop()

    def select_video(self):
        filename = filedialog.askopenfilename(
            initialdir="", title="Select video:",
            filetypes=(("mp4 files", "*.mp4"), ("avi files", "*.avi")))
        
        if filename and filename.endswith(('.mp4', '.avi')):
            self.video_path = filename
            self.selection_label.config(text=f"Selected: {filename.split('/')[-1]}")
            
            # Create Start button
            self.start_button = Button(
                self.root, text="Start Counting", bg='cyan3', width=30, 
                pady=15, command=self.prepare_counting
            )
            self.start_button.grid(row=1, column=1, sticky="nsew")
            self.start_button.config(borderwidth=5, relief=RAISED)
        else:
            self.selection_label.config(text="Please select a valid video file!")

    def prepare_counting(self):
        # Open video file
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.selection_label.config(text="Error opening video file!")
            return
            
        # Get first frame for ROI selection
        ret, frame = self.cap.read()
        if not ret:
            self.selection_label.config(text="Error reading video frame!")
            return
            
        # Step 2: Draw ROI polygon
        self.selection_label.config(text="Draw ROI polygon (ESC to finish)")
        pd = PolygonDrawer("Draw ROI - Press ESC when done")
        polygon_points = pd.run(frame)
        
        if len(polygon_points) < 3:
            self.selection_label.config(text="Please draw a valid polygon!")
            return
            
        # Create mask from polygon
        self.polygon_coords = polygon_points
        self.mask = np.zeros(frame.shape[:2], dtype="uint8")
        cv2.fillPoly(self.mask, [np.array(self.polygon_coords)], 255)
        
        # Step 3: Start counting
        self.selection_label.config(text="Counting vehicles...")
        self.setup_count_display()
        self.start_counting = True
        self.process_video()

    def setup_count_display(self):
        # Clear previous widgets
        for widget in self.count_frame.winfo_children():
            widget.destroy()
            
        # Create count labels
        self.count_labels = {}
        vehicles = [
            ("Bicycles", 0, 0), ("Motorbikes", 0, 1), ("Autorickshaws", 0, 2),
            ("Cars", 1, 0), ("Trucks", 1, 1), ("Rickshaws", 1, 2),
            ("Vans", 2, 0), ("Minitrucks", 2, 1), ("Buses", 2, 2)
        ]
        
        for text, row, col in vehicles:
            self.count_labels[text.lower()] = Label(
                self.count_frame, text=f"{text}: 0", borderwidth=3, relief=RAISED
            )
            self.count_labels[text.lower()].grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            self.count_labels[text.lower()]['bg'] = "CadetBlue1" if col % 2 == 0 else "SkyBlue1"
        
        # Total vehicles label
        self.total_label = Label(
            self.count_frame, text="Total vehicles: 0", fg='red', borderwidth=3, relief=RAISED
        )
        self.total_label.grid(row=3, column=0, columnspan=3, pady=10, sticky="nsew")
        self.total_label['background'] = "LightPink1"
        
        # Configure grid weights
        for i in range(4):
            self.count_frame.rowconfigure(i, weight=1)
        for i in range(3):
            self.count_frame.columnconfigure(i, weight=1)

    def process_video(self):
        while self.start_counting:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # UDA preprocessing
            frame = uda_preprocess_frame(frame)
            
            # Draw ROI polygon
            jump_pnts = np.array(self.polygon_coords, np.int32)
            cv2.polylines(frame, [jump_pnts], True, (0, 0, 255), 2)
            masked = cv2.bitwise_and(frame, frame, mask=self.mask)
            
            # YOLOv8 detection with custom filtering
            results = self.yolo_model(masked, verbose=False)[0]
            
            boxes, confidences, classIDs = [], [], []
            for result in results.boxes:
                x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
                conf = float(result.conf[0])
                cls_id = int(result.cls[0])
                
                # Get mapped class name
                vehicle_type = self.class_mapping.get(cls_id, None)
                
                # Only process if it's one of our target classes
                if vehicle_type and conf > self.class_thresholds[vehicle_type]:
                    boxes.append([x1, y1, x2 - x1, y2 - y1])
                    confidences.append(conf)
                    classIDs.append(cls_id)
            
            # Count vehicles
            self.count_vehicles(boxes, classIDs, frame)
            
            # Update GUI
            self.update_gui(frame, boxes, classIDs, confidences)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to stop
                self.start_counting = False
                
        self.cap.release()
        self.selection_label.config(text="Counting completed!")

    def count_vehicles(self, boxes, classIDs, frame):
        current_detections = {}
        
        for i, (box, classID) in enumerate(zip(boxes, classIDs)):
            x, y, w, h = box
            centerX = x + (w // 2)
            centerY = y + (h // 2)
            
            vehicle_type = self.class_mapping.get(classID, "unknown")
            if vehicle_type == "unknown":
                continue
                
            current_detections[(centerX, centerY)] = self.vehicle_count
            
            if not self.box_in_previous_frames((centerX, centerY, w, h), current_detections):
                self.vehicle_count += 1
                self.count[vehicle_type] += 1
                
            # Draw vehicle ID and type
            cv2.putText(frame, f"{vehicle_type}:{self.vehicle_count}", (centerX, centerY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)
        
        # Update previous detections
        self.previous_frame_detections.pop(0)
        self.previous_frame_detections.append(current_detections)

    def box_in_previous_frames(self, current_box, current_detections):
        centerX, centerY, width, height = current_box
        dist = np.inf
        
        for i in range(self.FRAMES_BEFORE_CURRENT):
            coordinate_list = list(self.previous_frame_detections[i].keys())
            if not coordinate_list:
                continue
                
            temp_dist, index = spatial.KDTree(coordinate_list).query([(centerX, centerY)])
            if temp_dist < dist:
                dist = temp_dist
                frame_num = i
                coord = coordinate_list[index[0]]
        
        if dist > (max(width, height)/2):
            return False
            
        current_detections[(centerX, centerY)] = self.previous_frame_detections[frame_num][coord]
        return True

    def update_gui(self, frame, boxes, classIDs, confidences):
        # Update count labels
        self.count_labels["bicycles"].config(text=f"Bicycles: {self.count['bicycle']}")
        self.count_labels["motorbikes"].config(text=f"Motorbikes: {self.count['motorbike']}")
        self.count_labels["autorickshaws"].config(text=f"Autorickshaws: {self.count['auto']}")
        self.count_labels["cars"].config(text=f"Cars: {self.count['car']}")
        self.count_labels["trucks"].config(text=f"Trucks: {self.count['truck']}")
        self.count_labels["rickshaws"].config(text=f"Rickshaws: {self.count['rickshaw']}")
        self.count_labels["vans"].config(text=f"Vans: {self.count['van']}")
        self.count_labels["minitrucks"].config(text=f"Minitrucks: {self.count['minitruck']}")
        self.count_labels["buses"].config(text=f"Buses: {self.count['bus']}")
        self.total_label.config(text=f"Total vehicles: {self.vehicle_count}")
        
        # Display frame
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        self.draw_detection_boxes(boxes, classIDs, confidences, cv2image)
        
        img = Image.fromarray(cv2image)
        img = img.resize((640, 480))  # Fixed size for display
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        
        self.root.update()

    def draw_detection_boxes(self, boxes, classIDs, confidences, frame):
        for i, (box, classID, conf) in enumerate(zip(boxes, classIDs, confidences)):
            x, y, w, h = box
            color = (0, 255, 0)  # Green boxes
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label with class name and confidence
            vehicle_type = self.class_mapping.get(classID, "unknown")
            label = f"{vehicle_type}: {conf:.2f}"
            cv2.putText(frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

if __name__ == "__main__":
    counter = VehicleCounter()
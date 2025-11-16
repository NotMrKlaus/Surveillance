from insightface.app import FaceAnalysis
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel
from PIL import Image, ImageTk

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

class ImagePreview:
    def __init__(self, path):
        self.root = Toplevel()
        self.root.title("Image Preview")
        self.root.geometry("400x500")
        
        img = Image.open(path)
        img = img.resize((380, 450), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        
        label = tk.Label(self.root, image=photo)
        label.image = photo
        label.pack(pady=20)
        
        tk.Button(self.root, text="OK", command=self.root.destroy).pack()

class FaceMatcher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Face Matcher")
        self.root.geometry("1000x600")
        
        self.img1_path = None
        self.img2_path = None
        self.img1_label = None
        self.img2_label = None
        self.result_label = None
        
        self.setup_ui()
    
    def setup_ui(self):
        tk.Button(self.root, text="Select Image 1", command=self.select_img1, bg='lightblue').pack(pady=10)
        tk.Button(self.root, text="Select Image 2", command=self.select_img2, bg='lightgreen').pack(pady=10)
        tk.Button(self.root, text="Compare", command=self.compare, bg='orange').pack(pady=10)
        
        img_frame = tk.Frame(self.root)
        img_frame.pack(pady=20)
        
        self.img1_label = tk.Label(img_frame, text="Image 1", width=40, height=20, bg='gray')
        self.img1_label.pack(side=tk.LEFT, padx=10)
        
        self.img2_label = tk.Label(img_frame, text="Image 2", width=40, height=20, bg='gray')
        self.img2_label.pack(side=tk.LEFT, padx=10)
        
        self.result_label = tk.Label(self.root, text="", font=('Arial', 16), fg='red')
        self.result_label.pack(pady=20)
    
    def select_img1(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if path:
            ImagePreview(path)  # Show preview
            self.img1_path = path
            self.display_image(self.img1_label, path)
    
    def select_img2(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if path:
            ImagePreview(path)  # Show preview
            self.img2_path = path
            self.display_image(self.img2_label, path)
    
    def display_image(self, label, path):
        img = Image.open(path)
        img = img.resize((200, 200), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        label.config(image=photo, text="", compound='top')
        label.image = photo
    
    def compare(self):
        if not self.img1_path or not self.img2_path:
            messagebox.showerror("Error", "Select both images!")
            return
        
        img1 = cv2.imread(self.img1_path)
        img2 = cv2.imread(self.img2_path)
        
        faces1 = app.get(img1)
        faces2 = app.get(img2)
        
        if not faces1 or not faces2:
            self.result_label.config(text="No faces detected!", fg='red')
            return
        
        emb1 = faces1[0].normed_embedding
        emb2 = faces2[0].normed_embedding
        distance = np.dot(emb1, emb2)
        
        is_same = distance > 0.6
        color = 'green' if is_same else 'red'
        text = f"SAME PERSON! ({distance:.3f})" if is_same else f"DIFFERENT! ({distance:.3f})"
        
        self.result_label.config(text=text, fg=color)
    
    def run(self):
        self.root.mainloop()

FaceMatcher().run()
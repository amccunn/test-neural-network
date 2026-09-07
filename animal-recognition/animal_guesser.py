import time
import tkinter as tk
import numpy as np
from tkinter import messagebox
from PIL import Image, ImageGrab

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def apply_sigmoid(pixel_matrix):
    return [sigmoid((pixel - 100) / 25) for pixel in pixel_matrix]

class AnimalSelectionScreen:

    def __init__(self, root):
        self.root = root
        self.root.title("Select an Animal")
        self.root.geometry("200x400")

        tk.Label(root, text="Select an animal to draw:").pack(pady=10)

        self.animal_var = tk.StringVar(value="cat")
        animals = ["cat", "dog", "elephant", "giraffe", "lion", "monkey", "panda", "penguin", "rabbit", "tiger"]
        for animal in animals:
            tk.Radiobutton(root, text=animal.capitalize(),
                           variable=self.animal_var, value=animal).pack(anchor="w")

        tk.Button(root, text="Start Drawing", command=self.start_drawing).pack(pady=10)

    def start_drawing(self):
        selected_animal = self.animal_var.get()
        self.root.destroy()  # Close the selection window
        drawing_root = tk.Tk()
        DrawingApp(drawing_root, selected_animal)
        drawing_root.mainloop()


class DrawingApp:

    def __init__(self, root, animal):
        self.root = root
        self.root.title("Drawing Game")
        self.animal = animal
        tk.Label(root, text=f"Draw a {self.animal}").pack(pady=10)

        # Canvas setup
        self.canvas = tk.Canvas(root, width=600, height=400, bg="white")
        self.canvas.pack(pady=10)

        # Shape selection
        self.shape_var = tk.StringVar(value="line")
        shapes = ["line", "rectangle", "oval", "freehand"]
        tk.Label(root, text="Select Shape:").pack()
        for shape in shapes:
            tk.Radiobutton(root, text=shape.capitalize(),
                           variable=self.shape_var, value=shape).pack(anchor="w")

        # Buttons
        tk.Button(root, text="Clear Canvas", command=self.clear_canvas).pack(pady=5)
        tk.Button(root, text="Submit Image", command=self.submit).pack(pady=5)

        # Event bindings
        self.start_x = None
        self.start_y = None
        self.current_item = None

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def on_click(self, event):
        """Store starting coordinates when mouse is clicked."""
        self.start_x, self.start_y = event.x, event.y
        if self.shape_var.get() == "freehand":
            self.current_item = self.canvas.create_line(
                self.start_x, self.start_y, event.x, event.y, fill="black"
            )

    def on_drag(self, event):
        """Draw shape dynamically while dragging."""
        shape = self.shape_var.get()
        if shape == "freehand":
            self.canvas.create_line(
                self.start_x, self.start_y, event.x, event.y, fill="black"
            )
            self.start_x, self.start_y = event.x, event.y
        else:
            if self.current_item:
                self.canvas.delete(self.current_item)
            if shape == "line":
                self.current_item = self.canvas.create_line(
                    self.start_x, self.start_y, event.x, event.y, fill="black"
                )
            elif shape == "rectangle":
                self.current_item = self.canvas.create_rectangle(
                    self.start_x, self.start_y, event.x, event.y, outline="black"
                )
            elif shape == "oval":
                self.current_item = self.canvas.create_oval(
                    self.start_x, self.start_y, event.x, event.y, outline="black"
                )

    def on_release(self, event):
        """Finalize shape when mouse is released."""
        self.on_drag(event)  # Ensure final shape is drawn
        self.current_item = None

    def clear_canvas(self):
        """Clear all drawings from the canvas."""
        if messagebox.askyesno("Clear Canvas", "Are you sure you want to clear the canvas?"):
            self.canvas.delete("all")

    def get_grayscale_pixels(self):

        # Force Tkinter to update geometry so coordinates are accurate
        self.root.update()
        
        # Get screen coordinates of the canvas widget
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        w = x + self.canvas.winfo_width()
        h = y + self.canvas.winfo_height()
        
        # Capture directly from screen and convert to grayscale ('L')
        img = ImageGrab.grab(bbox=(x, y, w, h)).convert("L")
        
        width, height = img.size
        raw_pixels = list(img.getdata())
        
        # # Convert flat list to 2D matrix
        # pixel_matrix = [
        #     raw_pixels[i * width:(i + 1) * width] 
        #     for i in range(height)
        # ]
        with open ("training_data/" + self.animal + "_" + str(time.time()) + ".txt", "a") as f:
            f.write(str(list(map(float, apply_sigmoid(raw_pixels)))))


    def submit(self):
        """Submit the drawing and close the application."""
        self.get_grayscale_pixels()
        self.root.quit()

# Run the application
if __name__ == "__main__":

    root = tk.Tk()
    app = AnimalSelectionScreen(root)
    root.mainloop()

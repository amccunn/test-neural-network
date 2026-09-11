import time
import tkinter as tk
from tkinter import messagebox
import os
import json
import numpy as np
from PIL import Image, ImageGrab, ImageTk
import ctypes

try:
    # This fixes the screen scaling coordinate mismatch on Windows
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass  # Just pass if

ANIMALS = ["cat", "dog", "elephant", "giraffe", "lion", "monkey", "panda", "penguin", "rabbit", "tiger"]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def apply_sigmoid(pixel_matrix):
    return [[sigmoid((pixel - 100) / 25) for pixel in row] for row in pixel_matrix]

class GameController:

    #This class manages the game flow, including player turns, scoring, and transitions between screens.
    def __init__(self, root):

        self.root = root
        self.root.title("Drawing Game")
        self.root.geometry("600x600")
        
        self.p1_score = 0
        self.p2_score = 0
        self.target_score = 50  # Default to win
        
        self.current_chooser = 1
        self.current_animal = ""
        
        # Temporary file paths for the drawings
        self.p1_image_path = "p1_temp.png"
        self.p2_image_path = "p2_temp.png"

        self.show_setup_screen()

    def clear_window(self):
        #allows us to clear the window before showing a new screen, so we dont need to create/destroy multiple windows
        for widget in self.root.winfo_children():
            widget.destroy()

    #sets up inital "welcome screen"
    def show_setup_screen(self):
        self.clear_window()
        tk.Label(self.root, text="Welcome to the Drawing Game!", font=("Arial", 18, "bold")).pack(pady=20)
        
        tk.Label(self.root, text="Points needed to win:", font=("Arial", 12)).pack(pady=10)
        self.target_var = tk.IntVar(value=30)
        tk.Entry(self.root, textvariable=self.target_var, font=("Arial", 12)).pack(pady=5)
        
        tk.Button(self.root, text="Start Game", font=("Arial", 14), 
                  command=self.start_game).pack(pady=20)

    def start_game(self):
        self.target_score = self.target_var.get()
        self.show_animal_selection()

    def show_animal_selection(self):
        self.clear_window()
        tk.Label(self.root, text=f"Player {self.current_chooser}, select an animal!", font=("Arial", 16, "bold")).pack(pady=20)

        self.animal_var = tk.StringVar(value=ANIMALS[0])
        
        # Frame for radio buttons
        frame = tk.Frame(self.root)
        frame.pack(pady=10)
        
        for animal in ANIMALS:
            tk.Radiobutton(frame, text=animal.capitalize(), font=("Arial", 12),
                           variable=self.animal_var, value=animal).pack(anchor="w")

        tk.Button(self.root, text="Confirm Animal", font=("Arial", 14), 
                  command=self.start_p1_draw).pack(pady=20)

    def start_p1_draw(self):
        self.current_animal = self.animal_var.get()
        self.clear_window()
        DrawingScreen(self.root, self, player_name="Player 1", 
                      animal=self.current_animal, save_path=self.p1_image_path, next_step=self.start_p2_draw)

    def start_p2_draw(self):
        self.clear_window()
        DrawingScreen(self.root, self, player_name="Player 2", 
                      animal=self.current_animal, save_path=self.p2_image_path, next_step=self.show_rating_screen)

    def show_rating_screen(self):
        self.clear_window()
        RatingScreen(self.root, self)

    def apply_scores(self, p1_points, p2_points):
        self.p1_score += p1_points
        self.p2_score += p2_points
        self.show_scoreboard()

    def show_scoreboard(self):
        self.clear_window()
        tk.Label(self.root, text="Scoreboard", font=("Arial", 20, "bold")).pack(pady=20)
        
        tk.Label(self.root, text=f"Player 1: {self.p1_score} / {self.target_score}", font=("Arial", 16)).pack(pady=10)
        tk.Label(self.root, text=f"Player 2: {self.p2_score} / {self.target_score}", font=("Arial", 16)).pack(pady=10)

        if self.p1_score >= self.target_score or self.p2_score >= self.target_score:
            winner = "Player 1" if self.p1_score > self.p2_score else "Player 2"
            if self.p1_score == self.p2_score:
                winner = "It's a Tie!"
            tk.Label(self.root, text=f"WINNER: {winner}!!!", font=("Arial", 20, "bold"), fg="green").pack(pady=30)
            tk.Button(self.root, text="Play Again", font=("Arial", 14), command=self.reset_game).pack()
        else:
            # Swap who chooses the animal
            self.current_chooser = 2 if self.current_chooser == 1 else 1
            tk.Button(self.root, text="Next Round", font=("Arial", 14), command=self.show_animal_selection).pack(pady=30)

    def reset_game(self):
        self.p1_score = 0
        self.p2_score = 0
        self.current_chooser = 1
        self.show_setup_screen()


class DrawingScreen:
    """Adapted from your DrawingApp to work as a frame within the GameController."""
    def __init__(self, root, controller, player_name, animal, save_path, next_step):
        self.root = root
        self.controller = controller
        self.player_name = player_name
        self.animal = animal
        self.save_path = save_path
        self.next_step = next_step

        self.SIZE = 350
        self.COLOURS = {"black": "#000000", "white": "#FFFFFF"}

        tk.Label(root, text=f"{self.player_name}: Draw a {self.animal}!", font=("Arial", 16, "bold")).pack(pady=5)

        # Tools Frame
        tools_frame = tk.Frame(root)
        tools_frame.pack()

        self.shape_var = tk.StringVar(value="freehand")
        shapes = ["freehand", "line", "rectangle", "oval"]
        for shape in shapes:
            tk.Radiobutton(tools_frame, text=shape.capitalize(),
                           variable=self.shape_var, value=shape).pack(side=tk.LEFT, padx=5)

        # Canvas setup
        self.canvas = tk.Canvas(root, width=self.SIZE, height=self.SIZE, bg=self.COLOURS["white"], highlightbackground="black", highlightthickness=2)
        self.canvas.pack(pady=10)

        # Buttons
        tk.Button(root, text="Clear Canvas", command=self.clear_canvas).pack(pady=5)
        tk.Button(root, text="Submit Drawing", font=("Arial", 12, "bold"), command=self.submit).pack(pady=10)

        # Event bindings
        self.start_x = None
        self.start_y = None
        self.current_item = None

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def on_click(self, event):
        self.start_x, self.start_y = event.x, event.y
        if self.shape_var.get() == "freehand":
            self.current_item = self.canvas.create_line(self.start_x, self.start_y, event.x, event.y, fill=self.COLOURS["black"], width=2)

    def on_drag(self, event):
        shape = self.shape_var.get()
        if shape == "freehand":
            self.canvas.create_line(self.start_x, self.start_y, event.x, event.y, fill=self.COLOURS["black"], width=2)
            self.start_x, self.start_y = event.x, event.y
        else:
            if self.current_item:
                self.canvas.delete(self.current_item)
            if shape == "line":
                self.current_item = self.canvas.create_line(self.start_x, self.start_y, event.x, event.y, fill=self.COLOURS["black"], width=2)
            elif shape == "rectangle":
                self.current_item = self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline=self.COLOURS["black"], width=2)
            elif shape == "oval":
                self.current_item = self.canvas.create_oval(self.start_x, self.start_y, event.x, event.y, outline=self.COLOURS["black"], width=2)

    def on_release(self, event):
        self.on_drag(event)
        self.current_item = None

    def clear_canvas(self):
        if messagebox.askyesno("Clear Canvas", "Are you sure you want to clear the canvas?"):
            self.canvas.delete("all")

    def submit(self):
            # Force update to ensure coordinates are right
            self.root.update_idletasks()
            
            # Grab image from screen
            x = self.canvas.winfo_rootx()
            y = self.canvas.winfo_rooty()
            w = x + self.canvas.winfo_width()
            h = y + self.canvas.winfo_height()
            
            # Capture the image once
            img = ImageGrab.grab(bbox=(x, y, w, h))
            
            #Save temp PNG for the Rating Screen UI
            img.save(self.save_path) 
            
            #Save JSON Data for your training model
            img_gray = img.convert("L")
            width, height = img_gray.size
            raw_pixels = list(img_gray.getdata())

            pixels_2d = np.array([
                raw_pixels[i * width:(i + 1) * width]
                for i in range(height)
            ])
            
            # Ensure the directory exists so it doesn't crash
            if not os.path.exists("training_data"):
                os.makedirs("training_data")

            # Save the JSON file
            file_name = f"training_data/{self.animal}_{time.time()}.json"
            with open(file_name, "w") as f:
                json.dump(apply_sigmoid(pixels_2d.tolist()), f)
            
            # Move to the next phase of the game
            self.next_step()

class RatingScreen:
    """Displays both images and allows players to rate each other."""
    def __init__(self, root, controller):
        self.root = root
        self.controller = controller
        
        tk.Label(root, text=f"Rate the {controller.current_animal.capitalize()}s!", font=("Arial", 18, "bold")).pack(pady=10)

        # Frame to hold both images side by side
        img_frame = tk.Frame(root)
        img_frame.pack()

        # Load P1 Image
        try:
            self.img1 = ImageTk.PhotoImage(Image.open(controller.p1_image_path).resize((250, 250)))
        except Exception:
            self.img1 = None # Fallback if error

        p1_frame = tk.Frame(img_frame)
        p1_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(p1_frame, text="Player 1's Drawing", font=("Arial", 12)).pack()
        tk.Label(p1_frame, image=self.img1, relief=tk.SUNKEN).pack()
        tk.Label(p1_frame, text="Player 2, rate this (1-10):").pack(pady=5)
        self.p1_rating = tk.Scale(p1_frame, from_=1, to=10, orient=tk.HORIZONTAL)
        self.p1_rating.pack()

        # Load P2 Image
        try:
            self.img2 = ImageTk.PhotoImage(Image.open(controller.p2_image_path).resize((250, 250)))
        except Exception:
            self.img2 = None

        p2_frame = tk.Frame(img_frame)
        p2_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(p2_frame, text="Player 2's Drawing", font=("Arial", 12)).pack()
        tk.Label(p2_frame, image=self.img2, relief=tk.SUNKEN).pack()
        tk.Label(p2_frame, text="Player 1, rate this (1-10):").pack(pady=5)
        self.p2_rating = tk.Scale(p2_frame, from_=1, to=10, orient=tk.HORIZONTAL)
        self.p2_rating.pack()

        tk.Button(root, text="Submit Ratings", font=("Arial", 14, "bold"), command=self.submit_ratings).pack(pady=20)

    def submit_ratings(self):
        # Give P1 points based on P2's rating, and vice versa
        p1_points_earned = self.p1_rating.get()
        p2_points_earned = self.p2_rating.get()
        self.controller.apply_scores(p1_points_earned, p2_points_earned)


if __name__ == "__main__":
    root = tk.Tk()
    app = GameController(root)
    root.mainloop()
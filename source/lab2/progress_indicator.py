import tkinter as tk
from tkinter import ttk
import time

class ProgressIndicator:
    def __init__(self, parent):
        self.parent = parent
        self.progress_window = tk.Toplevel(parent)
        self.progress_window.title("Загрузка")
        self.progress_window.geometry("400x100")
        self.progress_window.resizable(False, False)

        self.progress_label = tk.Label(self.progress_window, text="Выполняются эксперименты...")
        self.progress_label.pack(pady=10)

        self.progress_bar = ttk.Progressbar(self.progress_window, orient="horizontal", length=350, mode="determinate")
        self.progress_bar.pack()

        # Задаем максимальное значение для прогресс-бара
        self.progress_bar["maximum"] = 100

    def update_progress(self, value):
        self.progress_bar["value"] = value
        self.progress_bar.update()
    
    def set_label(self, new_text):
        self.progress_label.config(text=new_text)

    def close(self):
        self.progress_window.destroy()
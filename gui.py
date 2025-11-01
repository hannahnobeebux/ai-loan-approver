import tkinter as tk

class Gui:

    def __init__(self):
        self.window = tk.Tk()
        self.create_window()

    def create_window(self):
        self.window.title("AI Loan Manager - Prototype")
        self.centre_sized_window(800, 500)

    def centre_sized_window(self, width, height):
        self.window.update_idletasks()
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.window.geometry(f"{width}x{height}+{x}+{y}")


if __name__ == '__main__':
    frontend = Gui()
    frontend.window.mainloop()
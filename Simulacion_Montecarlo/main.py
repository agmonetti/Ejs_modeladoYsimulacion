import tkinter as tk
from interfaz import SimuladorApp

if __name__ == "__main__":
    root = tk.Tk()
    
    # Intenta aplicar un tema más moderno si está disponible en Linux
    try:
        style = tk.ttk.Style()
        style.theme_use('clam')
    except:
        pass
        
    app = SimuladorApp(root)
    root.mainloop()
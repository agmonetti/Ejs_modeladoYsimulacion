import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import motor_montecarlo as motor

class SimuladorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulador Monte Carlo - UADE")
        self.root.geometry("1200x850")
        
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.tab_1d = ttk.Frame(self.notebook)
        self.tab_multi = ttk.Frame(self.notebook)
        self.tab_estadistica = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_1d, text="Integrales Simples (1D)")
        self.notebook.add(self.tab_multi, text="Integrales Dobles/Triples")
        self.notebook.add(self.tab_estadistica, text="Análisis Estadístico (Intervalos)")
        
        self.construir_tab_1d()
        self.construir_tab_multi()
        self.construir_tab_estadistica()

    # =========================================================
    #  TAB 1D
    # =========================================================
    def construir_tab_1d(self):
        frame_top = ttk.LabelFrame(self.tab_1d, text="Parámetros 1D")
        frame_top.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(frame_top, text="f(x)=").grid(row=0, column=0)
        self.entry_fx = ttk.Entry(frame_top, width=20)
        self.entry_fx.grid(row=0, column=1, padx=5)
        self.entry_fx.insert(0, "log(x)")
        
        ttk.Label(frame_top, text="a=").grid(row=0, column=2)
        self.entry_a = ttk.Entry(frame_top, width=5)
        self.entry_a.grid(row=0, column=3, padx=5)
        self.entry_a.insert(0, "1")
        
        ttk.Label(frame_top, text="b=").grid(row=0, column=4)
        self.entry_b = ttk.Entry(frame_top, width=5)
        self.entry_b.grid(row=0, column=5, padx=5)
        self.entry_b.insert(0, "5")
        
        ttk.Label(frame_top, text="N=").grid(row=0, column=6)
        self.entry_N = ttk.Entry(frame_top, width=8)
        self.entry_N.grid(row=0, column=7, padx=5)
        self.entry_N.insert(0, "10000")
        
        ttk.Label(frame_top, text="Seed:").grid(row=0, column=8)
        self.entry_seed = ttk.Entry(frame_top, width=6)
        self.entry_seed.grid(row=0, column=9, padx=5)
        
        ttk.Button(frame_top, text="Hit-or-Miss", command=self.accion_hit_or_miss).grid(row=0, column=10, padx=5)
        ttk.Button(frame_top, text="Promedio", command=self.accion_valor_promedio).grid(row=0, column=11, padx=5)
        ttk.Button(frame_top, text="Convergencia", command=self.accion_convergencia).grid(row=0, column=12, padx=5)

        # --- NUEVO TECLADO MATEMÁTICO AVANZADO ---
        frame_teclado = ttk.Frame(frame_top)
        frame_teclado.grid(row=1, column=0, columnspan=13, pady=5)
        btn_teclado = ttk.Button(frame_teclado, text="🖩 Abrir Teclado Matemático Completo", 
                                 command=lambda: self.abrir_teclado_avanzado(self.entry_fx))
        btn_teclado.pack(pady=5)

        # --- PANEL INFERIOR (TABLA Y GRÁFICO) ---
        frame_bottom = ttk.Frame(self.tab_1d)
        frame_bottom.pack(fill="both", expand=True, padx=5, pady=5)
        
        frame_tabla = ttk.LabelFrame(frame_bottom, text="Muestras (Limitado a 1000 filas)")
        frame_tabla.pack(side="left", fill="y", expand=False)
        
        columnas = ("x", "y", "f(x)", "Éxito")
        self.tabla = ttk.Treeview(frame_tabla, columns=columnas, show="headings", height=20)
        for col in columnas:
            self.tabla.heading(col, text=col)
            self.tabla.column(col, width=70, anchor="center")
            
        scrollbar = ttk.Scrollbar(frame_tabla, orient="vertical", command=self.tabla.yview)
        self.tabla.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.tabla.pack(side="left", fill="y", expand=True)
        
        frame_grafico = ttk.LabelFrame(frame_bottom, text="Gráfico Interactivo y Resultados")
        frame_grafico.pack(side="right", fill="both", expand=True, padx=5)
        
        self.fig_1d, self.ax_1d = plt.subplots(figsize=(6, 4))
        self.canvas_1d = FigureCanvasTkAgg(self.fig_1d, master=frame_grafico)
        self.canvas_1d.get_tk_widget().pack(fill="both", expand=True)
        
        self.lbl_pizarra = ttk.Label(frame_grafico, text="", font=("Courier", 11, "bold"), justify="center", foreground="black")
        self.lbl_pizarra.pack(side="bottom", pady=10)

    # =========================================================
    #  TAB MULTIVARIABLE
    # =========================================================
    def construir_tab_multi(self):
        frame_top = ttk.LabelFrame(self.tab_multi, text="Parámetros Multivariables")
        frame_top.pack(fill="x", padx=5, pady=5)
        
        self.var_dim = tk.IntVar(value=2)
        ttk.Radiobutton(frame_top, text="Doble (2D)", variable=self.var_dim, value=2, command=self.toggle_dim).grid(row=0, column=0)
        ttk.Radiobutton(frame_top, text="Triple (3D)", variable=self.var_dim, value=3, command=self.toggle_dim).grid(row=0, column=1)

        ttk.Label(frame_top, text="f =").grid(row=0, column=2)
        self.entry_f_multi = ttk.Entry(frame_top, width=25)
        self.entry_f_multi.grid(row=0, column=3, padx=5)
        self.entry_f_multi.insert(0, "exp(x+y)")

        ttk.Label(frame_top, text="N=").grid(row=0, column=4)
        self.entry_N_multi = ttk.Entry(frame_top, width=8)
        self.entry_N_multi.grid(row=0, column=5, padx=5)
        self.entry_N_multi.insert(0, "10000")

        ttk.Label(frame_top, text="Confianza (%):").grid(row=0, column=6)
        self.combo_confianza_multi = ttk.Combobox(frame_top, values=["50","55","60","65","70","75","80", "85", "90", "95", "99"], width=4)
        self.combo_confianza_multi.grid(row=0, column=7, padx=5)
        self.combo_confianza_multi.set("95")

        # --- BOTÓN CON TEXTO DINÁMICO ---
        self.btn_multi = ttk.Button(frame_top, text="Calcular Integral Doble", command=self.accion_multi)
        self.btn_multi.grid(row=0, column=8, padx=10)
        
        ttk.Button(frame_top, text="🖩 Teclado", command=lambda: self.abrir_teclado_avanzado(self.entry_f_multi)).grid(row=0, column=9, padx=5)

        frame_lims = ttk.Frame(frame_top)
        frame_lims.grid(row=1, column=0, columnspan=10, pady=5)
        
        ttk.Label(frame_lims, text="X: a=").grid(row=0, column=0)
        self.ax = ttk.Entry(frame_lims, width=5); self.ax.grid(row=0, column=1); self.ax.insert(0, "0")
        ttk.Label(frame_lims, text="b=").grid(row=0, column=2)
        self.bx = ttk.Entry(frame_lims, width=5); self.bx.grid(row=0, column=3); self.bx.insert(0, "2")

        ttk.Label(frame_lims, text="  |  Y: a=").grid(row=0, column=4)
        self.ay = ttk.Entry(frame_lims, width=5); self.ay.grid(row=0, column=5); self.ay.insert(0, "1")
        ttk.Label(frame_lims, text="b=").grid(row=0, column=6)
        self.by = ttk.Entry(frame_lims, width=5); self.by.grid(row=0, column=7); self.by.insert(0, "3")

        ttk.Label(frame_lims, text="  |  Z: a=").grid(row=0, column=8)
        self.az = ttk.Entry(frame_lims, width=5); self.az.grid(row=0, column=9); self.az.insert(0, "0")
        ttk.Label(frame_lims, text="b=").grid(row=0, column=10)
        self.bz = ttk.Entry(frame_lims, width=5); self.bz.grid(row=0, column=11); self.bz.insert(0, "1")
        
        self.az.config(state="disabled"); self.bz.config(state="disabled")

        frame_bottom = ttk.Frame(self.tab_multi)
        frame_bottom.pack(fill="both", expand=True, padx=5, pady=5)
        
        frame_grafico = ttk.LabelFrame(frame_bottom, text="Mapa de Dispersión (Max 2000 ptos)")
        frame_grafico.pack(side="left", fill="both", expand=True)
        self.fig_multi = plt.Figure(figsize=(6, 4))
        self.canvas_multi = FigureCanvasTkAgg(self.fig_multi, master=frame_grafico)
        self.canvas_multi.get_tk_widget().pack(fill="both", expand=True)
        
        frame_resultados = ttk.LabelFrame(frame_bottom, text="Resultados Estadísticos")
        frame_resultados.pack(side="right", fill="both", expand=True, padx=5)
        self.lbl_resultados_multi = ttk.Label(frame_resultados, text="...", font=("Courier", 13, "bold"), justify="left", foreground="#1a1a1a", background="#f5f5f5")
        self.lbl_resultados_multi.pack(padx=15, pady=15, fill="both", expand=True, anchor="nw")
        

    # =========================================================
    #  TAB ESTADÍSTICA
    # =========================================================
    def construir_tab_estadistica(self):
        frame_top = ttk.Frame(self.tab_estadistica)
        frame_top.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(frame_top, text="Simulaciones (M):").pack(side="left", padx=5)
        self.entry_M = ttk.Entry(frame_top, width=6)
        self.entry_M.pack(side="left", padx=5)
        self.entry_M.insert(0, "1000")
        
        ttk.Label(frame_top, text=" | Confianza (%):").pack(side="left", padx=5)
        self.combo_confianza = ttk.Combobox(frame_top, values=["50","55","60","65","70","75","80", "85", "90", "95", "99"], width=4)
        self.combo_confianza.pack(side="left", padx=5)
        self.combo_confianza.set("95") 
        
        # --- RESTAURADO: MÁXIMO ERROR ---
        ttk.Label(frame_top, text=" | Máx Error:").pack(side="left", padx=5)
        self.entry_ancho = ttk.Entry(frame_top, width=6)
        self.entry_ancho.pack(side="left", padx=5)
        self.entry_ancho.insert(0, "0.01")
        
        # --- NUEVO: FACTOR 'j' OPCIONAL ---
        ttk.Label(frame_top, text=" | Factor (j) opcional:").pack(side="left", padx=5)
        self.entry_j = ttk.Entry(frame_top, width=4)
        self.entry_j.pack(side="left", padx=5)
        # Arranca vacío para que sea opcional
        
        ttk.Button(frame_top, text="Ejecutar Análisis", command=self.accion_estadistica).pack(side="left", padx=15)
        
        self.frame_res = ttk.LabelFrame(self.tab_estadistica, text="Intervalos y Resultados")
        self.frame_res.pack(fill="x", padx=10, pady=5)
        self.lbl_stats = ttk.Label(self.frame_res, text="...", font=("Arial", 11))
        self.lbl_stats.pack(padx=10, pady=10)
        
        frame_grafico = ttk.LabelFrame(self.tab_estadistica, text="Histograma de Convergencia")
        frame_grafico.pack(fill="both", expand=True, padx=10, pady=5)
        self.fig_stats, self.ax_stats = plt.subplots(figsize=(6, 4))
        self.canvas_stats = FigureCanvasTkAgg(self.fig_stats, master=frame_grafico)
        self.canvas_stats.get_tk_widget().pack(fill="both", expand=True)
   # =========================================================
    #  VENTANA DE TECLADO MATEMÁTICO (POP-UP)
    # =========================================================
    def abrir_teclado_avanzado(self, entry_destino):
        ventana = tk.Toplevel(self.root)
        ventana.title("Teclado Matemático de SymPy")
        
        # --- AUMENTAMOS EL TAMAÑO Y PERMITIMOS REDIMENSIONAR ---
        ventana.geometry("930x560") 
        ventana.resizable(True, True) 
        ventana.transient(self.root) 
        
        def insertar(texto):
            if texto == 'C':
                entry_destino.delete(0, tk.END)
            else:
                entry_destino.insert(tk.END, texto)
                
        botones_math = {
            "Aritmética Básica": ['+', '-', '*', '/', '**', '(', ')', 'x', 'y', 'z'],
            "Trigonometría": ['sin(', 'cos(', 'tan(', 'sec(', 'csc(', 'cot('],
            "Trig. Inversas (Arcos)": ['asin(', 'acos(', 'atan('],
            "Hiperbólicas": ['sinh(', 'cosh(', 'tanh('],
            "Logaritmos y Raíces": ['log(', 'log10(', 'sqrt(', 'exp('],
            "Constantes": ['pi', 'E']
        }
        
        # Contenedor principal para darle un margen general elegante
        main_frame = ttk.Frame(ventana)
        main_frame.pack(fill="both", expand=True, padx=20, pady=15)
        
        for categoria, teclas in botones_math.items():
            frame_cat = ttk.LabelFrame(main_frame, text=categoria)
            frame_cat.pack(fill="x", padx=5, pady=6) # Más espacio entre categorías
            for col_idx, tecla in enumerate(teclas):
                # Botones más anchos (width=8) y con más separación
                ttk.Button(frame_cat, text=tecla, width=8, 
                           command=lambda t=tecla: insertar(t)).grid(row=0, column=col_idx, padx=4, pady=6)
            
        ttk.Button(main_frame, text="Limpiar Todo (C)", width=25, command=lambda: insertar('C')).pack(pady=15)

    # =========================================================
    #  CONTROLADORES LÓGICOS
    # =========================================================
    def aplicar_semilla(self):
        seed_txt = self.entry_seed.get().strip()
        if seed_txt:
            try:
                np.random.seed(int(seed_txt))
            except ValueError:
                messagebox.showwarning("Aviso", "La semilla debe ser un número entero.")
                np.random.seed(None)
        else:
            np.random.seed(None) 

    def toggle_dim(self):
        """Alterna controles y cambia el texto del botón según 2D o 3D"""
        if self.var_dim.get() == 3:
            self.az.config(state="normal")
            self.bz.config(state="normal")
            self.btn_multi.config(text="Calcular Integral Triple")
        else:
            self.az.config(state="disabled")
            self.bz.config(state="disabled")
            self.btn_multi.config(text="Calcular Integral Doble")

    def leer_parametros_1d(self):
        self.aplicar_semilla()
        try:
            a = float(eval(self.entry_a.get().replace('pi', str(np.pi))))
            b = float(eval(self.entry_b.get().replace('pi', str(np.pi))))
            N = int(self.entry_N.get())
            txt_f = self.entry_fx.get()
            f, err = motor.compilar_funcion(txt_f, 'x')
            if err: messagebox.showerror("Error", err); return None
            return f, a, b, N, txt_f
        except:
            messagebox.showerror("Error", "Revise los parámetros numéricos.")
            return None

    def accion_hit_or_miss(self):
        params = self.leer_parametros_1d()
        if not params: return
        f, a, b, N, txt_f = params
        
        I, xr, yr, exitos, f_eval = motor.simular_hit_or_miss_1d(f, a, b, N)
        self.ax_1d.clear()
        
        limite_graf = min(N, 5000)
        self.ax_1d.scatter(xr[:limite_graf][exitos[:limite_graf]], yr[:limite_graf][exitos[:limite_graf]], color='green', s=5, alpha=0.5)
        self.ax_1d.scatter(xr[:limite_graf][~exitos[:limite_graf]], yr[:limite_graf][~exitos[:limite_graf]], color='red', s=5, alpha=0.5)
        
        xp = np.linspace(a, b, 500)
        self.ax_1d.plot(xp, f(xp), color='blue', lw=2)
        self.ax_1d.set_title(f"Método Hit-or-Miss | Integral ~= {I:.6f}")
        self.canvas_1d.draw()
        
        self.tabla.delete(*self.tabla.get_children()) 
        limite_tabla = min(N, 1000) 
        for i in range(limite_tabla):
            marca = "✓" if exitos[i] else "✗"
            self.tabla.insert("", "end", values=(f"{xr[i]:.4f}", f"{yr[i]:.4f}", f"{f_eval[i]:.4f}", marca))
            
        self.lbl_pizarra.config(text=f"Aprox: Î ≅ {I:.6f} \n(Método Hit-or-Miss no usa sumatoria directa)")

    def accion_valor_promedio(self):
        params = self.leer_parametros_1d()
        if not params: return
        f, a, b, N, txt_f = params
        I, xr, f_eval = motor.simular_valor_medio_1d(f, a, b, N)
        
        self.ax_1d.clear()
        limite_graf = min(N, 5000)
        self.ax_1d.scatter(xr[:limite_graf], f_eval[:limite_graf], color='orange', s=5, alpha=0.5)
        xp = np.linspace(a, b, 500)
        self.ax_1d.plot(xp, f(xp), color='blue', lw=2)
        self.ax_1d.axhline(I/(b-a), color='purple', linestyle='--', lw=2, label=f'Media: {I/(b-a):.4f}')
        self.ax_1d.set_title(f"Método Valor Promedio | Integral ~= {I:.6f}")
        self.ax_1d.legend()
        self.canvas_1d.draw()
        
        self.tabla.delete(*self.tabla.get_children())
        limite_tabla = min(N, 1000)
        for i in range(limite_tabla):
            self.tabla.insert("", "end", values=(f"{xr[i]:.4f}", "-", f"{f_eval[i]:.4f}", "Eval"))

        promedio = np.mean(f_eval[np.isfinite(f_eval)])
        base = b - a
        texto_pizarra = (
            f"(b - a) = {base:.4f}\n"
            f"Î ≅ (b - a) * [ (1/N) * Σ f(xi) ]\n"
            f"Î ≅ {base:.4f} * {promedio:.6f}\n"
            f"Î ≅ {I:.6f}"
        )
        self.lbl_pizarra.config(text=texto_pizarra)

    def accion_convergencia(self):
        params = self.leer_parametros_1d()
        if not params: return
        f, a, b, N, txt_f = params
        
        conv = motor.simular_convergencia_1d(f, a, b, N)
        gauss = motor.integral_gauss(f, a, b, 5)
        
        self.ax_1d.clear()
        self.ax_1d.plot(np.arange(1, N+1), conv, color='blue', lw=1.5, label='Aproximación MC')
        self.ax_1d.axhline(gauss, color='red', linestyle='--', lw=2, label=f'Exacto (Gauss: {gauss:.6f})')
        if N > 500: self.ax_1d.set_xscale('log')
        self.ax_1d.set_title("Convergencia MC vs Gauss")
        self.ax_1d.legend()
        self.canvas_1d.draw()
        self.lbl_pizarra.config(text="")

    def accion_estadistica(self):
        params = self.leer_parametros_1d()
        if not params: return
        f, a, b, N, txt_f = params
        
        try:
            M = int(self.entry_M.get())
            texto_confianza = self.combo_confianza.get().replace('%', '').strip()
            nivel_conf = float(texto_confianza) / 100.0
            
            if not (0 < nivel_conf < 1):
                messagebox.showerror("Error", "La confianza debe estar entre 0.1 y 99.9")
                return
                
            max_error = float(self.entry_ancho.get())
            j_str = self.entry_j.get().strip() # Leemos si escribiste algo en la 'j'
            
        except ValueError:
            messagebox.showerror("Error", "Revise los parámetros estadísticos. Use números válidos.")
            return
        
        stats = motor.analisis_estadistico_1d(f, a, b, N, M, nivel_confianza=nivel_conf)
        
        # 1. Validación del Error Máximo
        margen_error_real = stats['z_score'] * stats['error_estandar']
        if margen_error_real <= max_error:
            validacion = f"-- CUMPLE -- (Error {margen_error_real:.5f} <= {max_error})"
        else:
            validacion = f"-- NO CUMPLE -- (Error {margen_error_real:.5f} > {max_error})"
            
        # 2. Lógica Opcional del Factor 'j'
        texto_j = ""
        if j_str:
            try:
                factor_j = float(j_str)
                n_requerido = int(N * (factor_j**2))
                texto_j = f"\n📌 PREGUNTA DE PARCIAL: Para reducir error {factor_j}x, se necesita N = {n_requerido}"
            except ValueError:
                pass # Si escribió letras en la 'j', lo ignoramos
        
        txt = (f"Función: f(x)={txt_f}  |  Muestras(M): {M}  |  Dardos(N actual): {N}\n"
               f"Media (μ): {stats['media']:.6f}  |  Desvío Estándar (σ): {stats['desviacion']:.6f}\n"
               f"Error Estándar de la Media: {stats['error_estandar']:.6f}\n"
               f"IC {stats['confianza_pct']}%: [ {stats['ic'][0]:.6f} , {stats['ic'][1]:.6f} ]\n"
               f"Validación: {validacion} {texto_j}")
               
        self.lbl_stats.config(text=txt)
        
        self.ax_stats.clear()
        self.ax_stats.hist(stats['distribucion'], bins='auto', color='skyblue', edgecolor='black')
        self.ax_stats.axvline(stats['media'], color='red', ls='--', lw=2, label='Media')
        
        self.ax_stats.axvline(stats['ic'][0], color='green', ls=':', lw=2, label=f'IC Inferior ({stats["confianza_pct"]}%)')
        self.ax_stats.axvline(stats['ic'][1], color='green', ls=':', lw=2, label=f'IC Superior ({stats["confianza_pct"]}%)')
        
        self.ax_stats.set_title(f"Distribución e Intervalo de Confianza al {stats['confianza_pct']}%")
        self.ax_stats.legend()
        self.canvas_stats.draw()

    def accion_multi(self):
        self.aplicar_semilla()
        try:
            dim = self.var_dim.get()
            N = int(self.entry_N_multi.get())
            txt_f = self.entry_f_multi.get()
            confianza_txt = self.combo_confianza_multi.get().replace('%', '').strip()
            confianza = float(confianza_txt) / 100.0
            
            p_ax = float(eval(self.ax.get().replace('pi', str(np.pi))))
            p_bx = float(eval(self.bx.get().replace('pi', str(np.pi))))
            p_ay = float(eval(self.ay.get().replace('pi', str(np.pi))))
            p_by = float(eval(self.by.get().replace('pi', str(np.pi))))
            
            # Calcular z-score dinámico
            z_score = motor.calcular_z_score(confianza)
            
            if dim == 2:
                f, err = motor.compilar_funcion(txt_f, 'x y')
                if err: messagebox.showerror("Error", err); return
                lims = [(p_ax, p_bx), (p_ay, p_by)]
                I, x, y, f_eval, stats = motor.simular_integral_multiple(f, lims, N, 2)
                
                escala = stats['escala']
                G = stats['desv_std']  # Desviación estándar
                EE = stats['error_est']  # Error estimado
                IC_inf = I - z_score * EE * escala  # IC inferior
                IC_sup = I + z_score * EE * escala  # IC superior
                
                # Gráfico
                self.fig_multi.clf()
                ax = self.fig_multi.add_subplot(111)
                limite = min(N, 2000)
                sc = ax.scatter(x[:limite], y[:limite], c=f_eval[:limite], cmap='viridis', s=15, alpha=0.8)
                self.fig_multi.colorbar(sc, ax=ax, label="f(x,y)")
                ax.set_title(f"Integral Doble ~= {I:.6f}")
                self.canvas_multi.draw()
                
                # Mostrar resultados estadísticos en label con formato mejorado
                resultado = f"""╔═══════════════════════════════════╗
║   INTEGRAL DOBLE - RESULTADOS     ║
╚═══════════════════════════════════╝

Escala = {escala:.6f}

Î  = {I:.6f}
G (σ) = {G:.6f}
EE = {EE:.8f}
Z-score = {z_score:.6f}

┌─ INTERVALO DE CONFIANZA {confianza*100:.0f}% ─┐
│ Inferior:  {IC_inf:.6f}
│ Superior:  {IC_sup:.6f}
│ Ancho: {IC_sup - IC_inf:.6f}
└──────────────────────────────┘

N = {stats['N']}"""
                self.lbl_resultados_multi.config(text=resultado)
                                                
            elif dim == 3:
                p_az = float(eval(self.az.get().replace('pi', str(np.pi))))
                p_bz = float(eval(self.bz.get().replace('pi', str(np.pi))))
                f, err = motor.compilar_funcion(txt_f, 'x y z')
                if err: messagebox.showerror("Error", err); return
                lims = [(p_ax, p_bx), (p_ay, p_by), (p_az, p_bz)]
                I, x, y, z, f_eval, stats = motor.simular_integral_multiple(f, lims, N, 3)
                
                escala = stats['escala']
                G = stats['desv_std']  # Desviación estándar
                EE = stats['error_est']  # Error estimado
                IC_inf = I - z_score * EE * escala  # IC inferior
                IC_sup = I + z_score * EE * escala  # IC superior
                
                # Gráfico
                self.fig_multi.clf()
                ax = self.fig_multi.add_subplot(111, projection='3d')
                limite = min(N, 2000)
                sc = ax.scatter(x[:limite], y[:limite], z[:limite], c=f_eval[:limite], cmap='plasma', s=15)
                self.fig_multi.colorbar(sc, ax=ax, label="f(x,y,z)")
                ax.set_title(f"Integral Triple (Volumen) ~= {I:.6f}")
                self.canvas_multi.draw()
                
                # Mostrar resultados estadísticos en label con formato mejorado
                resultado = f"""╔═══════════════════════════════════╗
║   INTEGRAL TRIPLE - RESULTADOS    ║
╚═══════════════════════════════════╝

Escala = {escala:.6f}

Î  = {I:.6f}
G (σ) = {G:.6f}
EE = {EE:.8f}
Z-score = {z_score:.6f}

┌─ INTERVALO DE CONFIANZA {confianza*100:.0f}% ─┐
│ Inferior:  {IC_inf:.6f}
│ Superior:  {IC_sup:.6f}
│ Ancho: {IC_sup - IC_inf:.6f}
└──────────────────────────────┘

N = {stats['N']}"""
                self.lbl_resultados_multi.config(text=resultado)
                
        except Exception as e:
            messagebox.showerror("Error", f"Verifique límites: {str(e)}")
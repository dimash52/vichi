import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

from scipy.sparse.linalg import LinearOperator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# =========================
# Problem definition
# =========================

def u_exact(x, y):
    return np.sin(x) * np.sin(y ** 2)


def f_rhs(x, y):
    return np.sin(x) * (np.sin(y) ** 2) * (0.7 + 4.4 * y * y) - 2.0 * np.sin(x) * (np.cos(y) ** 2)


def g_boundary(x, y):
    return u_exact(x, y)


# =========================
# Domain utilities
# =========================

def in_domain(i, j, n):
    return 0 <= i <= n and 0 <= j <= n and j <= i + n // 2


def is_boundary(i, j, n):
    return in_domain(i, j, n) and (
        i == 0 or i == n or j == 0 or j == n or j == i + n // 2
    )


def build_interior_nodes(n):
    nodes = []
    idx = {}
    for i in range(n + 1):
        for j in range(n + 1):
            if in_domain(i, j, n) and not is_boundary(i, j, n):
                idx[(i, j)] = len(nodes)
                nodes.append((i, j))
    return nodes, idx


def build_F(n, nodes):
    """
    Правая часть для НЕСКАЛИРОВАННОЙ матрицы A:
        A u = f + boundary/h^2
    """
    h = 1.0 / n
    h2 = h * h
    F = np.zeros(len(nodes), dtype=float)

    for k, (i, j) in enumerate(nodes):
        x, y = i * h, j * h
        val = f_rhs(x, y)

        for di, dj, c in [(1, 0, 0.7), (-1, 0, 0.7), (0, 1, 1.1), (0, -1, 1.1)]:
            ii, jj = i + di, j + dj
            if in_domain(ii, jj, n) and is_boundary(ii, jj, n):
                val += (c / h2) * g_boundary(ii * h, jj * h)

        F[k] = val

    return F


def apply_A_vec(z, n, nodes, idx):
    """
    Действие НЕСКАЛИРОВАННОЙ матрицы A:
        A = (1/h^2) * stencil
    """
    h = 1.0 / n
    inv_h2 = 1.0 / (h * h)

    y = np.zeros_like(z)
    for k, (i, j) in enumerate(nodes):
        val = 3.6 * z[k]

        for di, dj, c in [(1, 0, 0.7), (-1, 0, 0.7), (0, 1, 1.1), (0, -1, 1.1)]:
            ii, jj = i + di, j + dj
            if in_domain(ii, jj, n) and not is_boundary(ii, jj, n):
                val -= c * z[idx[(ii, jj)]]

        y[k] = inv_h2 * val

    return y


def reconstruct_grid(n, nodes, idx, z):
    h = 1.0 / n
    U = np.full((n + 1, n + 1), np.nan, dtype=float)
    for i in range(n + 1):
        for j in range(n + 1):
            if not in_domain(i, j, n):
                continue
            x, y = i * h, j * h
            if is_boundary(i, j, n):
                U[i, j] = g_boundary(x, y)
            else:
                U[i, j] = z[idx[(i, j)]]
    return U


def exact_grid(n):
    h = 1.0 / n
    U = np.full((n + 1, n + 1), np.nan, dtype=float)
    for i in range(n + 1):
        for j in range(n + 1):
            if in_domain(i, j, n):
                U[i, j] = u_exact(i * h, j * h)
    return U


def inf_error_against_exact(n, nodes, idx, z):
    U_num = reconstruct_grid(n, nodes, idx, z)
    U_ex = exact_grid(n)
    return float(np.nanmax(np.abs(U_num - U_ex)))


def build_operator(n):
    nodes, idx = build_interior_nodes(n)
    m = len(nodes)

    def mv(v):
        return apply_A_vec(v, n, nodes, idx)

    return LinearOperator((m, m), matvec=mv, dtype=float), nodes, idx


# =========================
# Iterative methods
# =========================

class _StopRequested(Exception):
    pass


def jacobi_matrix_free(n, tol=1e-10, maxiter=50000, progress_callback=None, update_every=50, stop_callback=None):
    nodes, idx = build_interior_nodes(n)
    F = build_F(n, nodes)
    h = 1.0 / n
    h2 = h * h

    u = np.zeros(len(nodes), dtype=float)
    u_new = np.zeros_like(u)

    for it in range(1, maxiter + 1):
        if stop_callback is not None and stop_callback():
            raise _StopRequested()

        for k, (i, j) in enumerate(nodes):
            sum_x = 0.0
            sum_y = 0.0

            for di, dj, c in [(1, 0, 0.7), (-1, 0, 0.7), (0, 1, 1.1), (0, -1, 1.1)]:
                ii, jj = i + di, j + dj
                if in_domain(ii, jj, n) and not is_boundary(ii, jj, n):
                    if di != 0:
                        sum_x += c * u[idx[(ii, jj)]]
                    else:
                        sum_y += c * u[idx[(ii, jj)]]

            u_new[k] = (sum_x + sum_y + h2 * F[k]) / 3.6

        step_norm = float(np.linalg.norm(u_new - u, ord=np.inf))
        if progress_callback is not None and update_every > 0 and (it == 1 or it % update_every == 0):
            progress_callback(it, u_new.copy(), step_norm, None)

        if step_norm < tol:
            return u_new, it, nodes, idx

        u[:] = u_new[:]

    return u, maxiter, nodes, idx


def minimal_residual_matrix_free(n, tol=1e-10, maxiter=50000, progress_callback=None, update_every=50, stop_callback=None):
    nodes, idx = build_interior_nodes(n)
    F = build_F(n, nodes)
    u = np.zeros(len(nodes), dtype=float)

    for it in range(1, maxiter + 1):
        if stop_callback is not None and stop_callback():
            raise _StopRequested()

        Au = apply_A_vec(u, n, nodes, idx)
        r = F - Au
        residual_norm = float(np.linalg.norm(r, ord=np.inf))
        if residual_norm < tol:
            return u, it, nodes, idx

        Ar = apply_A_vec(r, n, nodes, idx)
        denom = float(Ar @ Ar)
        if denom == 0.0:
            return u, it, nodes, idx

        tau = float(Ar @ r) / denom
        du = tau * r
        u = u + du
        step_norm = float(np.linalg.norm(du, ord=np.inf))

        if progress_callback is not None and update_every > 0 and (it == 1 or it % update_every == 0):
            progress_callback(it, u.copy(), step_norm, residual_norm)

    return u, maxiter, nodes, idx


# =========================
# Spectral estimates: power method
# =========================

def dominant_eigenpair(Aop, tol=1e-12, maxiter=20000, seed=12345, stop_callback=None, ui_pump=None):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=Aop.shape[0])
    nx = np.linalg.norm(x)
    if nx == 0.0:
        x = np.ones(Aop.shape[0], dtype=float)
        nx = np.linalg.norm(x)
    x /= nx

    lam = float(x @ (Aop @ x))

    for it in range(1, maxiter + 1):
        if stop_callback is not None and stop_callback():
            raise _StopRequested()

        y = Aop @ x
        ny = float(np.linalg.norm(y))
        if ny == 0.0:
            raise RuntimeError("Power method failed: zero vector encountered.")

        x = y / ny
        lam = float(x @ (Aop @ x))
        resid = float(np.linalg.norm((Aop @ x) - lam * x, ord=np.inf))

        if ui_pump is not None and it % 5 == 0:
            ui_pump()

        if resid <= tol:
            return lam, x, it

    return lam, x, maxiter


def power_method_max(Aop, tol=1e-12, maxiter=20000, seed=12345, stop_callback=None, ui_pump=None):
    lam, vec, it = dominant_eigenpair(
        Aop, tol=tol, maxiter=maxiter, seed=seed,
        stop_callback=stop_callback, ui_pump=ui_pump
    )
    return lam, vec, it


def apply_B_vec(v, n, nodes, idx, tau):
    return v - tau * apply_A_vec(v, n, nodes, idx)


def power_method_min_via_shift(n, tol=1e-12, maxiter=20000, seed=54321, stop_callback=None, ui_pump=None):
    nodes, idx = build_interior_nodes(n)
    m = len(nodes)

    h = 1.0 / n
    tau = (h * h) / 8.0

    def mv(v):
        return apply_B_vec(v, n, nodes, idx, tau)

    Bop = LinearOperator((m, m), matvec=mv, dtype=float)

    lam_B, vec_B, it_B = dominant_eigenpair(
        Bop, tol=tol, maxiter=maxiter, seed=seed,
        stop_callback=stop_callback, ui_pump=ui_pump
    )

    lam_min = (1.0 - lam_B) / tau
    return lam_min, vec_B, it_B


def extreme_eigenvalues(n, tol=1e-12, stop_callback=None, ui_pump=None):
    Aop, _, _ = build_operator(n)

    lam_max, vec_max, it_max = power_method_max(
        Aop, tol=tol, stop_callback=stop_callback, ui_pump=ui_pump
    )

    lam_min, vec_min, it_min = power_method_min_via_shift(
        n, tol=tol, stop_callback=stop_callback, ui_pump=ui_pump
    )

    return lam_min, lam_max, it_min, it_max


# =========================
# GUI widgets
# =========================

class MethodResultBox:
    def __init__(self, master, title):
        self.frame = ttk.LabelFrame(master, text=title, padding=8)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(1, weight=1)

        self.summary_var = tk.StringVar(value="Ожидание расчёта")
        ttk.Label(self.frame, textvariable=self.summary_var, justify="left").grid(
            row=0, column=0, sticky="ew", pady=(0, 6)
        )

        self.text = scrolledtext.ScrolledText(
            self.frame, height=12, wrap="word", font=("Consolas", 10)
        )
        self.text.grid(row=1, column=0, sticky="nsew")

    def clear(self):
        self.text.delete("1.0", "end")
        self.summary_var.set("Ожидание расчёта")

    def write(self, text):
        self.text.insert("end", text)
        self.text.see("end")

    def set_summary(self, text):
        self.summary_var.set(text)


# =========================
# Main application
# =========================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Попов Д.Р. 23121")
        self.geometry("1280x860")
        self.minsize(1160, 780)

        self.status_var = tk.StringVar(value="Готово")
        self.n_var = tk.StringVar(value="20")
        self.h_var = tk.StringVar(value="")
        self.tol_var = tk.StringVar(value="1e-6")
        self.maxiter_var = tk.StringVar(value="50000")
        self.method_var = tk.StringVar(value="Jacobi")
        self.live_update_var = tk.BooleanVar(value=True)
        self.update_every_var = tk.StringVar(value="50")

        self.solution_grid = None
        self.current_n = None
        self.current_method = None
        self.cbar = None
        self.image = None
        self._plot_shape = None

        self.stop_requested = False
        self.is_running = False

        self._setup_style()
        self._build_ui()
        self._update_h()

    def _setup_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background="#f5f6f8")
        style.configure("TLabel", background="#f5f6f8")
        style.configure("TLabelframe", background="#f5f6f8")
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("TButton", padding=6)
        style.configure("Accent.TButton", padding=8,
                        font=("Segoe UI", 10, "bold"))

    def _build_ui(self):
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=0)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)
        root.rowconfigure(1, weight=0)

        left = ttk.Frame(root)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(4, weight=1)

        right = ttk.Frame(root)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=3)
        right.rowconfigure(1, weight=2)

        params = ttk.LabelFrame(left, text="Параметры", padding=10)
        params.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        params.columnconfigure(1, weight=1)

        ttk.Label(params, text="N (чётное):").grid(
            row=0, column=0, sticky="w", pady=4)
        n_entry = ttk.Entry(params, textvariable=self.n_var, width=12)
        n_entry.grid(row=0, column=1, sticky="w", padx=5, pady=4)
        n_entry.bind("<KeyRelease>", lambda e: self._update_h())
        n_entry.bind("<FocusOut>", lambda e: self._update_h())

        ttk.Label(params, text="h = 1/n:").grid(row=1,
                                                column=0, sticky="w", pady=4)
        ttk.Entry(params, textvariable=self.h_var, width=12, state="readonly").grid(
            row=1, column=1, sticky="w", padx=5, pady=4
        )

        ttk.Label(params, text="Epsilon:").grid(
            row=2, column=0, sticky="w", pady=4)
        ttk.Entry(params, textvariable=self.tol_var, width=12).grid(
            row=2, column=1, sticky="w", padx=5, pady=4
        )

        ttk.Label(params, text="MaxIter:").grid(
            row=3, column=0, sticky="w", pady=4)
        ttk.Entry(params, textvariable=self.maxiter_var, width=12).grid(
            row=3, column=1, sticky="w", padx=5, pady=4
        )

        ttk.Label(params, text="Метод:").grid(
            row=4, column=0, sticky="w", pady=4)
        ttk.Radiobutton(params, text="Jacobi", variable=self.method_var, value="Jacobi").grid(
            row=4, column=1, sticky="w", padx=5, pady=2
        )
        ttk.Radiobutton(params, text="MR", variable=self.method_var, value="MR").grid(
            row=5, column=1, sticky="w", padx=5, pady=2
        )

        live = ttk.LabelFrame(left, text="Live-обновление", padding=10)
        live.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        live.columnconfigure(1, weight=1)

        ttk.Checkbutton(live, text="Обновлять график каждые", variable=self.live_update_var).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Entry(live, textvariable=self.update_every_var, width=10).grid(
            row=0, column=1, sticky="w", padx=5
        )
        ttk.Label(live, text="итераций").grid(row=0, column=2, sticky="w")

        actions = ttk.LabelFrame(left, text="Действия", padding=10)
        actions.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        self.btn_solve = ttk.Button(
            actions, text="Решить", style="Accent.TButton", command=self.solve)
        self.btn_solve.pack(fill="x", pady=4)

        self.btn_stop = ttk.Button(
            actions, text="Стоп", command=self.request_stop, state="disabled")
        self.btn_stop.pack(fill="x", pady=4)

        self.btn_eigs = ttk.Button(
            actions, text="Найти λmin и λmax", command=self.compute_eigs)
        self.btn_eigs.pack(fill="x", pady=4)

        self.btn_cond = ttk.Button(
            actions, text="Оценить cond(A)", command=self.compute_condition_number)
        self.btn_cond.pack(fill="x", pady=4)

        self.btn_clear = ttk.Button(
            actions, text="Очистить", command=self.clear_all)
        self.btn_clear.pack(fill="x", pady=4)

        info = ttk.LabelFrame(left, text="Информация", padding=10)
        info.grid(row=3, column=0, sticky="ew")
        self.info_var = tk.StringVar(
            value="Выберите метод и нажмите 'Решить'.")
        ttk.Label(info, textvariable=self.info_var,
                  wraplength=250, justify="left").pack(fill="x")

        log_frame = ttk.LabelFrame(left, text="Журнал", padding=8)
        log_frame.grid(row=4, column=0, sticky="nsew", pady=(10, 0))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.main_log = scrolledtext.ScrolledText(
            log_frame, height=12, wrap="word", font=("Consolas", 10))
        self.main_log.grid(row=0, column=0, sticky="nsew")

        plot_frame = ttk.LabelFrame(right, text="Поле решения", padding=6)
        plot_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(8.4, 5.8), dpi=100)
        self.ax = self.fig.add_axes([0.08, 0.10, 0.74, 0.84])
        self.cbar_ax = self.fig.add_axes([0.86, 0.10, 0.03, 0.84])
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        results = ttk.Frame(right)
        results.grid(row=1, column=0, sticky="nsew")
        results.columnconfigure(0, weight=1)
        results.columnconfigure(1, weight=1)
        results.rowconfigure(0, weight=1)

        self.box_jacobi = MethodResultBox(results, "Jacobi")
        self.box_jacobi.frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        self.box_mr = MethodResultBox(results, "MR")
        self.box_mr.frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        status = ttk.Label(root, textvariable=self.status_var,
                           anchor="w", relief="sunken")
        status.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        self._write_main("Готово. Задайте параметры и нажмите 'Решить'.")

    def _set_running(self, running):
        self.is_running = running
        if running:
            self.btn_solve.config(state="disabled")
            self.btn_eigs.config(state="disabled")
            self.btn_cond.config(state="disabled")
            self.btn_clear.config(state="disabled")
            self.btn_stop.config(state="normal")
        else:
            self.btn_solve.config(state="normal")
            self.btn_eigs.config(state="normal")
            self.btn_cond.config(state="normal")
            self.btn_clear.config(state="normal")
            self.btn_stop.config(state="disabled")
        self.update_idletasks()

    def _set_status(self, text):
        self.status_var.set(text)
        self.update_idletasks()

    def _pump_ui(self):
        try:
            self.update_idletasks()
            self.update()
        except tk.TclError:
            pass

    def _write_main(self, text):
        self.main_log.insert("end", text + "\n")
        self.main_log.see("end")

    def _update_h(self):
        try:
            n = int(self.n_var.get())
            self.h_var.set(f"{1.0 / n:.6g}" if n > 0 else "")
        except ValueError:
            self.h_var.set("")

    def _read_params(self):
        try:
            n = int(self.n_var.get())
            tol = float(self.tol_var.get())
            maxiter = int(self.maxiter_var.get())
            update_every = int(self.update_every_var.get())
        except ValueError:
            raise ValueError("Проверь N, eps, maxiter и шаг обновления.")

        if n < 2 or n % 2 != 0:
            raise ValueError("N должно быть чётным и больше 2.")
        if update_every <= 0:
            raise ValueError(
                "Число итераций обновления должно быть положительным.")

        h = 1.0 / n
        self.h_var.set(f"{h:.6g}")
        return n, tol, maxiter, update_every, h

    def request_stop(self):
        self.stop_requested = True
        self._set_status("Запрошена остановка...")

    def clear_all(self):
        if self.is_running:
            self.request_stop()
            return
        self.main_log.delete("1.0", "end")
        self.box_jacobi.clear()
        self.box_mr.clear()
        self.ax.clear()
        self.cbar_ax.clear()
        self.image = None
        self._plot_shape = None
        self.cbar = None
        self.canvas.draw()
        self.info_var.set("Выберите метод и нажмите 'Решить'.")
        self._set_status("Готово")

    def _render_solution(self, U, n, method_name, iteration=None):
        masked = np.ma.masked_invalid(U)
        data = masked.T
        shape = data.shape

        if self.image is None or self._plot_shape != shape:
            self.ax.clear()
            self.cbar_ax.clear()
            self.image = self.ax.imshow(
                data,
                origin="lower",
                extent=[0, 1, 0, 1],
                aspect="auto",
                interpolation="nearest",
            )
            self._plot_shape = shape
            self.cbar = self.fig.colorbar(self.image, cax=self.cbar_ax)
        else:
            self.image.set_data(data)
            finite = np.asarray(data)[np.isfinite(data)]
            if finite.size > 0:
                self.image.set_clim(vmin=float(np.min(finite)),
                                    vmax=float(np.max(finite)))
            if self.cbar is not None:
                self.cbar.update_normal(self.image)

        title = f"Приближённое решение u(x,y), {method_name}, N={n}"
        if iteration is not None:
            title += f", итерация {iteration}"
        self.ax.set_title(title)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.grid(True, alpha=0.25)
        self.canvas.draw_idle()
        self._pump_ui()

    def _result_box(self, method):
        return self.box_jacobi if method == "Jacobi" else self.box_mr

    def solve(self):
        if self.is_running:
            return

        try:
            n, tol, maxiter, update_every, h = self._read_params()
            method = self.method_var.get()
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            return

        self.stop_requested = False
        self._set_running(True)

        box = self._result_box(method)
        box.clear()
        self._set_status("Вычисление решения...")
        self._write_main(
            f"[{method}] n={n}, h={h:.6g}, eps={tol}, maxiter={maxiter}, update_every={update_every}"
        )

        live = bool(self.live_update_var.get())
        if n > 100:
            live = False

        try:
            if method == "Jacobi":
                nodes_local, idx_local = build_interior_nodes(n)

                def progress(it, z_current, step_norm, residual_norm):
                    if not live:
                        return
                    err = inf_error_against_exact(
                        n, nodes_local, idx_local, z_current)
                    box.set_summary(
                        f"Итерация: {it}, ||Δu||∞ = {step_norm:.6e}")
                    box.write(
                        f"it = {it}, ||Δu||∞ = {step_norm:.6e}\n")
                    self._render_solution(reconstruct_grid(
                        n, nodes_local, idx_local, z_current), n, method, it)
                    self._set_status(f"Jacobi: итерация {it}")
                    if self.stop_requested:
                        raise _StopRequested()

                z, iters, nodes, idx = jacobi_matrix_free(
                    n,
                    tol=tol,
                    maxiter=maxiter,
                    progress_callback=progress,
                    update_every=update_every,
                    stop_callback=lambda: self.stop_requested,
                )
                err = inf_error_against_exact(n, nodes, idx, z)
                U = reconstruct_grid(n, nodes, idx, z)
                self._render_solution(U, n, method, iters)
                box.set_summary(f"Готово | итераций: {iters}")
                box.write(f"Итог: итераций = {iters}\n")
                self._write_main(
                    f"[Jacobi] итераций = {iters}")
                self.info_var.set(f"Jacobi завершён. N={n}, h={h:.6g}.")
                self._set_status("Jacobi завершён")

            elif method == "MR":
                nodes_local, idx_local = build_interior_nodes(n)

                def progress(it, z_current, step_norm, residual_norm):
                    if not live:
                        return
                    err = inf_error_against_exact(
                        n, nodes_local, idx_local, z_current)
                    if residual_norm is None:
                        box.set_summary(
                            f"Итерация: {it}, ||Δu||∞ = {step_norm:.6e}")
                        box.write(
                            f"it = {it}, ||Δu||∞ = {step_norm:.6e}\n")
                    else:
                        box.set_summary(
                            f"Итерация: {it}, ||r||∞ = {residual_norm:.6e}")
                        box.write(
                            f"it = {it}, ||r||∞ = {residual_norm:.6e}\n")
                    self._render_solution(reconstruct_grid(
                        n, nodes_local, idx_local, z_current), n, method, it)
                    self._set_status(f"MR: итерация {it}")
                    if self.stop_requested:
                        raise _StopRequested()

                z, iters, nodes, idx = minimal_residual_matrix_free(
                    n,
                    tol=tol,
                    maxiter=maxiter,
                    progress_callback=progress,
                    update_every=update_every,
                    stop_callback=lambda: self.stop_requested,
                )
                err = inf_error_against_exact(n, nodes, idx, z)
                U = reconstruct_grid(n, nodes, idx, z)
                self._render_solution(U, n, method, iters)
                box.set_summary(f"Готово | итераций: {iters}")
                box.write(f"Итог: итераций = {iters}\n")
                self._write_main(
                    f"[MR] итераций = {iters}")
                self.info_var.set(f"MR завершён. N={n}, h={h:.6g}.")
                self._set_status("MR завершён")

            else:
                messagebox.showerror("Ошибка", "Неизвестный метод.")
                return

        except _StopRequested:
            self._write_main("Остановлено пользователем.")
            self._set_status("Остановлено")
            self.info_var.set("Вычисление остановлено пользователем.")
        except Exception as e:
            self._set_status("Ошибка")
            messagebox.showerror("Ошибка вычисления", str(e))
        finally:
            self._set_running(False)

    def compute_eigs(self):
        if self.is_running:
            return

        try:
            n, tol, maxiter, update_every, _ = self._read_params()
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            return

        self.stop_requested = False
        self._set_running(True)
        self._set_status("Вычисление собственных чисел (степенной метод)...")
        self._write_main("Вычисляю λmin и λmax степенным методом...")

        try:
            lam_min, lam_max, it_min, it_max = extreme_eigenvalues(
                n,
                tol=min(tol, 1e-12),
                stop_callback=lambda: self.stop_requested,
                ui_pump=self._pump_ui,
            )
            self._write_main(
                f"lambda_min = {lam_min:.12f} (итераций: {it_min})")
            self._write_main(
                f"lambda_max = {lam_max:.12f} (итераций: {it_max})")
            self._write_main(f"cond(A) ≈ {abs(lam_max / lam_min):.12f}")
            self._set_status("Собственные числа найдены")
        except _StopRequested:
            self._write_main(
                "Вычисление собственных чисел остановлено пользователем.")
            self._set_status("Остановлено")
        except Exception as e:
            self._set_status("Ошибка")
            messagebox.showerror("Ошибка eigenvalues", str(e))
        finally:
            self._set_running(False)

    def compute_condition_number(self):
        if self.is_running:
            return

        try:
            n, tol, maxiter, update_every, _ = self._read_params()
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            return

        self.stop_requested = False
        self._set_running(True)
        self._set_status("Оценка cond(A)...")
        self._write_main("Оцениваю cond(A)...")

        try:
            lam_min, lam_max, it_min, it_max = extreme_eigenvalues(
                n,
                tol=min(tol, 1e-12),
                stop_callback=lambda: self.stop_requested,
                ui_pump=self._pump_ui,
            )
            if abs(lam_min) < 1e-15:
                raise ZeroDivisionError(
                    "lambda_min слишком близко к нулю, cond(A) не вычислить.")
            kappa = abs(lam_max / lam_min)
            self._write_main(f"cond(A) ≈ {kappa:.12f}")
            self._set_status("cond(A) оценено")
        except _StopRequested:
            self._write_main("Оценка cond(A) остановлена пользователем.")
            self._set_status("Остановлено")
        except Exception as e:
            self._set_status("Ошибка")
            messagebox.showerror("Ошибка cond(A)", str(e))
        finally:
            self._set_running(False)


if __name__ == "__main__":
    app = App()
    app.mainloop()

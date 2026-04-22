import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw

import threading
import time
import subprocess
import glob

import einops
import huggingface_hub
import matplotlib
import numpy
import pandas
import pydicom
import scipy
import skimage
import smcore
import sv_ttk
import timm
import torch
import torchvision

## BEGIN UI
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 900

WHITE = "#ffffff"
TEXT_BLACK = "#111111"
TEXT_GRAY = "#6b6b6b"
PURPLE_LIGHT = "#d6b8ff"
PURPLE = "#6f4bdc"
PURPLE_HOVER = "#8c66ea"
PURPLE_BORDER = "#cfb0ff"
DARK_PANEL = "#232323"
DARK_TEXT = "#f2f2f2"

TITLE = "AS-OCT Auto-Seg"
SUBTITLE = "Powered by SimpleMind AI"
EYE_LOGO_PATH = './assets/asoct_logo.png'
SM_LOGO_PATH = './assets/sm_logo.png'

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")

## END UI

## CONFIG
SIMPLEMIND_DIR = "./" # DELETE AFTER CONFIGURING DYNAMIC FOLDER OPTIONS
PLAN           = "plans/oct"
CSV_FILE       = "data/oct_images.csv"
BB_ADDR        = "127.0.0.1:8080"
GPU_NUM        = "0"

## PLACEHOLDERS
image_main_structure_segmentations = [
    './asoct_test/Main structures segmentations/0010_HVS002_OS_Scan_3_Dewarped.png',
    './asoct_test/Main structures segmentations/0010_HVS002_OS_Scan_3_Dewarped_cornea.png',
    './asoct_test/Main structures segmentations/0010_HVS002_OS_Scan_3_Dewarped_iris.png',
    './asoct_test/Main structures segmentations/0010_HVS002_OS_Scan_3_Dewarped_lens.png',
]
image_measurements = [
    './asoct_test/Measurements/0001_HVS001_OS_Scan_0_Dewarped_ACD_-_Anterior_Chamber_Depth.png',
    './asoct_test/Measurements/0001_HVS001_OS_Scan_0_Dewarped_ACW_-_Anterior_Chamber_Width.png',
    './asoct_test/Measurements/0001_HVS001_OS_Scan_0_Dewarped_AOD_Measurements.png',
    './asoct_test/Measurements/0001_HVS001_OS_Scan_0_Dewarped_Iris_Area_Measurements.png',
    './asoct_test/Measurements/0001_HVS001_OS_Scan_0_Dewarped_Iris_Curvature_Measurements.png',
    './asoct_test/Measurements/0001_HVS001_OS_Scan_0_Dewarped_IT750_-_Iris_Thickness_at_750µm.png',
    './asoct_test/Measurements/0001_HVS001_OS_Scan_0_Dewarped_LT_-_Lens_Thickness.png',
    './asoct_test/Measurements/0001_HVS001_OS_Scan_0_Dewarped_LV_-_Lens_Vault.png',
    './asoct_test/Measurements/0001_HVS001_OS_Scan_0_Dewarped_Pupil_Diameter_Measurement.png',
    './asoct_test/Measurements/0001_HVS001_OS_Scan_0_Dewarped_SSA_Measurements.png',
    './asoct_test/Measurements/0001_HVS001_OS_Scan_0_Dewarped_TISA_Measurements.png',
]
image_scleral_spur_segmentations = [
    './asoct_test/Scleral spur segmentations/input_image.png',
    './asoct_test/Scleral spur segmentations/left_half_image.png',
    './asoct_test/Scleral spur segmentations/left_half_image-image_crop_annotation.png',
    './asoct_test/Scleral spur segmentations/left_half_image-image_half_left.png',
    './asoct_test/Scleral spur segmentations/right_half_image.png',
    './asoct_test/Scleral spur segmentations/right_half_image-image_half_right.png',
    './asoct_test/Scleral spur segmentations/ss_detection_left_half_image.png',
    './asoct_test/Scleral spur segmentations/ss_detection_left_half_image-image_scleral_spur.png',
    './asoct_test/Scleral spur segmentations/ss_detection_left_half_image-neural_net-ss.png',
]
path_main_structure_segmentations = './asoct_test/Main structures segmentations/'
path_measurements = './asoct_test/Measurements/'
path_scleral_spur_segmentations = './asoct_test/Scleral spur segmentations/'

def load_logo(path: str, size: int) -> ImageTk.PhotoImage:
    img = Image.open(path).convert("RGBA")
    img = img.resize((size, size), Image.LANCZOS)
    return ImageTk.PhotoImage(img)


def make_placeholder_thumbnail(size=(152, 94), label="") -> ImageTk.PhotoImage:
    width, height = size
    img = Image.new("RGBA", (width, height), (245, 245, 245, 255))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((2, 2, width - 2, height - 2), radius=8, outline=(220, 220, 220, 255), width=2)
    draw.rectangle((width * 0.20, height * 0.34, width * 0.78, height * 0.38), fill=(145, 145, 145, 255))
    draw.polygon(
        [(width * 0.22, height * 0.58), (width * 0.38, height * 0.38), (width * 0.52, height * 0.58)],
        fill=(160, 160, 160, 255),
    )
    draw.polygon(
        [(width * 0.48, height * 0.58), (width * 0.60, height * 0.44), (width * 0.74, height * 0.58)],
        fill=(120, 120, 120, 255),
    )
    if label:
        draw.text((8, height - 16), label[:14], fill=(110, 110, 110, 255))
    return ImageTk.PhotoImage(img)


def load_thumbnail_from_file(path: str, size=(152, 94)) -> ImageTk.PhotoImage:
    try:
        img = Image.open(path).convert("RGBA")
        img.thumbnail(size, Image.LANCZOS)
        return ImageTk.PhotoImage(img)
    except Exception:
        return make_placeholder_thumbnail(size=size, label=os.path.basename(path).split(".")[0])


def list_images_in_folder(folder: str):
    images = []
    if not folder or not os.path.isdir(folder):
        return images
    try:
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and name.lower().endswith(IMAGE_EXTS):
                images.append(path)
    except OSError:
        pass
    return images


def scan_study_root(root_folder: str):
    """Return [(folder_name, folder_path, [images...]), ...].

    Uses subfolders that contain images. If none are found, falls back to the root folder itself.
    """
    groups = []
    if not root_folder or not os.path.isdir(root_folder):
        return groups

    try:
        entries = sorted(os.listdir(root_folder))
    except OSError:
        entries = []

    for entry in entries:
        path = os.path.join(root_folder, entry)
        if os.path.isdir(path):
            imgs = list_images_in_folder(path)
    if imgs:
        groups.append(("Input and Preprocessing", path, imgs))
        # change this to the actual directory(hardcoded)
        groups.append(("Segmentations: Main Structures", path_main_structure_segmentations, image_main_structure_segmentations))
        groups.append(("Segmentations: Scleral Spur", path_scleral_spur_segmentations, image_scleral_spur_segmentations))
        groups.append(("Measurements and Annotations", path_measurements, image_measurements))

    if groups:
        return groups

    root_imgs = list_images_in_folder(root_folder)
    if root_imgs:
        base = os.path.basename(root_folder.rstrip(os.sep)) or "Selected Study"
        return [(base, root_folder, root_imgs)]

    return []


class PillButton(tk.Canvas):
    def __init__(self, master, text, command=None, width=220, height=52,
                 fill=PURPLE_LIGHT, hover_fill=PURPLE_HOVER):
        super().__init__(master, width=width, height=height, highlightthickness=0, bg=WHITE)
        self.command = command
        self.text = text
        self.width = width
        self.height = height
        self.normal_fill = fill
        self.hover_fill = hover_fill
        self._enabled = True
        self.configure(cursor="hand2")
        self._draw(self.normal_fill)
        self.bind("<Button-1>", self._clicked)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def _round_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [
            x1 + r, y1,
            x2 - r, y1,
            x2, y1,
            x2, y1 + r,
            x2, y2 - r,
            x2, y2,
            x2 - r, y2,
            x1 + r, y2,
            x1, y2,
            x1, y2 - r,
            x1, y1 + r,
            x1, y1,
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        if enabled:
            self.configure(cursor="hand2")
            self._draw(self.normal_fill)
        else:
            self.configure(cursor="")
            self._draw("#e0e0e0")  # greyed-out fill

    def _draw(self, fill):
        self.delete("all")
        self._round_rect(2, 2, self.width - 2, self.height - 2, 22, fill=fill, outline=fill)
        text_color = "#aaaaaa" if not self._enabled else TEXT_BLACK
        self.create_text(
            self.width // 2,
            self.height // 2,
            text=self.text,
            fill=text_color,
            font=("Helvetica", 17, "normal"),
        )

    def _clicked(self, _event):
        if self._enabled and callable(self.command):
            self.command()

    def _on_enter(self, _event):
        if self._enabled:
            self._draw(self.hover_fill)

    def _on_leave(self, _event):
        self._draw(self.normal_fill if self._enabled else "#e0e0e0")


class TileButton(tk.Frame):
    def __init__(self, master, title, on_click=None, image=None, value_text=None, selected=False):
        super().__init__(master, bg=WHITE)
        self.title = title
        self.on_click = on_click
        self.image = image
        self.value_text = value_text
        self.selected = selected
        self.icon_ref = None
        self.thumb_ref = None
        self._build()

    def _build(self):
        self.configure(cursor="hand2")
        self.bind("<Button-1>", self._clicked)

        if self.value_text is None:
            # Folder tile
            self.icon_ref = self._make_folder_icon(64, selected=self.selected)
            icon_label = tk.Label(self, image=self.icon_ref, bg=WHITE, borderwidth=0)
            icon_label.pack()
            icon_label.bind("<Button-1>", self._clicked)

            title_label = tk.Label(
                self,
                text=self.title,
                bg=WHITE,
                fg=TEXT_BLACK,
                font=("Helvetica", 10, "normal"),
                justify="center",
            )
            title_label.pack(pady=(4, 0))
            title_label.bind("<Button-1>", self._clicked)
        else:
            # File / measurement tile
            frame = tk.Frame(self, bg=WHITE)
            frame.pack()
            self.thumb_ref = self.image if self.image is not None else make_placeholder_thumbnail()
            thumb = tk.Label(frame, image=self.thumb_ref, bg=WHITE, borderwidth=0)
            thumb.pack()
            thumb.bind("<Button-1>", self._clicked)
            if self.selected:
                thumb.configure(highlightthickness=3, highlightbackground=PURPLE_BORDER)

            title_label = tk.Label(
                self,
                text=self.title,
                bg=WHITE,
                fg=TEXT_BLACK,
                font=("Helvetica", 10, "normal"),
                justify="center",
            )
            title_label.pack(pady=(8, 0))
            title_label.bind("<Button-1>", self._clicked)

            value_label = tk.Label(
                self,
                text=self.value_text,
                bg=WHITE,
                fg=TEXT_BLACK,
                font=("Helvetica", 11, "bold"),
                justify="center",
            )
            value_label.pack(pady=(3, 0))
            value_label.bind("<Button-1>", self._clicked)

    def _make_folder_icon(self, size=64, selected=False):
        bg = (240, 230, 255, 255) if selected else (255, 255, 255, 255)
        outline = (111, 75, 220, 255)
        fill = (111, 75, 220, 255)

        img = Image.new("RGBA", (size, size), bg)
        draw = ImageDraw.Draw(img)

        if selected:
            draw.rounded_rectangle((2, 2, size - 2, size - 2), radius=12, fill=(230, 214, 255, 255))

        draw.rounded_rectangle((10, 24, size - 10, size - 14), radius=8, outline=outline, width=5)
        draw.rectangle((10, 24, size - 26, 34), fill=bg)
        draw.polygon([(12, 24), (26, 24), (32, 18), (48, 18), (48, 24)], fill=fill)

        return ImageTk.PhotoImage(img)

    def _clicked(self, _event):
        if callable(self.on_click):
            self.on_click()


class ASOCTApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(TITLE)
        self.configure(bg=WHITE)
        # Remove the maximize button entirely (cross-platform)
        self.resizable(False, False)
        try:
            # Windows: strip WS_MAXIMIZEBOX from the window style via ctypes
            import ctypes
            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
            GWL_STYLE    = -16
            WS_MAXIMIZEBOX = 0x00010000
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_STYLE)
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_STYLE, style & ~WS_MAXIMIZEBOX)
        except Exception:
            pass  # Non-Windows platforms: resizable(False,False) already hides/disables it
        # Open maximized / full-screen on all platforms
        try:
            self.state("zoomed")               # Windows & some Linux WMs
        except tk.TclError:
            try:
                self.attributes("-zoomed", True)   # Linux (GNOME / KDE)
            except tk.TclError:
                self.attributes("-fullscreen", True)  # macOS fallback

        self.eye_logo = None
        self.sm_logo = None
        self.selected_root_folder = None
        self.selected_folder_label = tk.StringVar(value="No folder selected yet")

        self.results_mode = "library"
        self.folder_groups = []
        self.selected_folder_index = 0
        self.selected_image_index = 0
        self.preview_ref = None

        # Hardcoded folder directories
        self._plan_var = tk.StringVar(value=PLAN)
        self._csv_var  = tk.StringVar(value=CSV_FILE)
        self._addr_var = tk.StringVar(value=BB_ADDR)
        self._gpu_var  = tk.StringVar(value=GPU_NUM)

        # State
        self._bb_proc         = None
        self._plan_proc       = None
        self._observer        = None
        self._images          = []      # list of (timestamp, filepath)
        self._slide_idx       = 0
        self._slide_job       = None
        self._paused          = False
        self._waiting_for_new = False
        self._wait_start      = 0.0
        self._last_count      = 0
        self._running         = False
        self._working_dir     = None

        self._build_ui()

    def _build_ui(self):
        self.container = tk.Frame(self, bg=WHITE)
        self.container.pack(fill="both", expand=True)

        self.home_screen = tk.Frame(self.container, bg=WHITE)
        self.results_screen = tk.Frame(self.container, bg=WHITE)
        self.home_screen.place(relwidth=1, relheight=1)
        self.results_screen.place(relwidth=1, relheight=1)

        self._build_home_screen()
        self._build_results_screen()
        self._show_home()

    def _build_home_screen(self):
        center = tk.Frame(self.home_screen, bg=WHITE)
        center.place(relx=0.5, rely=0.5, anchor="center")

        self.eye_logo = load_logo(EYE_LOGO_PATH, 170)
        tk.Label(center, image=self.eye_logo, bg=WHITE, borderwidth=0, highlightthickness=0).pack(pady=(0, 12))

        tk.Label(
            center,
            text=TITLE,
            bg=WHITE,
            fg=TEXT_BLACK,
            font=("Helvetica", 46, "normal"),
        ).pack(pady=(22, 14))

        subtitle_row = tk.Frame(center, bg=WHITE)
        subtitle_row.pack(pady=(0, 20))
        tk.Label(
            subtitle_row,
            text=SUBTITLE,
            bg=WHITE,
            fg=TEXT_BLACK,
            font=("Helvetica", 18, "normal"),
        ).pack(side="left")

        self.sm_logo = load_logo(SM_LOGO_PATH, 34)
        tk.Label(subtitle_row, image=self.sm_logo, bg=WHITE, borderwidth=0, highlightthickness=0).pack(side="left", padx=(10, 0))

        tk.Label(
            center,
            textvariable=self.selected_folder_label,
            bg=WHITE,
            fg=TEXT_GRAY,
            font=("Helvetica", 11, "normal"),
            wraplength=520,
            justify="center",
        ).pack(pady=(0, 28))

        btn_row = tk.Frame(center, bg=WHITE)
        btn_row.pack(pady=(0, 12))
        self.run_btn = PillButton(btn_row, "Run Analysis", command=self._on_run_analysis, width=250, height=64)
        self.run_btn.pack(side="left", padx=16)
        self.load_btn = PillButton(btn_row, "Load Study", command=self._on_load_study, width=250, height=64)
        self.load_btn.pack(side="left", padx=16)
        self.view_btn = PillButton(btn_row, "View Results", command=self._show_results, width=250, height=64)
        self.view_btn.pack(side="left", padx=16)
        self.view_btn.set_enabled(False)  # disabled until analysis completes

        # Status label shown while pipeline is running
        self.status_label = tk.Label(
            center, text="", bg=WHITE, fg=PURPLE,
            font=("Helvetica", 12, "italic")
        )
        self.status_label.pack(pady=(4, 0))

    def _build_results_screen(self):
        top = tk.Frame(self.results_screen, bg=WHITE)
        top.pack(fill="x", padx=18, pady=(14, 8))

        back_btn = tk.Label(top, text="Home", fg=PURPLE, bg=WHITE, font=("Helvetica", 30), cursor="hand2")
        back_btn.pack(side="left")
        back_btn.bind("<Button-1>", lambda _e: self._show_home())

        tabs = tk.Frame(top, bg=WHITE)
        tabs.pack(side="left", padx=(8, 0))

        self.library_tab = tk.Canvas(tabs, width=220, height=44, bg=WHITE, highlightthickness=0)
        self.library_tab.pack(side="left")
        self.measurement_tab = tk.Canvas(tabs, width=220, height=44, bg=WHITE, highlightthickness=0)
        self.measurement_tab.pack(side="left")

        self.library_tab.bind("<Button-1>", lambda _e: self._set_mode("library"))
        self.measurement_tab.bind("<Button-1>", lambda _e: self._set_mode("measurements"))

        # ── LAYOUT FIX ──────────────────────────────────────────────────────────
        # Use a PanedWindow (or simply two equal frames) so the right panel
        # always occupies exactly half of the 1000 px window width.
        # Left panel: scrollable content area  (~478 px usable after padding)
        # Right panel: dark preview panel      (~478 px usable after padding)
        # Total: 478 + 4(left padx) + 4(right padx) ≈ 1000 px
        body = tk.Frame(self.results_screen, bg=WHITE)
        body.pack(fill="both", expand=True, padx=0, pady=0)
        body.grid_columnconfigure(0, weight=1, uniform="half")
        body.grid_columnconfigure(1, weight=1, uniform="half")
        body.grid_rowconfigure(0, weight=1)

        self.left_panel = tk.Frame(body, bg=WHITE)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(18, 6), pady=(0, 14))

        self.right_panel = tk.Frame(body, bg=DARK_PANEL)
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 18), pady=(0, 14))
        self.right_panel.pack_propagate(False)
        # ────────────────────────────────────────────────────────────────────────

        self._build_left_content()
        self._build_right_content()
        self._render_tabs()
        self._render_results()

    def _build_left_content(self):
        self.library_view = tk.Frame(self.left_panel, bg=WHITE)
        self.measurements_view = tk.Frame(self.left_panel, bg=WHITE)
        self.library_view.pack(fill="both", expand=True)
        self.measurements_view.pack_forget()

        self.folder_grid = tk.Frame(self.library_view, bg=WHITE)
        self.folder_grid.pack(anchor="nw", pady=(8, 8), fill="x")

        self.file_grid = tk.Frame(self.library_view, bg=WHITE)
        self.file_grid.pack(anchor="nw", pady=(0, 8), fill="x")

        self.measurement_file_grid = tk.Frame(self.measurements_view, bg=WHITE)
        self.measurement_file_grid.pack(anchor="nw", pady=(8, 8), fill="x")

        # initial render happens after the right panel is built

    def _build_right_content(self):
        self.preview_outer = tk.Frame(self.right_panel, bg=DARK_PANEL)
        self.preview_outer.pack(fill="both", expand=True, padx=12, pady=12)

        # Canvas fills the available width; height is driven by the panel's
        # actual pixel height so it is never truncated.
        self.preview_canvas = tk.Canvas(
            self.preview_outer, bg=WHITE, highlightthickness=0
        )
        self.preview_canvas.pack(fill="both", expand=True, padx=4, pady=(0, 12))

        tool_row = tk.Frame(self.preview_outer, bg=DARK_PANEL)
        tool_row.pack(pady=(0, 10))
        tool_style = dict(bg=PURPLE_LIGHT, fg=TEXT_BLACK, bd=0, relief="flat", font=("Helvetica", 18, "bold"), cursor="hand2")
        tk.Button(tool_row, text="<", width=2, height=1, command=self._prev_item, **tool_style).pack(side="left", padx=8)
        tk.Button(tool_row, text="Zoom", width=2, height=1, command=self._zoom_item, **tool_style).pack(side="left", padx=8)
        tk.Button(tool_row, text="Pan", width=2, height=1, command=self._pan_item, **tool_style).pack(side="left", padx=8)
        tk.Button(tool_row, text=">", width=2, height=1, command=self._next_item, **tool_style).pack(side="left", padx=8)

        self.preview_mode_label = tk.Label(self.preview_outer, text="", bg=DARK_PANEL, fg=DARK_TEXT, font=("Helvetica", 12, "bold"))
        self.preview_mode_label.pack(pady=(6, 4))

        self.preview_title = tk.Label(self.preview_outer, text="", bg=DARK_PANEL, fg=DARK_TEXT, font=("Helvetica", 12, "bold"))
        self.preview_title.pack(pady=(0, 4))

        self.preview_detail = tk.Label(self.preview_outer, text="", bg=DARK_PANEL, fg=DARK_TEXT, font=("Helvetica", 10), justify="center")
        self.preview_detail.pack(pady=(0, 4))

    def _render_tabs(self):
        self.library_tab.delete("all")
        self.measurement_tab.delete("all")

        left_fill = PURPLE if self.results_mode == "library" else PURPLE_LIGHT
        left_fg = WHITE if self.results_mode == "library" else TEXT_BLACK
        right_fill = PURPLE if self.results_mode == "measurements" else PURPLE_LIGHT
        right_fg = WHITE if self.results_mode == "measurements" else TEXT_BLACK

        self._round_tab(self.library_tab, 0, 0, 220, 40, 18, fill=left_fill)
        self.library_tab.create_text(110, 20, text="Image Library", fill=left_fg, font=("Helvetica", 13, "bold" if self.results_mode == "library" else "normal"))

        self._round_tab(self.measurement_tab, 0, 0, 220, 40, 18, fill=right_fill)
        self.measurement_tab.create_text(110, 20, text="Measurements", fill=right_fg, font=("Helvetica", 13, "bold" if self.results_mode == "measurements" else "normal"))

    def _round_tab(self, canvas, x1, y1, x2, y2, r, fill):
        points = [
            x1 + r, y1,
            x2 - r, y1,
            x2, y1,
            x2, y1 + r,
            x2, y2 - r,
            x2, y2,
            x2 - r, y2,
            x1 + r, y2,
            x1, y2,
            x1, y2 - r,
            x1, y1 + r,
            x1, y1,
        ]
        canvas.create_polygon(points, smooth=True, fill=fill, outline=fill)

    def _show_home(self):
        self.home_screen.lift()
        self.title(TITLE)

    def _show_results(self):
        if not self.selected_root_folder:
            messagebox.showwarning("No folder selected", "Please click Run Analysis and choose a folder first.")
            return
        print(self.selected_root_folder)
        # Append the working directory with the output folder for the image output
        self.tester = self.selected_root_folder.get().strip()
        print(f"tester: {self.tester}")
        self.pattern_directory = os.path.join(self._working_dir, "output_*/oct")
        print(f"pattern: {self.pattern_directory}")
        self.output_directory = glob.glob(self.pattern_directory)[0]
        print(f"output: {self.output_directory}")

        self.folder_groups = scan_study_root(self.output_directory)
        if not self.folder_groups:
            messagebox.showwarning("No images found", "The selected folder does not contain any image files.")
            return

        self.selected_folder_index = 0
        self.selected_image_index = 0
        self.results_mode = "library"
        self.results_screen.lift()
        self.title("AS-OCT Auto-Seg Results")
        self._render_results()

    def _on_run_analysis(self):
        folder = filedialog.askdirectory(title="Select Study Folder")
        if folder:
            self.selected_root_folder = tk.StringVar(value=folder)
            self.selected_folder_label.set(f"Selected folder: {folder}")            
        else:
            self.selected_root_folder = None
            self.selected_folder_label.set("No folder selected yet")

        ### Start pipeline
        self._start_pipeline()

    def _on_load_study(self):
            folder = filedialog.askdirectory(title="Select Processed Study Folder")
            if not folder:
                return
            self.selected_root_folder = tk.StringVar(value=folder)
            self._working_dir = folder
            self.selected_folder_label.set(f"Selected folder: {folder}")
            self.status_label.config(text="Study loaded - press View Results to continue")
            self.view_btn.set_enabled(True)

    def _start_status_timer(self):
        """Animate a running indicator on the status label."""
        self._status_tick = 0
        self._tick_status()

    def _tick_status(self):
        dots = "." * (self._status_tick % 4)
        self.status_label.config(text=f"Running analysis{dots}")
        self._status_tick += 1
        self._status_timer_id = self.after(500, self._tick_status)

    def _on_pipeline_done(self):
        """Called on the Tk thread once the pipeline subprocess finishes."""
        # Stop the animated status label
        if hasattr(self, '_status_timer_id'):
            self.after_cancel(self._status_timer_id)
        self.status_label.config(text="Analysis complete - ready to view results")
        # Re-enable buttons
        self.run_btn.set_enabled(True)
        self.view_btn.set_enabled(True)

    def _start_pipeline(self):
        sm_dir = self.selected_root_folder.get().strip()
        if not os.path.isdir(sm_dir):
            messagebox.showerror("Error", f"Simplemind directory not found:\n{sm_dir}")
            return

        t = threading.Thread(target=self._pipeline_thread, daemon=True)
        t.start()

    def _pipeline_thread(self):
        sm_dir  = self.selected_root_folder.get().strip()
        plan    = self._plan_var.get().strip()
        csv     = self._csv_var.get().strip()
        addr    = self._addr_var.get().strip()
        gpu     = self._gpu_var.get().strip()

        env = os.environ.copy()
        # Activate micromamba env
        mamba_bin = os.path.expanduser("~/micromamba/envs/smcore/bin")
        if os.path.isdir(mamba_bin):
            env["PATH"] = mamba_bin + os.pathsep + env.get("PATH", "")

        # 1. Start blackboard
        print("Starting blackboard (core start server)...")
        try:
            self._bb_proc = subprocess.Popen(
                ["core", "start", "server"],
                cwd=sm_dir, env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            print("⚠  'core' command not found - skipping BB start.")

        time.sleep(2)

        # 2. Run plan
        cmd = [
            "python", "run_plan.py", plan,
            "--dataset_csv", csv,
            "--gpu_num", gpu,
            "--addr", addr,
            "--dashboard"
        ]
        print(f"Running: {' '.join(cmd)}")
        print("Running plan...")

        try:
            self._plan_proc = subprocess.Popen(
                cmd, cwd=sm_dir, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1
            )
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            print("Error - check log")
            return

        # Start polling immediately — _poll_folder runs on Tk thread every 2s
        self.after(0, self._poll_folder)

        # Stream log output
        for line in self._plan_proc.stdout:
            line = line.rstrip()
            if line:
                print(line)
            if self._working_dir is None and "working-" in line:
                for token in line.split():
                    if "working-" in token:
                        candidate = token.strip("'\".,:")
                        if not os.path.isabs(candidate):
                            candidate = os.path.join(sm_dir, candidate)
                        candidate = os.path.normpath(candidate)
                        if os.path.isdir(candidate):
                            self._working_dir = candidate
                            print(f"Working dir (from log): {candidate}")
                            break

        ret = self._plan_proc.wait()
        print(f"Plan exited (code {ret})")
        print("Done - showing results")
        # Signal the Tk main thread that the pipeline is finished
        self.after(0, self._on_pipeline_done)

    #  Continuous folder poll 
    def _poll_folder(self):
        sm_dir = self.selected_root_folder.get().strip()
        if self._working_dir is None and sm_dir:
            matches = sorted(glob.glob(os.path.join(sm_dir, "working-*")),
                             key=os.path.getctime)
            if matches:
                self._working_dir = matches[-1]
                print(f"Found working dir: {self._working_dir}")
                return

        base = self._working_dir or sm_dir
        if not base or not os.path.isdir(base):
            self.after(2000, self._poll_folder)
            return

        all_pngs = self._collect_pngs(base)
        existing_paths = {fp for _, fp in self._images}
        new_found = [fp for fp in all_pngs if fp not in existing_paths]

        if new_found:
            was_empty = not self._images
            for fp in new_found:
                ts = os.path.getctime(fp)
                self._images.append((ts, fp))
            self._images.sort(key=lambda x: x[0])
            if was_empty:
                self._start_slideshow()

        self.after(2000, self._poll_folder)

    def _set_mode(self, mode):
        if mode not in ("library", "measurements"):
            return
        self.results_mode = mode
        self.selected_image_index = 0  # reset preview to first image on tab switch
        self._render_results()

    def _render_results(self):
        self._render_tabs()
        if self.results_mode == "library":
            self.measurements_view.pack_forget()
            self.library_view.pack(fill="both", expand=True)
            self._render_library_view()
        else:
            self.library_view.pack_forget()
            self.measurements_view.pack(fill="both", expand=True)
            self._render_measurements_view()
        self._render_preview()

    def _render_library_view(self):
        for child in self.folder_grid.winfo_children():
            child.destroy()
        for child in self.file_grid.winfo_children():
            child.destroy()

        if not self.folder_groups:
            return

        cols = 4
        for idx, (folder_name, _folder_path, _images) in enumerate(self.folder_groups):
            r = idx // cols
            c = idx % cols
            tile = TileButton(
                self.folder_grid,
                folder_name,
                on_click=lambda i=idx: self._select_folder(i),
                selected=(idx == self.selected_folder_index),
            )
            tile.grid(row=r, column=c, padx=10, pady=(0, 8), sticky="n")

        self._render_file_thumbnails(use_measurement_view=False)

    def _render_measurements_view(self):
        for child in self.measurement_file_grid.winfo_children():
            child.destroy()

        if not self.folder_groups:
            tk.Label(self.measurements_view, text="No study selected.", bg=WHITE, fg=TEXT_GRAY, font=("Helvetica", 12, "italic")).pack(pady=20)
            return

        self._render_file_thumbnails(use_measurement_view=True)

    def _get_measurements_folder_index(self):
        """Return the index of the 'Measurements and Annotations' folder group, or -1 if not found."""
        for i, (folder_name, _path, _images) in enumerate(self.folder_groups):
            if "measurement" in folder_name.lower():
                return i
        return -1

    def _render_file_thumbnails(self, use_measurement_view=False):
        target = self.measurement_file_grid if use_measurement_view else self.file_grid
        for child in target.winfo_children():
            child.destroy()

        if use_measurement_view:
            # Always show the Measurements and Annotations folder in the Measurements tab
            meas_idx = self._get_measurements_folder_index()
            folder_idx = meas_idx if meas_idx != -1 else self.selected_folder_index
        else:
            folder_idx = self.selected_folder_index

        images = self.folder_groups[folder_idx][2]
        if not images:
            tk.Label(target, text="No image files in this folder.", bg=WHITE, fg=TEXT_GRAY, font=("Helvetica", 12, "italic")).pack(anchor="w", pady=16)
            return

        cols = 4
        self.file_thumb_refs = []
        print(images)
        for idx, img_path in enumerate(images):
            r = idx // cols
            c = idx % cols
            thumb = load_thumbnail_from_file(img_path, size=(100, 66))
            self.file_thumb_refs.append(thumb)
            tile = tk.Frame(target, bg=WHITE)
            tile.grid(row=r, column=c, padx=14, pady=(0, 18), sticky="n")
            border = tk.Label(
                tile,
                image=thumb,
                bg=WHITE,
                borderwidth=0,
                highlightthickness=3 if idx == self.selected_image_index else 0,
                highlightbackground=PURPLE_BORDER,
                cursor="hand2",
            )
            border.pack()
            border.bind("<Button-1>", lambda _e, i=idx: self._select_image(i))

            name = os.path.basename(img_path)
            tk.Label(tile, text=name, bg=WHITE, fg=TEXT_BLACK, font=("Helvetica", 9, "normal"), wraplength=110, justify="center").pack(pady=(6, 0))

    def _render_preview(self):
        self.preview_canvas.delete("all")

        if not self.folder_groups:
            self.preview_mode_label.config(text="")
            self.preview_title.config(text="")
            self.preview_detail.config(text="")
            return

        if self.results_mode == "measurements":
            meas_idx = self._get_measurements_folder_index()
            folder_idx = meas_idx if meas_idx != -1 else self.selected_folder_index
        else:
            folder_idx = self.selected_folder_index

        images = self.folder_groups[folder_idx][2]
        if not images:
            self.preview_mode_label.config(text=self.results_mode.capitalize())
            self.preview_title.config(text=self.folder_groups[folder_idx][0])
            self.preview_detail.config(text="")
            self.preview_ref = make_placeholder_thumbnail((500, 380), label="No images")
            self.preview_canvas.create_image(250, 190, image=self.preview_ref)
            return

        self.selected_image_index = max(0, min(self.selected_image_index, len(images) - 1))
        image_path = images[self.selected_image_index]
        self.preview_mode_label.config(text=self.results_mode.capitalize())
        self.preview_title.config(text=os.path.basename(image_path))
        self.preview_detail.config(text="")

        # Use the actual rendered canvas dimensions so the image fills the panel
        self.preview_canvas.update_idletasks()
        canvas_w = self.preview_canvas.winfo_width() or 450
        canvas_h = self.preview_canvas.winfo_height() or 500

        try:
            img = Image.open(image_path).convert("RGBA")
            img.thumbnail((canvas_w - 8, canvas_h - 8), Image.LANCZOS)
            self.preview_ref = ImageTk.PhotoImage(img)
        except Exception:
            self.preview_ref = make_placeholder_thumbnail(
                (canvas_w - 8, canvas_h - 8),
                label=os.path.basename(image_path).split(".")[0]
            )

        self.preview_canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.preview_ref)

    def _select_folder(self, idx):
        self.selected_folder_index = idx
        self.selected_image_index = 0
        self._render_results()

    def _select_image(self, idx):
        self.selected_image_index = idx
        self._render_results()

    def _active_images(self):
        """Return the image list for the currently active tab."""
        if self.results_mode == "measurements":
            meas_idx = self._get_measurements_folder_index()
            folder_idx = meas_idx if meas_idx != -1 else self.selected_folder_index
        else:
            folder_idx = self.selected_folder_index
        return self.folder_groups[folder_idx][2]

    def _prev_item(self):
        if not self.folder_groups:
            return
        images = self._active_images()
        if not images:
            return
        self.selected_image_index = (self.selected_image_index - 1) % len(images)
        self._render_results()

    def _next_item(self):
        if not self.folder_groups:
            return
        images = self._active_images()
        if not images:
            return
        self.selected_image_index = (self.selected_image_index + 1) % len(images)
        self._render_results()

    def _zoom_item(self):
        messagebox.showinfo("Prototype", "Zoom will be connected in the next step.")

    def _pan_item(self):
        messagebox.showinfo("Prototype", "Pan will be connected in the next step.")


if __name__ == "__main__":
    app = ASOCTApp()
    app.mainloop()
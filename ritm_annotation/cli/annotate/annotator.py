# flake8: noqa E501

import json
import logging
import tkinter as tk
from gettext import gettext as _
from sys import exit
from tkinter import messagebox, ttk

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from PIL import Image

from ritm_annotation.cli.annotate.canvas import CanvasImage
from ritm_annotation.cli.annotate.controller import InteractiveController
from ritm_annotation.cli.annotate.wrappers import (
    BoundedNumericalEntry,
    FocusButton,
    FocusCheckButton,
    FocusHorizontalScale,
    FocusLabelFrame,
)
from ritm_annotation.inference.utils import find_checkpoint, load_is_model
from ritm_annotation.utils.misc import (
    get_default_weight,
    ignore_params_then_call,
)

logger = logging.getLogger(__name__)


class InteractiveDemoApp(ttk.Frame):
    def __init__(self, master, args, model, tasks_iterator, classes_first=True):
        logger.info("Initializing...")
        super().__init__(master)
        self.classes_first = classes_first
        self.tasks = list(tasks_iterator)
        self._current_task_idx = 0
        self.master = master
        master.title(
            _(
                "Reviving Iterative Training with Mask Guidance for Interactive Segmentation"
            )  # noqa: E501
        )
        master.withdraw()
        master.update_idletasks()
        x = (master.winfo_screenwidth() - master.winfo_reqwidth()) / 2
        y = (master.winfo_screenheight() - master.winfo_reqheight()) / 2
        master.geometry("+%d+%d" % (x, y))
        self.pack(fill="both", expand=True)

        self.brs_modes = [
            "NoBRS",
            "RGB-BRS",
            "DistMap-BRS",
            "f-BRS-A",
            "f-BRS-B",
            "f-BRS-C",
        ]
        self.limit_longest_size = args.limit_longest_size

        self.controller = InteractiveController(
            model,
            args.device,
            predictor_params={"brs_mode": "NoBRS"},
            update_image_callback=self._update_image,
        )

        self._init_state()
        self._add_menu()
        self._add_canvas()
        self._add_buttons()

        master.bind("<Escape>", lambda event: exit(0))
        master.bind(
            "j",
            lambda event: self.state["visualize_mask"].set(
                not self.state["visualize_mask"].get()
            ),
        )
        master.bind(
            "k",
            lambda event: self.state["visualize_blender_contrast"].set(
                not self.state["visualize_blender_contrast"].get()
            ),
        )
        master.bind("l", lambda event: self._save_mask_callback())
        master.bind("h", lambda event: self._undo_click())
        master.bind("m", lambda event: self._reset_last_object())
        master.bind("n", lambda event: self._load_image_callback())
        master.bind("b", lambda event: self._goto_previous_task())

        self.state["zoomin_params"]["skip_clicks"].trace(
            mode="w", callback=self._reset_predictor
        )
        self.state["zoomin_params"]["target_size"].trace(
            mode="w", callback=self._reset_predictor
        )
        self.state["zoomin_params"]["expansion_ratio"].trace(
            mode="w", callback=self._reset_predictor
        )
        self.state["predictor_params"]["net_clicks_limit"].trace(
            mode="w", callback=self._change_brs_mode
        )
        self.state["lbfgs_max_iters"].trace(mode="w", callback=self._change_brs_mode)
        self._change_brs_mode()
        self._current_task = None

    def _handle_classe_finalizada(self, classe):
        messagebox.showwarning(
            _("Class finished"),
            _("The class '{class_name}' was finalized").format(class_name=classe),
        )

    def goto_task(self, idx=None):
        if idx is None:
            idx = self._current_task_idx + 1
        if idx >= len(self.tasks):
            raise StopIteration()
        print("goto_task", idx, self.tasks[idx])
        self._current_task = self.tasks[idx]
        self._current_task_idx = idx
        return self._current_task

    def _goto_previous_task(self):
        if self._current_task_idx < 1:
            return
        self._current_task_idx -= 1
        self._load_image_callback(current_task=self.goto_task(self._current_task_idx))

    def _get_current_task(self, ask_next=False):
        current_class = (
            self._current_task.class_name
            if self._current_task is not None
            else _("* no one *")
        )
        try:
            if self._current_task is None or ask_next:
                logger.debug(_("Pulling next task from the iterator"))
                next_task = self.goto_task()
                if current_class != next_task.class_name:
                    self._handle_classe_finalizada(current_class)
                self._current_task = next_task
            self.task_label.config(
                text=f"class:{self._current_task.class_name} {self._current_task.image}"  # noqa: E501
            )
            return self._current_task
        except StopIteration:
            self._handle_classe_finalizada(current_class)
            exit(0)

    def _init_state(self):
        self.state = {
            "zoomin_params": {
                "use_zoom_in": tk.BooleanVar(value=True),
                "fixed_crop": tk.BooleanVar(value=True),
                "skip_clicks": tk.IntVar(value=-1),
                "target_size": tk.IntVar(value=min(400, self.limit_longest_size)),
                "expansion_ratio": tk.DoubleVar(value=1.4),
            },
            "predictor_params": {"net_clicks_limit": tk.IntVar(value=8)},
            "brs_mode": tk.StringVar(value="NoBRS"),
            "prob_thresh": tk.DoubleVar(value=0.5),
            "lbfgs_max_iters": tk.IntVar(value=20),
            "alpha_blend": tk.DoubleVar(value=0.5),
            "click_radius": tk.IntVar(value=3),
            "visualize_mask": tk.BooleanVar(value=False),
            "visualize_blender_contrast": tk.BooleanVar(value=False),
        }

    def _add_menu(self):
        self.menubar = FocusLabelFrame(self, bd=1)
        self.menubar.pack(side=tk.TOP, fill="x")

        button = FocusButton(
            self.menubar,
            text=_("Load image") + " (n)",
            command=self._load_image_callback,
        )
        button.pack(side=tk.LEFT)
        self.save_mask_btn = FocusButton(
            self.menubar,
            text=_("Save mask") + " (l)",
            command=self._save_mask_callback,
        )
        self.save_mask_btn.pack(side=tk.LEFT)
        self.save_mask_btn.configure(state=tk.DISABLED)

        self.previous_image_btn = FocusButton(
            self.menubar,
            text=_("Previous image") + " (b)",
            command=self._goto_previous_task,
        )
        self.previous_image_btn.pack(side=tk.LEFT)
        # self.previous_image_btn.configure(state=tk.DISABLED)

        self.load_mask_btn = FocusButton(
            self.menubar, text=_("Load mask"), command=self._load_mask_callback
        )
        self.load_mask_btn.pack(side=tk.LEFT)
        self.load_mask_btn.configure(state=tk.DISABLED)

        button = FocusButton(
            self.menubar, text=_("About"), command=self._about_callback
        )
        button.pack(side=tk.LEFT)
        button = FocusButton(self.menubar, text=_("Exit"), command=self.master.quit)
        button.pack(side=tk.LEFT)

        self.task_label = tk.Label(self.menubar, text=_("* status *"))
        self.task_label.pack(side=tk.LEFT)

    def _add_canvas(self):
        self.canvas_frame = FocusLabelFrame(self, text="Image")
        self.canvas_frame.rowconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            self.canvas_frame,
            highlightthickness=0,
            cursor="hand1",
            width=400,
            height=400,
        )
        self.canvas.grid(row=0, column=0, sticky="nswe", padx=5, pady=5)

        self.image_on_canvas = None
        self.canvas_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=5, pady=5)

    def _add_buttons(self):
        self.control_frame = FocusLabelFrame(self, text=_("Controls"))
        self.control_frame.pack(side=tk.TOP, fill="x", padx=5, pady=5)
        master = self.control_frame

        FocusCheckButton(
            master,
            text=f"{_('Mostrar máscara ao invés da imagem')} (j)",
            variable=self.state["visualize_mask"],
        ).pack(side=tk.TOP)
        self.state["visualize_mask"].trace(
            mode="w", callback=ignore_params_then_call(self._update_image)
        )
        FocusCheckButton(
            master,
            text=f"{_('Mostrar sobreposição da máscara com mais contraste')} (k)",
            variable=self.state["visualize_blender_contrast"],
        ).pack(side=tk.TOP)
        self.state["visualize_blender_contrast"].trace(
            mode="w", callback=ignore_params_then_call(self._update_image)
        )

        self.clicks_options_frame = FocusLabelFrame(master, text=_("Clicks management"))
        self.clicks_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        self.finish_object_button = FocusButton(
            self.clicks_options_frame,
            text=_("Finish\nobject"),
            bg="#b6d7a8",
            fg="black",
            width=10,
            height=2,
            state=tk.DISABLED,
            command=self.controller.finish_object,
        )
        self.finish_object_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        self.undo_click_button = FocusButton(
            self.clicks_options_frame,
            text=f"{_('Undo click')} (h)",
            bg="#ffe599",
            fg="black",
            width=10,
            height=2,
            state=tk.DISABLED,
            command=self._undo_click,
        )
        self.undo_click_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        self.reset_clicks_button = FocusButton(
            self.clicks_options_frame,
            text=f"{_('Reset clicks')} (m)",
            bg="#ea9999",
            fg="black",
            width=10,
            height=2,
            state=tk.DISABLED,
            command=self._reset_last_object,
        )
        self.reset_clicks_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)

        self.zoomin_options_frame = FocusLabelFrame(master, text=_("ZoomIn options"))
        self.zoomin_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusCheckButton(
            self.zoomin_options_frame,
            text=_("Use ZoomIn"),
            command=self._reset_predictor,
            variable=self.state["zoomin_params"]["use_zoom_in"],
        ).grid(row=0, column=0, padx=10)
        FocusCheckButton(
            self.zoomin_options_frame,
            text=_("Fixed crop"),
            command=self._reset_predictor,
            variable=self.state["zoomin_params"]["fixed_crop"],
        ).grid(row=1, column=0, padx=10)
        tk.Label(self.zoomin_options_frame, text=_("Skip clicks")).grid(
            row=0, column=1, pady=1, sticky="e"
        )
        tk.Label(self.zoomin_options_frame, text=_("Target size")).grid(
            row=1, column=1, pady=1, sticky="e"
        )
        tk.Label(self.zoomin_options_frame, text=_("Expand ratio")).grid(
            row=2, column=1, pady=1, sticky="e"
        )
        BoundedNumericalEntry(
            self.zoomin_options_frame,
            variable=self.state["zoomin_params"]["skip_clicks"],
            min_value=-1,
            max_value=None,
            vartype=int,
            name="zoom_in_skip_clicks",
        ).grid(row=0, column=2, padx=10, pady=1, sticky="w")
        BoundedNumericalEntry(
            self.zoomin_options_frame,
            variable=self.state["zoomin_params"]["target_size"],
            min_value=100,
            max_value=self.limit_longest_size,
            vartype=int,
            name="zoom_in_target_size",
        ).grid(row=1, column=2, padx=10, pady=1, sticky="w")
        BoundedNumericalEntry(
            self.zoomin_options_frame,
            variable=self.state["zoomin_params"]["expansion_ratio"],
            min_value=1.0,
            max_value=2.0,
            vartype=float,
            name="zoom_in_expansion_ratio",
        ).grid(row=2, column=2, padx=10, pady=1, sticky="w")
        self.zoomin_options_frame.columnconfigure((0, 1, 2), weight=1)

        self.brs_options_frame = FocusLabelFrame(master, text=_("BRS options"))
        self.brs_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        menu = tk.OptionMenu(
            self.brs_options_frame,
            self.state["brs_mode"],
            *self.brs_modes,
            command=self._change_brs_mode,
        )
        menu.config(width=11)
        menu.grid(rowspan=2, column=0, padx=10)
        self.net_clicks_label = tk.Label(
            self.brs_options_frame, text=_("Network clicks")
        )
        self.net_clicks_label.grid(row=0, column=1, pady=2, sticky="e")
        self.net_clicks_entry = BoundedNumericalEntry(
            self.brs_options_frame,
            variable=self.state["predictor_params"]["net_clicks_limit"],
            min_value=0,
            max_value=None,
            vartype=int,
            allow_inf=True,
            name="net_clicks_limit",
        )
        self.net_clicks_entry.grid(row=0, column=2, padx=10, pady=2, sticky="w")
        self.lbfgs_iters_label = tk.Label(
            self.brs_options_frame, text=_("L-BFGS\nmax iterations")
        )
        self.lbfgs_iters_label.grid(row=1, column=1, pady=2, sticky="e")
        self.lbfgs_iters_entry = BoundedNumericalEntry(
            self.brs_options_frame,
            variable=self.state["lbfgs_max_iters"],
            min_value=1,
            max_value=1000,
            vartype=int,
            name="lbfgs_max_iters",
        )
        self.lbfgs_iters_entry.grid(row=1, column=2, padx=10, pady=2, sticky="w")
        self.brs_options_frame.columnconfigure((0, 1), weight=1)

        self.prob_thresh_frame = FocusLabelFrame(
            master, text=_("Predictions threshold")
        )
        self.prob_thresh_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(
            self.prob_thresh_frame,
            from_=0.0,
            to=1.0,
            command=self._update_prob_thresh,
            variable=self.state["prob_thresh"],
        ).pack(padx=10)

        self.alpha_blend_frame = FocusLabelFrame(
            master, text=_("Alpha blending coefficient")
        )
        self.alpha_blend_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(
            self.alpha_blend_frame,
            from_=0.0,
            to=1.0,
            command=self._update_blend_alpha,
            variable=self.state["alpha_blend"],
        ).pack(padx=10, anchor=tk.CENTER)

        self.click_radius_frame = FocusLabelFrame(
            master, text=_("Visualisation click radius")
        )
        self.click_radius_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(
            self.click_radius_frame,
            from_=0,
            to=7,
            resolution=1,
            command=self._update_click_radius,
            variable=self.state["click_radius"],
        ).pack(padx=10, anchor=tk.CENTER)

    def _load_image_callback(self, current_task=None):
        self.menubar.focus_set()
        if self._check_entry(self):
            if current_task is None:
                current_task = self._get_current_task(ask_next=True)
            filename = str(current_task.image.resolve())
            logger.info(_("Loading '{filename}'").format(filename=filename))
            image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
            self.controller.set_image(image)
            points_json_name = current_task.class_name + ".json"
            points_json = current_task.output_dir / points_json_name
            if current_task.seed_mask is not None:
                points_json = current_task.seed_mask.parent / points_json_name
                if not points_json.exists():
                    points_json = current_task.output_dir / points_json_name
            if points_json.exists():
                with points_json.open("r") as f:
                    data = json.load(f)
                    for item in data:
                        self._click_callback(
                            item["is_positive"],
                            item["coords"][1],
                            item["coords"][0],
                        )
                    # from ritm_annotation.inference.clicker import Click
                    # data = [Click(item['is_positive'], item['coords'][1], item['coords'][0]) for item in data]
            else:
                if (
                    current_task.seed_mask is not None
                    and current_task.seed_mask.exists()
                ):
                    logger.debug(f"seed mask: {current_task.seed_mask}")
                    mask = cv2.imread(str(current_task.seed_mask), 0)
                    mask = mask > 127
                    self.controller.set_mask(mask)
            self.save_mask_btn.configure(state=tk.NORMAL)
            self.load_mask_btn.configure(state=tk.NORMAL)
            self._update_image()
            self._reset_predictor()

    def _undo_click(self):
        self.controller.undo_click()
        self._update_image()

    def _save_mask_callback(self):
        self.menubar.focus_set()
        if self._check_entry(self):
            mask = self.controller.result_mask
            if mask is None:
                return
            current_task = self._get_current_task()
            filename = str(current_task.mask_output.resolve())
            mask = (mask > 0) * 255
            logger.info(_("Saving '{filename}'").format(filename=filename))
            cv2.imwrite(filename, mask)
            points_json = current_task.output_dir / (current_task.class_name + ".json")
            with points_json.open("w") as f:
                # json.dump(self.controller.states, f)
                json.dump(
                    [item.to_json() for item in self.controller.clicker.get_state()],
                    f,
                )

    def _load_mask_callback(self):
        if not self.controller.net.with_prev_mask:
            messagebox.showwarning(
                _("Warning"),
                _(
                    "The current model doesn't support loading external masks. Please use ITER-M models for that purpose."
                ),
            )
            return

        self.menubar.focus_set()
        if self._check_entry(self):
            current_task = self._get_current_task()
            filename = current_task.mask_output.resolve()
            if not filename.exist() and current_task.seed_mask is not None:
                filename = current_task.seed_mask
            filename = str(filename)
            logger.debug(_("Loading '{filename}'").format(filename=filename))
            mask = cv2.imread(filename, 0)
            logger.debug(f"mask: {mask}")
            if mask is None:
                messagebox.showwarning(_("Warning"), _("No mask found"))
            else:
                mask = mask > 127
                self.controller.set_mask(mask)
                self._update_image()

    def _about_callback(self):
        self.menubar.focus_set()

        text = [
            _("Developed by:"),
            "K.Sofiiuk and I. Petrov",
            "The MIT License, 2021",
        ]

        messagebox.showinfo(_("About Demo"), "\n".join(text))

    def _reset_last_object(self):
        self.state["alpha_blend"].set(0.5)
        self.state["prob_thresh"].set(0.5)
        self.controller.reset_last_object()

    def _update_prob_thresh(self, value):
        if self.controller.is_incomplete_mask:
            self.controller.prob_thresh = self.state["prob_thresh"].get()
            self._update_image()

    def _update_blend_alpha(self, value):
        self._update_image()

    def _update_click_radius(self, *args):
        if self.image_on_canvas is None:
            return

        self._update_image()

    def _change_brs_mode(self, *args):
        if self.state["brs_mode"].get() == "NoBRS":
            self.net_clicks_entry.set("INF")
            self.net_clicks_entry.configure(state=tk.DISABLED)
            self.net_clicks_label.configure(state=tk.DISABLED)
            self.lbfgs_iters_entry.configure(state=tk.DISABLED)
            self.lbfgs_iters_label.configure(state=tk.DISABLED)
        else:
            if self.net_clicks_entry.get() == "INF":
                self.net_clicks_entry.set(8)
            self.net_clicks_entry.configure(state=tk.NORMAL)
            self.net_clicks_label.configure(state=tk.NORMAL)
            self.lbfgs_iters_entry.configure(state=tk.NORMAL)
            self.lbfgs_iters_label.configure(state=tk.NORMAL)

        self._reset_predictor()

    def _reset_predictor(self, *args, **kwargs):
        brs_mode = self.state["brs_mode"].get()
        prob_thresh = self.state["prob_thresh"].get()
        net_clicks_limit = (
            None
            if brs_mode == "NoBRS"
            else self.state["predictor_params"]["net_clicks_limit"].get()
        )

        if self.state["zoomin_params"]["use_zoom_in"].get():
            zoomin_params = {
                "skip_clicks": self.state["zoomin_params"]["skip_clicks"].get(),
                "target_size": self.state["zoomin_params"]["target_size"].get(),
                "expansion_ratio": self.state["zoomin_params"]["expansion_ratio"].get(),
            }
            if self.state["zoomin_params"]["fixed_crop"].get():
                zoomin_params["target_size"] = (
                    zoomin_params["target_size"],
                    zoomin_params["target_size"],
                )
        else:
            zoomin_params = None

        predictor_params = {
            "brs_mode": brs_mode,
            "prob_thresh": prob_thresh,
            "zoom_in_params": zoomin_params,
            "predictor_params": {
                "net_clicks_limit": net_clicks_limit,
                "max_size": self.limit_longest_size,
            },
            "brs_opt_func_params": {"min_iou_diff": 1e-3},
            "lbfgs_params": {"maxfun": self.state["lbfgs_max_iters"].get()},
        }
        self.controller.reset_predictor(predictor_params)

    def _click_callback(self, is_positive, x, y):
        self.canvas.focus_set()

        if self.image_on_canvas is None:
            messagebox.showwarning(_("Warning"), _("Please load an image first"))
            return

        if self._check_entry(self):
            self.controller.add_click(x, y, is_positive)

    def _update_image(self, reset_canvas=False):
        image = self.controller.get_visualization(
            alpha_blend=self.state["alpha_blend"].get(),
            click_radius=self.state["click_radius"].get(),
        )
        image_mask = (
            np.array(
                cv2.cvtColor(self.controller.result_mask, cv2.COLOR_GRAY2RGB),
                dtype="uint8",
            )
            * 255
        )
        if self.state["visualize_mask"].get():
            image = image_mask

        if self.state["visualize_blender_contrast"].get():
            image = (image // 2) + (image_mask // 2)

        if self.image_on_canvas is None:
            self.image_on_canvas = CanvasImage(self.canvas_frame, self.canvas)
            self.image_on_canvas.register_click_callback(self._click_callback)

        self._set_click_dependent_widgets_state()
        if image is not None:
            self.image_on_canvas.reload_image(Image.fromarray(image), reset_canvas)

    def _set_click_dependent_widgets_state(self):
        after_1st_click_state = (
            tk.NORMAL if self.controller.is_incomplete_mask else tk.DISABLED
        )
        before_1st_click_state = (
            tk.DISABLED if self.controller.is_incomplete_mask else tk.NORMAL
        )

        self.finish_object_button.configure(state=after_1st_click_state)
        self.undo_click_button.configure(state=after_1st_click_state)
        self.reset_clicks_button.configure(state=after_1st_click_state)
        self.zoomin_options_frame.set_frame_state(before_1st_click_state)
        self.brs_options_frame.set_frame_state(before_1st_click_state)

        if self.state["brs_mode"].get() == "NoBRS":
            self.net_clicks_entry.configure(state=tk.DISABLED)
            self.net_clicks_label.configure(state=tk.DISABLED)
            self.lbfgs_iters_entry.configure(state=tk.DISABLED)
            self.lbfgs_iters_label.configure(state=tk.DISABLED)

    def _check_entry(self, widget):
        all_checked = True
        if widget.winfo_children is not None:
            for w in widget.winfo_children():
                all_checked = all_checked and self._check_entry(w)

        if getattr(widget, "_check_bounds", None) is not None:
            all_checked = all_checked and widget._check_bounds(widget.get(), "-1")

        return all_checked


def handle(args):
    if not torch.cuda.is_available():
        logger.warn(
            _("CUDA support not detected. Stuff will be way slower and sluggish!")
        )
    logger.debug(_("classes: ") + ", ".join(args.classes))

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    assert args.input.is_dir(), _("Origin should be a folder")
    # assert args.input.is_dir(), _("Origem precisa ser uma pasta")
    assert not args.output.exists() or args.output.is_dir(), _(
        "Destination should be a folder, if it isn't exist it will be created"
    )
    # ), "Destino precisa ser uma pasta, se não existe vai ser criada"

    if args.checkpoint is None:
        logger.info(_("Checkpoint not specified, using default..."))
        args.checkpoint = get_default_weight()

    torch.backends.cudnn.determistic = True

    checkpoint_path = find_checkpoint(args.checkpoint.parent, args.checkpoint.name)
    model = load_is_model(checkpoint_path, args.device, cpu_dist_maps=True)

    def _file_iterator():
        return args.input.iterdir()

    def _class_iterator():
        return iter(args.classes)

    def _fileclass_iterator():
        if args.classes_first:
            for file in _file_iterator():
                for cls in _class_iterator():
                    yield file, cls
        else:
            for cls in _class_iterator():
                for file in _file_iterator():
                    yield file, cls

    def look_for_tasks():
        for image, class_name in _fileclass_iterator():
            if not image.is_file():
                continue
            image_name = image.name
            output_dir = args.output / image_name
            output_dir.mkdir(exist_ok=True, parents=True)
            mask_output = output_dir / f"{class_name}.png"
            if mask_output.exists():
                continue
            yield edict(
                dict(
                    output_dir=output_dir,
                    image=image,
                    name=image_name,
                    mask_output=mask_output,
                    class_name=class_name,
                    seed_mask=args.seed / image_name / f"{class_name}.png"
                    if args.seed is not None
                    else None,
                )
            )

    tasks = look_for_tasks()

    root = tk.Tk()
    root.minsize(960, 480)

    app_args = edict(dict(limit_longest_size=400, device=args.device))

    app = InteractiveDemoApp(
        root, app_args, model, tasks, classes_first=args.classes_first
    )
    root.deiconify()
    app.mainloop()

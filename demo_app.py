from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps

try:
    from PIL import ImageTk
except ImportError:  # pragma: no cover - depends on local Pillow build
    ImageTk = None

from astronomy_recognizer.recognizer import MessierRecognizer


INDEX_PATH = Path("artifacts/reference_index.json")


class DemoApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Messier Object Recognition System")
        self.root.geometry("980x760")
        self.root.configure(bg="#09111f")

        if not INDEX_PATH.exists():
            messagebox.showerror(
                "Missing index",
                "Reference index not found. Run `python3 main.py build-index` first.",
            )
            raise SystemExit(1)

        self.recognizer = MessierRecognizer(INDEX_PATH)
        self.preview_image: ImageTk.PhotoImage | None = None
        self.star_map_preview: ImageTk.PhotoImage | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#09111f")
        style.configure("TLabel", background="#09111f", foreground="#edf3ff")
        style.configure("Title.TLabel", font=("Helvetica", 22, "bold"))
        style.configure("Body.TLabel", font=("Helvetica", 11))
        style.configure("Result.TLabel", font=("Helvetica", 12), background="#10203a")
        style.configure("TButton", font=("Helvetica", 11, "bold"))

        container = ttk.Frame(self.root, padding=20)
        container.pack(fill="both", expand=True)

        ttk.Label(
            container,
            text="Astronomical Object Recognition System",
            style="Title.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            container,
            text="Upload an astronomical image and match it against the 110 Messier objects.",
            style="Body.TLabel",
        ).pack(anchor="w", pady=(6, 16))

        controls = ttk.Frame(container)
        controls.pack(fill="x", pady=(0, 16))

        ttk.Button(controls, text="Choose Image", command=self.choose_image).pack(side="left")

        self.file_label = ttk.Label(
            controls,
            text="No image selected",
            style="Body.TLabel",
        )
        self.file_label.pack(side="left", padx=(12, 0))

        content = ttk.Frame(container)
        content.pack(fill="both", expand=True)

        left_panel = ttk.Frame(content)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))

        right_panel = ttk.Frame(content)
        right_panel.pack(side="left", fill="both", expand=True)

        preview_panel = ttk.Frame(left_panel)
        preview_panel.pack(fill="both", expand=True)

        image_section = ttk.Frame(preview_panel)
        image_section.pack(fill="both", expand=True, pady=(0, 10))
        ttk.Label(image_section, text="Input Image", style="Body.TLabel").pack(anchor="w", pady=(0, 8))
        self.image_label = ttk.Label(image_section, text="Preview will appear here", anchor="center")
        self.image_label.pack(fill="both", expand=True)

        map_section = ttk.Frame(preview_panel)
        map_section.pack(fill="both", expand=True)
        ttk.Label(map_section, text="Star Map", style="Body.TLabel").pack(anchor="w", pady=(0, 8))
        self.star_map_label = ttk.Label(map_section, text="Star map will appear here", anchor="center")
        self.star_map_label.pack(fill="both", expand=True)

        ttk.Label(right_panel, text="Prediction Results", style="Body.TLabel").pack(anchor="w", pady=(0, 8))

        self.result_text = tk.Text(
            right_panel,
            wrap="word",
            bg="#10203a",
            fg="#edf3ff",
            insertbackground="#edf3ff",
            relief="flat",
            font=("Helvetica", 12),
            padx=16,
            pady=16,
        )
        self.result_text.pack(fill="both", expand=True)
        self.result_text.insert(
            "1.0",
            "Build the index, choose an image, and the system will show the best Messier matches here.",
        )
        self.result_text.configure(state="disabled")

    def choose_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Choose an astronomical image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        try:
            self._update_preview(file_path)
            matches = self.recognizer.predict(file_path, top_k=3)
            self.file_label.configure(text=file_path)
            self._show_results(file_path, matches)
            self._update_star_map(matches[0].star_map_image)
        except Exception as error:  # pragma: no cover - demo error path
            messagebox.showerror("Prediction error", str(error))

    def _update_preview(self, file_path: str) -> None:
        if ImageTk is None:
            self.image_label.configure(
                text="Preview unavailable in this environment.\nPrediction still works normally.",
                image="",
            )
            return

        image = Image.open(file_path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((420, 420))
        self.preview_image = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.preview_image, text="")

    def _update_star_map(self, file_path: str | None) -> None:
        if not file_path:
            self.star_map_label.configure(text="Star map not available.", image="")
            return

        if ImageTk is None:
            self.star_map_label.configure(
                text=f"Star map found:\n{file_path}\n\nPreview unavailable in this environment.",
                image="",
            )
            return

        image = Image.open(file_path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((420, 280))
        self.star_map_preview = ImageTk.PhotoImage(image)
        self.star_map_label.configure(image=self.star_map_preview, text="")

    def _show_results(self, file_path: str, matches) -> None:
        lines = [f"Input image: {file_path}", ""]
        top_match = matches[0]
        lines.extend(
            [
                "Best Match",
                f"Object: {top_match.title}",
                f"Messier ID: {top_match.messier_id}",
                f"Category: {top_match.category}",
                f"Confidence: {top_match.confidence_label} ({top_match.similarity_score}%)",
                f"Description: {top_match.description}",
                f"Star map: {top_match.star_map_image or 'Not available'}",
                "",
                "Top 3 Matches",
            ]
        )

        for index, match in enumerate(matches, start=1):
            lines.extend(
                [
                    f"{index}. {match.title}",
                    f"   Category: {match.category}",
                    f"   Confidence: {match.confidence_label} ({match.similarity_score}%)",
                    f"   Reference: {match.reference_image}",
                    f"   Star map: {match.star_map_image or 'Not available'}",
                    "",
                ]
            )

        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", "\n".join(lines).strip())
        self.result_text.configure(state="disabled")


def main() -> None:
    root = tk.Tk()
    DemoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

"""Flask web app for the image restoration project.

Run locally:
    python -m app.app                 # then open http://127.0.0.1:5000
Use a fine-tuned denoiser:
    DENOISER_CKPT=checkpoints/denoiser.pt python -m app.app
"""
import io
import sys
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from app import registry  # noqa: E402

MAX_SIDE = 1024
MAX_UPLOAD_MB = 10


def _read_rgb(file_storage):
    with Image.open(file_storage.stream) as im:
        im = im.convert("RGB")
        if max(im.size) > MAX_SIDE:
            im.thumbnail((MAX_SIDE, MAX_SIDE))
        return np.asarray(im, dtype=np.uint8)


def _read_gray(file_storage, size):
    with Image.open(file_storage.stream) as im:
        im = im.convert("L").resize((size[1], size[0]))
        return np.asarray(im, dtype=np.uint8)


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

    @app.get("/")
    def index():
        return render_template("index.html", tasks=registry.status())

    @app.get("/api/tasks")
    def tasks():
        return jsonify(registry.status())

    @app.post("/api/enhance")
    def enhance():
        task = request.form.get("task", "")
        if task not in registry.TASKS:
            return jsonify(error=f"task must be one of {list(registry.TASKS)}"), 400
        if "image" not in request.files:
            return jsonify(error="send an image file in the 'image' field"), 400
        try:
            img = _read_rgb(request.files["image"])
        except Exception:
            return jsonify(error="could not read that file as an image"), 400
        opts = {k: v for k, v in request.form.items() if k != "task"}
        if "mask" in request.files:
            opts["mask"] = _read_gray(request.files["mask"], img.shape[:2])
        try:
            fn, label = registry.get(task)
        except registry.ModelNotIntegrated:
            return jsonify(error=f"no model is integrated for '{task}' yet"), 501
        try:
            out = fn(img, **opts)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        buf = io.BytesIO()
        Image.fromarray(out).save(buf, format="PNG")
        buf.seek(0)
        resp = send_file(buf, mimetype="image/png")
        resp.headers["X-Model"] = label
        return resp

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)

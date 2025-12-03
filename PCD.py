import base64
import io
from typing import Dict, List

import cv2
import matplotlib

# Set backend sebelum import pyplot agar non-GUI
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)


# ==================== UTILITAS ==================== #
def img_to_base64(img: np.ndarray, color_space: str = "bgr") -> str:
    """
    Konversi ndarray OpenCV ke string base64 PNG untuk ditampilkan di HTML.
    color_space: bgr | rgb | hsv | lab | gray
    """
    display_img = img
    if len(img.shape) == 3:
        if color_space == "bgr":
            display_img = img  # OpenCV write expects BGR; biarkan apa adanya
        elif color_space == "rgb":
            display_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif color_space == "hsv":
            display_img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        elif color_space == "lab":
            display_img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    else:
        color_space = "gray"

    ok, buffer = cv2.imencode(".png", display_img)
    if not ok:
        raise ValueError("Gagal mengubah gambar ke PNG.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def fig_to_base64(fig) -> str:
    """Render matplotlib figure ke base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def gray_world(img_bgr: np.ndarray) -> np.ndarray:
    """Gray World White Balance untuk koreksi kecerahan."""
    img = img_bgr.astype(np.float32)
    mean_b = img[:, :, 0].mean()
    mean_g = img[:, :, 1].mean()
    mean_r = img[:, :, 2].mean()
    mean_gray = (mean_b + mean_g + mean_r) / 3.0

    img[:, :, 0] *= mean_gray / (mean_b + 1e-8)
    img[:, :, 1] *= mean_gray / (mean_g + 1e-8)
    img[:, :, 2] *= mean_gray / (mean_r + 1e-8)

    return np.clip(img, 0, 255).astype(np.uint8)


def create_color_sample(hue_std: float, bstar_std: float, size: int = 140) -> np.ndarray:
    """
    Membuat sampel warna sintetis berdasarkan Hue (0-360) dan b* (0-100).
    Warna disusun di ruang HSV agar mendekati warna pisang pada tiap kategori.
    """
    hue_cv = int(hue_std / 2)  # OpenCV Hue 0-180

    if hue_std > 95:  # Mentah
        sat = 70
        val = 70
    elif hue_std < 87:  # Matang penuh
        sat = 85
        val = 95
    else:  # Setengah matang
        sat = 75
        val = 85

    hsv_img = np.zeros((size, size, 3), dtype=np.uint8)
    hsv_img[:, :, 0] = hue_cv
    hsv_img[:, :, 1] = int(sat * 2.55)
    hsv_img[:, :, 2] = int(val * 2.55)

    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)


# ==================== PIPELINE PEMROSESAN ==================== #
def process_image(file_bytes: bytes) -> Dict:
    """Proses gambar pisang dan kembalikan langkah-langkah, metrik, dan visualisasi."""
    data = np.frombuffer(file_bytes, np.uint8)
    img0 = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img0 is None:
        raise ValueError("Tidak dapat membaca file gambar. Gunakan format JPG atau PNG.")

    steps: List[Dict] = []
    steps.append(
        {
            "title": "Gambar Asli",
            "caption": f"Ukuran awal: {img0.shape[1]}x{img0.shape[0]}",
            "img": img_to_base64(img0, "bgr"),
        }
    )

    # Resize
    target_size = 640
    resized = cv2.resize(img0, (target_size, target_size), interpolation=cv2.INTER_AREA)
    steps.append(
        {
            "title": "Resize 640x640",
            "caption": "Menyesuaikan ukuran sesuai proposal.",
            "img": img_to_base64(resized, "bgr"),
        }
    )

    # Gray World WB
    gw = gray_world(resized)
    steps.append(
        {
            "title": "Gray World White Balance",
            "caption": "Menormalkan kecerahan masing-masing channel.",
            "img": img_to_base64(gw, "bgr"),
        }
    )

    # Gaussian blur
    blur = cv2.GaussianBlur(gw, (5, 5), 0)
    steps.append(
        {
            "title": "Gaussian Blur 5x5",
            "caption": "Mengurangi noise sebelum thresholding.",
            "img": img_to_base64(blur, "bgr"),
        }
    )

    # Konversi ruang warna
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    H, S, V = cv2.split(hsv)
    _, _, B = cv2.split(lab)

    steps.extend(
        [
            {
                "title": "Ruang Warna HSV",
                "caption": "Gambar dalam ruang warna HSV.",
                "img": img_to_base64(hsv, "hsv"),
            },
            {
                "title": "Channel Hue (H)",
                "caption": "Channel Hue 0-180 (OpenCV).",
                "img": img_to_base64(H, "gray"),
            },
            {
                "title": "Channel Saturation (S)",
                "caption": "Channel Saturation 0-255.",
                "img": img_to_base64(S, "gray"),
            },
            {
                "title": "Channel b* (Lab)",
                "caption": "Channel b* 0-255 (menggambarkan kuning-biru).",
                "img": img_to_base64(B, "gray"),
            },
        ]
    )

    # Segmentasi HSV
    lower_hsv = np.array([15, 40, 40])
    upper_hsv = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask_closed = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2
    )
    segmented = cv2.bitwise_and(blur, blur, mask=mask_closed)

    steps.extend(
        [
            {
                "title": "Mask HSV",
                "caption": "Thresholding pada rentang warna kulit pisang.",
                "img": img_to_base64(mask, "gray"),
            },
            {
                "title": "Morphological Closing",
                "caption": "Menutup lubang kecil pada mask.",
                "img": img_to_base64(mask_closed, "gray"),
            },
            {
                "title": "Hasil Segmentasi",
                "caption": "Objek buah pisang yang tersegmentasi.",
                "img": img_to_base64(segmented, "bgr"),
            },
        ]
    )

    mask_bool = mask_closed.astype(bool)
    if mask_bool.any():
        mean_hue = float(H[mask_bool].mean())
        std_hue = float(H[mask_bool].std())
        mean_bstar = float(B[mask_bool].mean())
        std_bstar = float(B[mask_bool].std())
        pixel_count = int(np.sum(mask_bool))
    else:
        mean_hue = std_hue = mean_bstar = std_bstar = 0.0
        pixel_count = 0

    # Histogram
    hist_base64 = None
    if mask_bool.any():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].hist(H[mask_bool].flatten(), bins=50, color="orange", alpha=0.75)
        axes[0].axvline(mean_hue, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_hue:.1f}")
        axes[0].set_xlabel("Hue (OpenCV 0-180)")
        axes[0].set_ylabel("Frekuensi")
        axes[0].set_title("Distribusi Hue (area buah)")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].hist(B[mask_bool].flatten(), bins=50, color="blue", alpha=0.75)
        axes[1].axvline(mean_bstar, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_bstar:.1f}")
        axes[1].set_xlabel("b* (OpenCV 0-255)")
        axes[1].set_ylabel("Frekuensi")
        axes[1].set_title("Distribusi b* (area buah)")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        hist_base64 = fig_to_base64(fig)

    # Klasifikasi
    hue_standard = mean_hue * 2  # OpenCV Hue (0-180) -> 0-360
    bstar_standard = (mean_bstar / 255) * 100  # OpenCV b* (0-255) -> 0-100

    maturity_status = "Tidak diketahui"
    color_desc = "Belum ada data"

    if 74 <= hue_standard <= 120:
        maturity_status = "Mentah (R2-R3)"
        color_desc = "Hijau tua (berdasarkan Hue)"
    elif 64 <= hue_standard <= 73.9:
        maturity_status = "Setengah Matang (R4-R5)"
        color_desc = "Kuning kehijauan (berdasarkan Hue)"
    elif 30 <= hue_standard <= 63.9:
        maturity_status = "Matang Penuh (R6-R7)"
        color_desc = "Kuning cerah (berdasarkan Hue)"
    else:
        diff_mentah = abs(hue_standard - 110)
        diff_setengah = abs(hue_standard - 82)
        diff_matang = abs(hue_standard - 65)
        min_diff = min(diff_mentah, diff_setengah, diff_matang)
        if min_diff == diff_mentah:
            maturity_status = "Mentah (R2-R3)"
            color_desc = "Hijau tua (terdekat berdasarkan Hue)"
        elif min_diff == diff_setengah:
            maturity_status = "Setengah Matang (R4-R5)"
            color_desc = "Kuning kehijauan (terdekat berdasarkan Hue)"
        else:
            maturity_status = "Matang Penuh (R6-R7)"
            color_desc = "Kuning cerah (terdekat berdasarkan Hue)"

    if pixel_count == 0:
        bstar_note = "Tidak ada area tersegmentasi. Coba gambar lain atau atur batas HSV."
    elif maturity_status == "Mentah (R2-R3)" and not (28 <= bstar_standard <= 69.9):
        bstar_note = "Catatan: nilai b* kurang konsisten dengan kategori Mentah."
    elif maturity_status == "Setengah Matang (R4-R5)" and not (70 <= bstar_standard <= 84.9):
        bstar_note = "Catatan: nilai b* kurang konsisten dengan kategori Setengah Matang."
    elif maturity_status == "Matang Penuh (R6-R7)" and not (85 <= bstar_standard <= 120):
        bstar_note = "Catatan: nilai b* kurang konsisten dengan kategori Matang Penuh."
    else:
        bstar_note = "Nilai b* konsisten dengan klasifikasi berdasarkan Hue."

    colors_ref = {
        "Mentah (R2-R3)": {"hue": 110.0, "bstar": 32.0},
        "Setengah Matang (R4-R5)": {"hue": 82.0, "bstar": 38.0},
        "Matang Penuh (R6-R7)": {"hue": 65.0, "bstar": 78.0},
    }
    color_samples = [
        {
            "title": label,
            "caption": f"Hue {params['hue']:.1f}, b* {params['bstar']:.1f}",
            "img": img_to_base64(create_color_sample(params["hue"], params["bstar"]), "bgr"),
        }
        for label, params in colors_ref.items()
    ]

    metrics = {
        "pixel_count": pixel_count,
        "mean_hue_cv": mean_hue,
        "std_hue_cv": std_hue,
        "mean_b_cv": mean_bstar,
        "std_b_cv": std_bstar,
        "mean_hue_standard": hue_standard,
        "mean_b_standard": bstar_standard,
    }

    classification = {
        "status": maturity_status,
        "color_desc": color_desc,
        "bstar_note": bstar_note,
    }

    return {
        "steps": steps,
        "metrics": metrics,
        "hist_base64": hist_base64,
        "classification": classification,
        "color_samples": color_samples,
        "size_original": f"{img0.shape[1]}x{img0.shape[0]}",
        "original": steps[0],
    }


# ==================== WEB APP ==================== #
PAGE_TEMPLATE = """
<!doctype html>
<html lang="id">
<head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Analisis Kematangan Pisang</title>
      <style>
        :root {
          --bg: #f4efe9;
          --card: #fffdf8;
          --accent: #a66a44;
          --muted: #7a6d60;
          --text: #2d2622;
          --danger: #b45309;
          --border: #e7dfd4;
        }
        * { box-sizing: border-box; }
        body {
          margin: 0;
          font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
          background: var(--bg);
        color: var(--text);
        min-height: 100vh;
      }
      header { padding: 28px 24px 12px; text-align: center; }
      h1 { margin: 0 0 8px; letter-spacing: -0.5px; }
      p.lead { margin: 0; color: var(--muted); }
      .container { max-width: 1200px; margin: 0 auto; padding: 0 16px 48px; }
      .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 18px 18px 14px;
        box-shadow: 0 16px 45px rgba(0,0,0,0.06);
      }
      form { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
      .file-label {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 12px;
        border: 1px dashed var(--border);
        border-radius: 12px;
        background: #f9f4ec;
        cursor: pointer;
      }
      .file-btn {
        padding: 8px 12px;
        background: var(--accent);
        color: #fffdf8;
        border-radius: 10px;
        font-weight: 600;
        box-shadow: 0 6px 16px rgba(166,106,68,0.22);
      }
      .file-name { color: var(--muted); font-size: 14px; }
      input[type="file"] { display: none; }
      button {
        background: linear-gradient(135deg, #a66a44, #c48b62);
        color: #fffdf8;
        padding: 10px 16px;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        cursor: pointer;
        box-shadow: 0 10px 25px rgba(196,139,98,0.28);
      }
      button:hover { filter: brightness(1.03); }
      .error {
        margin-top: 12px;
        padding: 12px;
        border-radius: 10px;
        background: #fef2f2;
        border: 1px solid #fecdd3;
        color: #b91c1c;
      }
      .grid { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }
      .step img, .hist img, .colors img { width: 100%; border-radius: 12px; border: 1px solid #e5e7eb; }
      .step-title { font-weight: 700; margin: 8px 0 4px; }
      .muted { color: var(--muted); font-size: 13px; }
      .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-top: 12px; }
      .metric { padding: 12px; background: #f9fafb; border-radius: 12px; border: 1px solid var(--border); }
      .highlight {
        display: flex;
        flex-direction: column;
        gap: 10px;
        padding: 22px 24px;
        background: #ecfdf3;
        border: 1px solid #bbf7d0;
        border-radius: 14px;
        max-width: 560px;
        min-height: 220px;
        justify-content: center;
      }
      .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 14px;
        align-items: stretch;
      }
      .orig-card {
        background: #f9fafb;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 220px;
        box-shadow: 0 12px 28px rgba(0,0,0,0.05);
      }
      .orig-card img {
        width: 100%;
        height: 200px;
        border-radius: 10px;
        border: 1px solid var(--border);
        object-fit: contain;
        background: #ffffff;
        padding: 6px;
      }
      .highlight .badge {
        background: #15803d;
        color: #ecfdf3;
        border: none;
        box-shadow: 0 10px 25px rgba(21,128,61,0.22);
        padding: 12px 16px;
        align-self: flex-start;
        font-size: 18px;
      }
      .metric span { display: block; color: var(--muted); font-size: 12px; }
      .badge {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: #ecfdf3;
        color: #166534;
        border: 1px solid #bbf7d0;
        font-weight: 600;
        margin-top: 4px;
      }
      .section-title { margin: 24px 0 10px; font-size: 20px; letter-spacing: -0.2px; }
      footer { text-align: center; color: var(--muted); padding: 22px 0 32px; font-size: 13px; }
    </style>
</head>
<body>
  <header>
    <h1>Analisis Kematangan Pisang</h1>
    <p class="lead">Unggah foto pisang, sistem akan memproses otomatis dan menampilkan setiap tahap + metriknya.</p>
  </header>
  <div class="container">
    <div class="card">
      <form method="POST" enctype="multipart/form-data">
        <label class="file-label">
          <span class="file-btn">Pilih File</span>
          <span class="file-name" id="file-name">Belum ada file</span>
          <input type="file" name="image" accept="image/*" required onchange="document.getElementById('file-name').textContent = this.files[0]?.name || 'Belum ada file';" />
        </label>
        <button type="submit">Proses Gambar</button>
      </form>
      {% if error %}
        <div class="error">{{ error }}</div>
      {% endif %}
    </div>

    {% if result %}
      <h2 class="section-title">Ringkasan & Klasifikasi</h2>
      <div class="card">
        <div class="summary-grid">
          <div class="highlight">
            <span>Label Kematangan</span>
            <div class="badge">{{ result.classification.status }}</div>
            <div class="muted">{{ result.classification.color_desc }}</div>
          </div>
          <div class="orig-card">
            <img src="data:image/png;base64,{{ result.original.img }}" alt="Gambar Asli" />
          </div>
        </div>
        <div class="metrics" style="margin-top:14px;">
          <div class="metric">
            <span>Mean Hue (OpenCV 0-180)</span>
            <strong>{{ "%.2f"|format(result.metrics.mean_hue_cv) }}</strong>
            <div class="muted">Std: {{ "%.2f"|format(result.metrics.std_hue_cv) }}</div>
          </div>
          <div class="metric">
            <span>Mean Hue (Standar 0-360)</span>
            <strong>{{ "%.2f"|format(result.metrics.mean_hue_standard) }}</strong>
          </div>
          <div class="metric">
            <span>Mean b* (OpenCV 0-255)</span>
            <strong>{{ "%.2f"|format(result.metrics.mean_b_cv) }}</strong>
            <div class="muted">Std: {{ "%.2f"|format(result.metrics.std_b_cv) }}</div>
          </div>
          <div class="metric">
            <span>Mean b* (Standar 0-100)</span>
            <strong>{{ "%.2f"|format(result.metrics.mean_b_standard) }}</strong>
          </div>
          <div class="metric">
            <span>Jumlah piksel area buah</span>
            <strong>{{ result.metrics.pixel_count }}</strong>
            <div class="muted">{{ result.classification.bstar_note }}</div>
          </div>
        </div>
      </div>

      <h2 class="section-title">Tahapan Pemrosesan</h2>
      <div class="grid">
        {% for step in result.steps %}
          <div class="card step">
            <img src="data:image/png;base64,{{ step.img }}" alt="{{ step.title }}" />
            <div class="step-title">{{ loop.index }}. {{ step.title }}</div>
            <div class="muted">{{ step.caption }}</div>
          </div>
        {% endfor %}
      </div>

      {% if result.hist_base64 %}
        <h2 class="section-title">Histogram Area Tersegmentasi</h2>
        <div class="card hist">
          <img src="data:image/png;base64,{{ result.hist_base64 }}" alt="Histogram Hue dan b*" />
        </div>
      {% endif %}

      <h2 class="section-title">Contoh Warna Referensi</h2>
      <div class="grid colors">
        {% for ref in result.color_samples %}
          <div class="card">
            <img src="data:image/png;base64,{{ ref.img }}" alt="{{ ref.title }}" />
            <div class="step-title">{{ ref.title }}</div>
            <div class="muted">{{ ref.caption }}</div>
          </div>
        {% endfor %}
      </div>
    {% endif %}
  </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            error = "Pilih file gambar terlebih dahulu."
        else:
            try:
                result = process_image(file.read())
            except Exception as exc:  # noqa: BLE001
                error = str(exc)

    return render_template_string(PAGE_TEMPLATE, result=result, error=error)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

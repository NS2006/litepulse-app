# LitePulse — versi browser

Halaman statis yang mengukur detak jantung dari kamera depan, sepenuhnya di dalam browser.
Halaman ini menjalankan pipeline yang sama dengan aplikasi Android:

| Tahap | Aplikasi Android | Browser |
|---|---|---|
| Landmark wajah | MediaPipe tasks-vision 0.10.32 | MediaPipe tasks-vision web 0.10.32 |
| Masker ROI (dahi, kedua pipi) | `RoiExtractor.kt` | `js/roi.js` (polygon fill + INTER_AREA, identik dengan cv2) |
| Resample + DiffNormalized/Standardized | `LitePulsePreprocessor.kt` | `js/preprocess.js` |
| Model | `litepulse.ptl` (PyTorch Lite) | `models/litepulse.onnx` (ONNX Runtime Web, WASM, di dalam worker) |
| Detrend, Butterworth, puncak FFT, kualitas sinyal | `LitePulseHeartRateCalculator.kt` | `js/heartrate.js` |

Video tidak pernah keluar dari perangkat. Tidak ada proses build: hanya ES module biasa, dengan
library dari jsDelivr (versinya dikunci di `js/config.js`).

## Menjalankan secara lokal

Dari folder `web/`:

```bash
python verify/serve.py
```

Buka http://localhost:8765 (tambahkan `?debug` untuk melihat PNR, band ratio, fps, dan waktu
inferensi). Kamera hanya dapat diakses melalui `localhost` atau HTTPS, tidak melalui IP jaringan
lokal (LAN).

Untuk mencoba di ponsel Android melalui USB, biarkan server tetap berjalan, lalu jalankan:

```bash
adb reverse tcp:8765 tcp:8765
```

kemudian buka http://localhost:8765 di Chrome pada ponsel.

## Deploy

`.github/workflows/pages.yml` memublikasikan `index.html`, `css/`, `js/`, dan `models/` ke
GitHub Pages setiap kali ada push ke `main` yang mengubah folder `web/`. GitHub Pages
menggunakan HTTPS, sehingga kamera dapat diakses. Pengaturan satu kali: Settings → Pages →
Source: **GitHub Actions**.

Tautan "Unduh APK" (`APK_URL` di `js/config.js`) mengarah ke
`releases/latest/download/LitePulse.apk`, sehingga selalu mengunduh file bernama persis
`LitePulse.apk` dari GitHub Release terbaru. Lihat README utama untuk cara membuat rilis.

## Membuat ulang model

Dari folder `model/` (checkpoint berada di `model/pth_model/`):

```bash
pip install onnx onnxruntime
python litepulse_onnx_generator.py
```

Salin `litepulse.onnx` yang dihasilkan ke `web/models/`. Skrip akan berhenti dengan error jika
file yang disimpan tidak sama dengan model PyTorch.

`models/face_landmarker.task` adalah file yang sama dengan
`android/app/src/main/assets/face_landmarker.task`.

## Verifikasi terhadap Python

`verify/make_reference.py` menghitung output referensi dengan cara yang sama seperti saat
pelatihan (masker ROI cv2, normalisasi rPPG-Toolbox, checkpoint `.pth`, filter scipy).
`verify/verify.mjs` menjalankan modul-modul milik halaman ini dan file `.onnx` pada input yang
sama, lalu membandingkan hasilnya.

Dari folder `web/`:

```bash
python verify/make_reference.py verify/out
npm install
node verify/verify.mjs verify/out
```

Hasil yang diharapkan: `24/24 checks passed` (ROI dalam selisih ±1 tingkat keabuan dari cv2,
preprocessing dalam 1e-4, output model dalam 1e-5 dari `.pth`, dan detak jantung identik).

`verify/e2e.html` menjalankan seluruh pipeline halaman (MediaPipe, ROI, worker, perhitungan
detak jantung) pada foto diam yang diberi sinyal denyut sintetis dengan frekuensi yang
diketahui, tanpa memerlukan kamera. Simpan foto wajah Anda sendiri di `verify/tmp/face.png`
(diabaikan oleh git), jalankan server, lalu buka
http://localhost:8765/verify/e2e.html?bpm=60,75,90,110.

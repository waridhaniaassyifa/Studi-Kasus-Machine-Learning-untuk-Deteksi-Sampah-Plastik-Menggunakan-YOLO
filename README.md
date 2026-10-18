# Studi Kasus Machine Learning: Deteksi Sampah Menggunakan YOLO

## 1. Deskripsi Proyek

Proyek ini merupakan studi kasus Machine Learning di bidang *Computer Vision* yang bertujuan untuk mendeteksi dan mengidentifikasi jenis sampah secara otomatis melalui gambar menggunakan model YOLO (*You Only Look Once*).

Sistem memanfaatkan model deteksi objek yang telah dilatih (*pre-trained/fine-tuned model*) untuk mengenali beberapa kategori sampah, seperti botol plastik, kantong plastik, kaca, kertas, dan kaleng. Pengguna dapat menguji model menggunakan gambar sendiri maupun gambar yang tersedia dalam dataset.

Proyek ini dikembangkan sebagai sarana pembelajaran untuk memahami penerapan Machine Learning dalam mendukung identifikasi sampah secara visual.

## 2. Latar Belakang

Sampah merupakan salah satu permasalahan lingkungan yang membutuhkan penanganan secara tepat. Proses identifikasi sampah secara manual dapat memerlukan waktu dan tenaga, terutama ketika jumlah sampah cukup banyak.

Penerapan Machine Learning, khususnya *Object Detection*, dapat membantu mengenali objek sampah berdasarkan karakteristik visualnya. Melalui model YOLO, sistem dapat mendeteksi lokasi objek dalam gambar sekaligus memprediksi kelas objek tersebut.

Studi kasus ini berfokus pada penggunaan model deteksi sampah untuk memahami bagaimana teknologi Computer Vision dapat diterapkan dalam permasalahan lingkungan.

## 3. Tujuan

Tujuan dari proyek ini adalah:

- Memahami penerapan Machine Learning untuk deteksi objek.
- Mengimplementasikan model YOLO untuk mendeteksi sampah pada gambar.
- Mengenali kategori sampah berdasarkan hasil prediksi model.
- Menampilkan lokasi objek yang terdeteksi menggunakan *bounding box*.
- Mengevaluasi kemampuan model melalui pengujian menggunakan gambar baru.

## 4. Teknologi yang Digunakan

- **Python** — bahasa pemrograman utama.
- **YOLO** — model untuk deteksi objek.
- **Ultralytics** — pustaka untuk menjalankan model YOLO.
- **OpenCV** — untuk membaca dan memproses gambar jika diperlukan.
- **Visual Studio Code** — lingkungan pengembangan.
- **Git dan GitHub** — untuk pengelolaan versi dan publikasi proyek.

## 5. Kategori Objek

Model referensi dirancang untuk mendeteksi enam kategori objek berikut:

| No. | Kategori |
|---|---|
| 1 | Garbage Bag |
| 2 | Glass |
| 3 | Paper Bag |
| 4 | Pet Bottle |
| 5 | Plastic Bag |
| 6 | Can |

Kategori yang berhasil dideteksi bergantung pada kelas yang tersedia pada model dan hasil pelatihannya.

## 6. Dataset dan Model

Proyek ini menggunakan model deteksi sampah yang berasal dari proyek referensi berikut:

- **Repository referensi:** https://github.com/laibyy0/waste-detection-yolo
- **Dataset:** https://universe.roboflow.com/object-detection-cfmul/waste-detection-0momv/dataset/1

Model yang digunakan pada proyek referensi adalah YOLO26n. Pastikan kembali file bobot model (`best.pt`) dan konfigurasi kelas tersedia sebelum menjalankan program.

Dataset dan model mengikuti ketentuan lisensi serta penggunaan dari masing-masing sumber. Periksa lisensi sumber sebelum mendistribusikan ulang file model atau dataset.

## 7. Persyaratan Sistem

Pastikan perangkat telah memiliki:

- Python versi yang kompatibel dengan pustaka Ultralytics.
- Visual Studio Code.
- Pustaka Python yang tercantum dalam `requirements.txt`, jika tersedia.
- File bobot model yang diperlukan, misalnya `best.pt`.
- Gambar yang akan digunakan untuk pengujian.

## 8. Instalasi dan Cara Menjalankan

### Langkah 1 — Clone Repository

Buka terminal di Visual Studio Code, kemudian jalankan:

```bash
git clone URL_REPOSITORY_KAMU
cd NAMA_FOLDER_PROYEK
```

Ganti `URL_REPOSITORY_KAMU` dan `NAMA_FOLDER_PROYEK` sesuai repository milikmu.

### Langkah 2 — Buat Virtual Environment

```bash
python -m venv .venv
```

Aktifkan lingkungan virtual.

**Windows:**

```powershell
.venv\Scripts\Activate.ps1
```

Jika menggunakan Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### Langkah 3 — Instal Dependensi

Jika proyek memiliki file `requirements.txt`, jalankan:

```bash
pip install -r requirements.txt
```

Jika belum tersedia dan kode menggunakan Ultralytics, instal pustaka yang diperlukan:

```bash
pip install ultralytics opencv-python
```

### Langkah 4 — Siapkan Model

Letakkan file bobot model pada lokasi yang sesuai dengan kode program, misalnya:

```text
models/
└── best.pt
```

Nama dan lokasi file harus disesuaikan dengan konfigurasi pada program.

### Langkah 5 — Jalankan Deteksi

Jalankan skrip prediksi yang tersedia di repository. Contohnya, jika file program bernama `predict.py`:

```bash
python predict.py
```

Jika skrip berada di dalam folder `notebooks`, sesuaikan perintah dengan lokasi file dan cara pemanggilan yang digunakan oleh proyek tersebut.

Ikuti instruksi pada program untuk memilih gambar dan melihat hasil deteksi.

## 9. Alur Kerja Sistem

1. **Input gambar:** pengguna menyediakan gambar yang mengandung objek sampah.
2. **Pemrosesan gambar:** gambar dibaca dan disiapkan untuk model.
3. **Deteksi objek:** model YOLO memprediksi kelas dan lokasi objek.
4. **Visualisasi:** sistem menampilkan *bounding box*, label kelas, dan skor kepercayaan (*confidence score*).
5. **Evaluasi:** hasil prediksi dapat diperiksa untuk mengetahui objek yang terdeteksi dan objek yang mungkin terlewat.

## 10. Hasil yang Diharapkan

Sistem diharapkan mampu menampilkan objek sampah yang dikenali oleh model, lengkap dengan kotak pembatas dan label kategori. Hasil aktual dapat berbeda berdasarkan kualitas gambar, kondisi pencahayaan, sudut pengambilan gambar, ukuran objek, dan kemiripan objek dengan latar belakang.

Kemampuan model pada gambar pantai atau kondisi lingkungan yang berbeda perlu diuji secara langsung. Hasil deteksi pada dataset referensi tidak otomatis menjamin tingkat akurasi yang sama pada gambar baru.

## 11. Batasan Studi Kasus

- Sistem menggunakan model yang telah tersedia dan bukan merancang arsitektur YOLO baru.
- Hasil deteksi bergantung pada kategori yang dipelajari model.
- Model dapat menghasilkan deteksi yang keliru atau tidak mendeteksi objek tertentu.
- Pengujian pada kondisi lingkungan yang berbeda diperlukan untuk mengetahui keterbatasan model.
- Proyek ini merupakan demonstrasi pembelajaran dan belum dapat dianggap sebagai sistem pengelolaan sampah yang siap digunakan secara operasional.

## 12. Pengembangan Selanjutnya

Pengembangan proyek dapat dilakukan dengan menguji model pada gambar sampah di lingkungan nyata, membandingkan hasil prediksi pada latar belakang berbeda, serta mengevaluasi kesalahan deteksi seperti *False Positive* dan *False Negative*.

Pengembangan tersebut dapat membantu memahami keandalan model Computer Vision ketika menghadapi variasi objek dan kondisi lingkungan.

## 13. Sumber dan Atribusi

Proyek ini dibuat untuk tujuan pembelajaran Machine Learning dengan mengacu pada proyek referensi [Waste Detection YOLO](https://github.com/laibyy0/waste-detection-yolo).

Model, dataset, dan kode yang berasal dari proyek referensi tetap menjadi bagian dari sumber aslinya dan tunduk pada lisensi atau ketentuan penggunaan masing-masing. Perubahan yang dilakukan dalam repository ini sebaiknya dijelaskan secara transparan agar perbedaan antara materi referensi dan hasil pengembangan sendiri dapat diketahui.

---

**Catatan:** README ini merupakan dokumentasi studi kasus. Sesuaikan nama file, struktur folder, perintah eksekusi, serta hasil pengujian dengan implementasi yang benar-benar ada di repository.

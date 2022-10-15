# Laporan Proyek *Machine Learning* Terapan by Dicoding- Rizki Tetania 

## Deteksi Penyakit Liver Menggunakan Pendekatan Model *Machine Learning* (*Predictive Analytics* - Regresi)

![gambar hati](https://user-images.githubusercontent.com/88262711/195909494-47ed0c76-3dcd-496e-95c4-502c800ce578.jpg)


Penyakit Liver merupakan suatu istilah yang digunakan untuk gangguan pada liver atau hati yang menyebabkan organ tersebut tidak dapat berfungsi dengan baik [[1]](https://www.alodokter.com/penyakit-liver). Berdasarkan penelitian yang dilakukan oleh British Liver Trust memberikan informasi bahwa penyakit hati atau Liver merupakan penyebab kematian terbesar pada orang yang berusia antara 35-49 tahun, khususnya di Inggris. Penelitian tersebut juga mengungkapkan bahwa penyakit hati diperkirakan akan menggeser penyakit jantung sebagai penyebab terbesar kematian dini dalam beberapa tahun mendatang [[2]](https://litbang.kemendagri.go.id/website/liver-disebut-penyebab-kematian-terbesar-di-usia-35-49-tahun/). Seseorang sering tidak menyadari atau terlambat mengetahui penyakit Liver sehingga ketika diperiksa penyakit Liver sudah parah. Penyakit Liver sering dianggap sebagai *silent killer* (pembunuh diam-diam) karena adanya kemungkinan tidak timbul gejala. **Permasalahannya, penanganan pasien dengan penyakit hati pada tahap awal akan memperpanjang hidup pasien, namun sayangnya semua ahli kesehatan tidak memiliki keahlian khusus dalam mendiagnosis medis**. Sehingga penerapan *Artificial Intelegence in Medicine* (AIM) sangat diperlukan dalam mempermudah Dokter untuk mendeteksi penyakit Liver sejak awal.
Penelitian dengan judul *"Prediksi Penyakit Liver Dengan Menggunakan Metode Naive Bayes Dan K-Nearest Neighbour (KNN)"* memberikan kesimpulan bahwa algoritma terbaik dalam penentuan identifikasi penyakit Liver yaitu K-Nearest Neighbour (KNN) jika dibandingkan dengan Naive Bayes, dimana kedua model tersebut dievaluasi menggunakan hasil *Confusion Matrix* [[3]](https://jurnal.tau.ac.id/index.php/snartek/article/view/101/69). Sedangkan penelitian yang dilakukan oleh Pusporani, Qomariyah dan Irhamah mengenai klasifikasi pasien penderita penyakit Liver. Penelitian tersebut memberikan informasi bahwa metode *Machine Learning* terbaik untuk mengklasifikasikan pasien yaitu, berdasarkan nilai akurasi dan presisi maka metode SVM memberikan hasil yang terbaik dalam mengklasifikasikan, sedangkan apabila berdasarkan recall maka *K-Nearest Neighbour* memberikan hasil yang terbaik [[4]](https://media.neliti.com/media/publications/323508-klasifikasi-pasien-penderita-penyakit-li-496b23e3.pdf). Sehingga pada projek ini, akan berfokus dalam menemukan model *Machine Learning* terbaik untuk mendetaksi penyakit Liver, sehingga penyakit tersebut dapat dideteksi sejak dini.

## *Business Understanding*

Seperti yang sudah dijelaskan sebelumnya bahwa tingkat kematian yang disebabkan oleh penyakit Liver semakin bertambahnya tahun semakin tinggi, sehingga dengan adanya hal tersebut diperlukan suatu alat yang dapat mendeteksi penyakit Liver sejak dini, agar pertumbuhan nya dapat diatasi dan dicegah. Oleh karena itu, penting bagi dunia Kedokteran untuk mengetahui dan mendeteksi penyakit Liver pada pasien sejak dini. Pendeteksian tersebut akan digunakan untuk menentukan faktor-faktor apa yang cukup berpengaruh dalam menyebabkan seseorang menderita penyakit Liver.

### *Problem Statement*

Berdasarkan latar belakang yang diuraikan sebelumnya, maka projek ini dikembangkan untuk menjawab permasalahan berikut.
- Dari serangkaian faktor atau variabel yang ada, maka adakah faktor yang berhubungan satu sama lain yang menjadi penyebab penyakit Liver?
- Model apa yang paling tepat untuk mendeteksi penyakit Liver sejak dini?

### *Goals*

Jawaban dari permasalahan diatas, dapat dijelaskan sebagai berikut.
- Mengetahui faktor yang saling berkorelasi dalam menyebabkan seseorang menderita penyakit Liver.
- Membuat model *Machine Learning* yang dapat mendeteksi penyakit Liver sedini mungkin berdasarkan faktor atau variabel-variabel yang ada.
    
### *Solution Statement*

Berikut merupakan solusi untuk Goals atau tujuan yang ingin dicapai pada projek.
- Melakukan *Exploratory Data Analysis* (EDA) menggunakan library *corr()* untuk mengetahui korelasi antar varriabel.
- Melakukan perbandingan model yang terbentuk berdasarkan empat metode regresi yaitu *K-Nearest Neighbour (KNN)*, *Support Vector Machine (SVM)*, *Logistic Regression*, dan *Random Forest* (RF).

## *Data Understanding*

Data yang digunakan pada projek yaitu mengenai data pasien penderita peyakit Liver yang terjadi di India ( [*Indian Liver Patient Records*](https://www.kaggle.com/datasets/uciml/indian-liver-patient-records?resource=download) ). Peningkatan jumlah pasien penderita penyakit Liver disebabkan oleh fakor alcohol, rokok, narkoba dan virus. Dataset tersebut digunakan untuk mengevaluasi algoritma yang dapat memprediksi penyakit Liver, sehingga dapat membantu kinerja kedokteran untuk menghambat pertumbuhan penderita penyakit Liver. Berdasarkan data 583 pasien, 416 pasien terindikasi menderita penyakit Liver, sedangkan 167 pasien lainnya tidak terindikasi menderita penyakit Liver. 

### Variabel Projek

Variabel-variabel pada *Indian Liver Patient* dataset dapat dijelaskan sebagai berikut.

Tabel 1. Variabel Projek

| Variabel | Keterangan |
| ---------- | -------------- |
| *Age* | Usia pasien |
| *Gender* | Jenis kelamin pasien |
| *Total Bilirubin* | Jumlah sel darah merah dihati (mg/dl) | 
| *Direct Bilirubin* | Bilirubin bebas (mg/dl) |
| *Alkaline Phosphotase* | Enzim yang terkandung dari usus |
| *Alamine Aminotransferase* | Enzim yang terkandung didalam hati |
| *Aspartate Aminotransferase* | Enzim protein yang berada didalam hati |
| *Total Protiens* | Serum protein yang terdapat didalam hati (g/dl) |
| *Albumin* | Sintesa protein didalam hati |
| *Albumin and Globulin Ratio* | Menunjukkan perbandingan rasio albumin dan globulin didalam hati |
| *Dataset / Class* | Kategori kelas pasien yang menderita dan tidak menderita penyakit Liver |

Berdasarkan sebelas variabel yang terdapat pada dataset tersebut, terdapat empat variabel penting yang menentukan pasien menderita penyakit Liver yaitu, peningkatan *Total bilirubin*, peningkatan *Alamine Aminotranferase*, peningkatan *Aspartate Aminotransferase*, dan penurunan kadar *Albumin*.

### *Exploratory Data Analysis* (EDA)

Tahapan ini bertujuan untuk memahami data, melalui diskripsi data maupun secara visualisasi. Tahapan EDA pada projek ini dijelaskan sebagai berikut.

- EDA-Deskripsi Variabel

Pada saat mendiskripsikan data menggunakan <data.describe()> menunjukkan bahwa rata-rata usia pasien terindikasi menderita penyakiti Liver yaitu 44 tahun, seperti yang dijelaskan pada latar belakang bahwa usia tersebut merupakan rentang usia yang memiliki peluang terbesar menyebabkan kematian pada penderita penyakit Liver. Kemudian pada variabel kadar *Albumin* memiliki nilai tertinggi sebesar 5,50 mg/dl, dimana kisaran normal *Albumin* adalah 0 sampai 8 mg/dl, sehingga nilai tersebut masih dalam rentang kadar *Albumin* yang normal.

- EDA-Menangani *Missing value* dan *Outliers*

Deteksi *missing value* dapat mengggunakan fungsi yang ada pada library Pandas yaitu .isnull() atau .isna(), dengan menggunakan fungsi tersebut dapat membantu kita untuk mengetahui data yang hilang dari dataset. Berdasarkan fungsi tersebut, memberikan informasi bahwa hanya satu variabel dari sebelas variabel yang memiliki nilai yang hilang, yaitu variabel *Albumin and Globulin Ratio* sebanyak empat data. Terdapat tiga cara untuk mengatasi *missing value* yaitu dibiarkan, dihilangkan dan mensubtitusi nilai yang hilang menggunakan nilai mean / median / modus. Cara yang digunakan untuk mengatasi *missing value* pada projek ini yaitu dengan cara mensubtitusikan nilai *mean* variabel tersebut kedalam data yang memiliki nilai hilang. Sehingga jumlah data pada variabel *Albumin and Globulin Ratio* yaitu sebesar 583 data seperti jumlah data pada varaibel yang lain.

![Grafik Boxplot Data Kontinu](https://user-images.githubusercontent.com/88262711/195895033-56ac6c33-314f-43eb-8d11-87ba0e30bace.png)

Gambar 1. Grafik *Boxplot* Data Kontinu

Gambar 1 tersebut menunjukkan bahwa 7 dari 9 variabel kontinu memiliki *data outlier*, sehingga untuk mengatasi hal tersebut akan diatasi menggunakan metode IQR. Data yang berada dibawah nilai Q3 dan Q1 akan dianggap sebagai *data outlier* dan akan dihapus. Sehingga jumlah data yang sebelumnya sebesar 583 data, berubah menjadi 440 data. 

- EDA- *Univariate Analysis*

![Gambar Proporsi kategori pasien](https://user-images.githubusercontent.com/88262711/195895238-23cbf899-6f9c-46af-b129-ebcc0c83b574.png)

Gambar 2. Proporsi kategori pasien

Gambar 2 tersebut memberikan informasi bahwa terdapat 63,3% pasien terindikasi menderita penyakit Liver, sedangkan 36,7% lainnya tidak terindikasi menderita penyakit Liver. 

- EDA- *Multivariate Analysis* 

![Grafik plot Korelasi Data Kontinu](https://user-images.githubusercontent.com/88262711/195895370-1b95e834-9f0e-4b90-9a4a-d34e7ae6c069.png)

Gambar 3. Grafik plot Korelasi Data Kontinu

Informasi yang diperoleh berdasarkan Gambar 3 tersebut yaitu pada variabel *Total Bilirubin* dan *Direct Bilirubin* memiliki nilai korelasi rank spearman yang cukup tinggi sebesar 0,97, yang berarti antar kedua variabel tersebut memiliki hubungan yang cukup positif, sehingga apabila kadar *Total Bilirubin* mengalami kenaikan maka kadar *Direct Bilirubin* juga akan mengalami hal tersebut.

## *Data Preparation*

Data *preparation* atau biasa disebut sebagai tahapan data *preprocessing* merupakan tahapan yanng dilakukan karena dapat memberikan fungsi atau manfaat pada data mining. Proses ini utamanya dilakukan untuk memastikan kualitas data baik sebelum digunakan saat analisis data. Terdapat empat tahap yang digunakan dalam proses *preprocessing* projek, yaitu ;

### *Label Encoding*

Tahapan ini bertujuan untuk merubah data kategorik menjadi data numerik, dimana library Python yang paling umum digunakan adalah Scikit-Learn. Terdapat dua variabel yang bertipe data kategori yaitu variabel Gender dan Class. Kedua variabel tersebut akan dirubah menjadi numerik menggunakan fungsi LabelEncoder yang terdapat pada *library* <sklearn.preprocessing>. Setelah fungsi LabelEncoder dibentuk, maka kemudian lakukan proses fit_transform() dengan kolom variabel yang akan dirubah.

### Reduksi Dimensi dengan *Principal Component Analysis* (PCA)

[*Principal Component Analysis* (PCA)](https://dqlab.id/analisis-pca-sederhanakan-data-dengan-reduksi-dimensi-menggunakan-r) adalah salah satu metode reduksi dimensi pada *Machine Learning*. PCA akan memilih variabel-variabel yang mampu menjelaskan sebagian besar variabilitas data. PCA mengurangi dimensi dengan membentuk variabel-variabel baru yang disebut *Principal Components*. *Principal Components* yang merupakan kombinasi linier dari variabel-variabel lama. Penghitungan Varians dan *Principal Component* ini dapat dilakukan dengan menggunakan konsep nilai eigen (*eigenvalue*) dan vektor eigen (*eigenvector*) dari ilmu Aljabar Linier.

---
Manfaat dari penggunaan PCA yaitu; mengatasi multikolinearitas; mereduksi jumlah variabel yang akan dimasukkan kedalam model; jumlah variabel yang lebih sedikit tenu akan menyederhanakan model; dan juga mempercepat komputasi.

---

Penggunaan PCA pada projek ini, digunakan untuk mereduksi variabel *Total Bilirubin* dan *Direct Bilirubin* dibentuk menjadi satu dimensi, karena kedua variabel memiliki nilai korelasi yang cukup tinggi. Sehingga mereduksinya menjadi satu variabel atau dimensi, dapat mambantu mempercepat proses 
komputasi dan yang lainnya.

## Data *Training* dan Data *Testing*

[*Training* atau *Testing Split*](https://ilmudatapy.com/evaluasi-model-machine-learning-dengan-train-test-split/) merupakan salah satu metode yang digunakan untuk mengevaluasi peforma model *Machine Learning*. Metode evaluasi model ini membagi dataset menjadi dua bagian yaitu bagian yang digunakan untuk *training* data dan bagian untuk *testing* data dengan proporsi tertentu.
**Train data** digunakan untuk *fit* model *Machine Learning*, sedangkan *test data* digunakan untuk mengevaluasi hasil *fit* model tersebut. Metode train/test split ini akan memberikan hasil prediksi yang lebih akurat untuk data baru atau data yang belum pernah di- *train*. Karena data *testing* tidak digunakan untuk melatih model, maka model tidak mengetahui *outcome* dari data tersebut. Pada projek ini, proporsi pembagian data *training* : *testing* sebesar 90:10.

### Standarisasi

[Standarisasi](https://anzihory.medium.com/normalisasi-vs-standarisasi-101093633e18) adalah teknik lain dalam melakukan perubahan skala, dimana data yang dimiliki akan diubah sehingga memiliki nilai rata-rata sama dengan nol (terpusat) dan standar deviasi sama dengan satu. Proses standarisasi pada Python, dapat dilakukan dengan cara meng- *import* StandardScaler dari sklearn.preprocessing.

## *Modeling* 
Tahapan ini membahas mengenai model *Machine Learning* yang diguakan untuk menyelesaikan permasalahan. Terdapat empat macam algoritma yang digunakan yaitu *K-Nearest Neighbour (KNN)*, *Support Vector Machine (SVM)*, *Logistic Regression*, dan *Random Forest* (RF). Penjelasan ke-empat algoritma tersebut adalah sebagai berikut.

### *K-Nearest Neighbour* (KNN)
[*K-Nearest Neighbors atau KNN*](https://medium.com/@annisaayunda/knn-k-nearest-neighbors-with-forestfires-dataset-c810e603869d) adalah algoritma yang berfungsi untuk melakukan klasifikasi suatu data berdasarkan data pembelajaran (train data sets), yang diambil dari k tetangga terdekatnya ( *nearest neighbors* ). Dengan k merupakan banyaknya tetangga terdekat.
KNN digunakan untuk klasifikasi dan regresi. KNN hanya sebuah perkirakan, dan semua perhitungan ditunda sampai klasifikasi. Sebuah bobot dapat menetapkan apakah tetangga dekat lebih berpengaruh daripada tetangga yang lebih jauh. Kelebihan metode KNN yaitu mudah diterapkan dan tidak perlu membuat asumsi data sebelumnya. Sedangkan kekurangan metode ini yaitu membutuhkan cukup banyak waktu untuk melakukan prediksi, karena menghitung selisih setiap titik data. Tahapan Langkah algoritma metode [KNN](https://medium.com/@aida.mahmudah171/melakukan-prediksi-kelulusan-dengan-knn-di-jupyter-notebook-4c7c707acd2c) :
1. Menentukan parameter k (jumlah tetangga paling dekat).
2. Menghitung kuadrat jarak eucliden objek terhadap data training yang diberikan.
3. Mengurutkan hasil no 2 secara ascending (berurutan dari nilai tinggi ke rendah)
4. Mengumpulkan kategori Y (Klasifikasi nearest neighbor berdasarkan nilai k)
5. Dengan menggunakan kategori nearest neighbor yang paling mayoritas maka dapat dipredisikan kategori objek.

### *Support Vector Machine* (SVM)
[*Support Vector Machine*](https://iansuryap.medium.com/classification-with-support-vector-machine-svm-methode-7ad33d8951b3) adalah suatu teknik untuk melakukan prediksi, baik dalam hal kasus klasifikasi maupun regresi. Metode SVM berada dalam satu kelas dengan *Artificial Neural Network* (ANN) dalam hal fungsi dan kondisi permalasahan yang bisa diselesaikan. Teknik SVM yang digunakan untuk menjawab permasalahan yaitu menggunakan *Support Vector Regression) (SVR).
[*Support Vector Regression*](https://medium.com/@nurfauziah_uci/support-vector-regression-pada-harga-saham-netflix-dengan-python-2cc9deb169da) adalah salah satu metode regresi dengan menggunakan *Machine Learning* yang sangat populer. Konsep dari SVR adalh membuat sebuah *hyperplane* yang mendekati titik-titik data yang akan diperediksi, sehingga diperoleh estimasi yang memiliki nilai error yang sangat kecil.

---
Model yang dibangun beral dari sklearn.svm, dimana untuk jenis SVM yang digunakan pada projek ini yaitu *Support Vector Regression* (SVR), sehingga library yang di- *import* ialah SVR. Sedangkan untuk ketentuan kernel yang digunakan pada projek ini yaitu kernel linear, karena dinilai dapat membentuk hasil regresi dengan error yang kecil.

---

### *Logistic Regression*
[*Logistic Regression*](https://medium.com/@rismitawahyu/comparing-analysislogistic-regression-k-nearest-neighbors-k-nn-and-support-vector-machine-svm-67a5d0cc4091) adalah salah satu metode statistika yang menggambarkan hubungan antara variabel respon (y) dengan satu atau lebih variabel prediktor (x), dimana variabel respon dalam regresi logistik adalah biner atau dikotomi yaitu hanya memiliki dua kategori. Hasil untuk setiap pengamatan dapat diklasifikasikan sebagai “sukses” atau “gagal”. Klasifikasi ini diwakili dengan y = 1 untuk hasil pengamatan “sukses” dan y = 0 untuk hasil pengamatan “gagal”. Regresi logistik adalah cara statistik yang kuat dari pemodelan hasil binomial dengan satu atau lebih variabel penjelas. 
Metode ini bekerja dengan cara mengukur hubungan antara variabel target (yang ingin diprediksi) dan variabel input (fitur yang digunakan) dengan fungsi logistik. Probabilitas akan dihitung menggunakan fungsi sigmoid untuk mengubah nilai-nilai tadi menjadi 0 atau 1.

---
Library yang digunakan membangun model berasal dari sklearn.linear_model untuk meng- *import* library LogisticRegression, dimana tidak ada nilai parameter khusus yang digunakan. Model tersebut akan di evaluasi berdasarkan hasil *mean squared error*.

---

### *Random Forest*
Algoritma [*Random Forest*](https://algorit.ma/blog/cara-kerja-algoritma-random-forest-2022/) disebut sebagai salah satu algoritma *Machine Learning* terbaik, sama seperti Naïve Bayes dan Neural Network. Random Forest adalah kumpulan dari decision tree atau pohon keputusan. Algoritma ini merupakan kombinasi masing-masing tree dari decision tree yang kemudian digabungkan menjadi satu model. Biasanya, Random Forest dipakai untuk masalah regresi dan klasifikasi dengan kumpulan data yang berukuran besar. 
*Random Forest* bekerja dengan membangun beberapa decision tree dan menggabungkannya demi mendapatkan prediksi yang lebih stabil dan akurat. ‘Hutan’ yang dibangun oleh Random Forest adalah kumpulan *decision tree* di mana biasanya dilatih dengan metode *bagging*. Ide umum dari metode bagging adalah kombinasi model pembelajaran untuk meningkatkan hasil keseluruhan
Algoritma *Random Forest* meningkatkan keacakan pada model sambil menumbuhkan *tree*. Alih-alih mencari fitur yang paling penting saat memisahkan sebuah node, Random Forest mencari fitur terbaik di antara subset fitur yang acak. Alhasil, cara ini menghasilkan keragaman yang luas dan umumnya menghasilkan model yang lebih baik.

---
Algoritma yang digunakan untuk membenuk model *Random Forest* yaitu dengan menggunakan library sklearn.ensemble untuk mng- *import* library RandomForestRegressor. Sedangkan untuk nilai parameter yang digunakan adalah sebagai berikut. 

- n_estimator: jumlah trees (pohon) di forest. Di sini kita set n_estimator=50.
- max_depth: kedalaman atau panjang pohon. Ia merupakan ukuran seberapa banyak pohon dapat membelah (splitting) untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan.
- random_state: digunakan untuk mengontrol random number generator yang digunakan. 
- n_jobs: jumlah job (pekerjaan) yang digunakan secara paralel. Ia merupakan komponen untuk mengontrol thread atau proses yang berjalan secara paralel. n_jobs=-1 artinya semua proses berjalan secara paralel.

---

**Berdasarkan empat model yang terbentuk, maka untuk pemilihan model terbaik dapat ditentukan berdasarakan hasil evaluasi, dimana model tersebut dapat memprediksi atau mendeteksi penyakit pasien secara tepat dan memiliki nilai error yang kecil.**

## *Evaluation*
[Metrik evaluasi](https://learn.microsoft.com/id-id/azure/machine-learning/component-reference/evaluate-model) yang ditampilkan untuk model regresi dirancang untuk memperkirakan jumlah kesalahan. Model dianggap cocok dengan data dengan baik jika perbedaan antara nilai yang diamati dan diprediksi kecil. Namun, melihat pola residu (perbedaan antara satu titik prediksi dan nilai aktualnya yang sesuai) dapat memberi tahu tentang potensi bias dalam model. Metrik yang digunakan pada projek ini yaitu *Mean Square Error* (MSE), adalah rata-rata perbedaan kuadrat antara nilai prediksi dan aktual. Ini digunakan untuk regresi. Nilai MSE selalu positif. Semakin baik model dalam memprediksi nilai aktual, semakin kecil nilai MSE. Formulasi perhitungan MSE dapat dijelaskan pada persamaan berikut.

![mse](https://user-images.githubusercontent.com/88262711/195906174-0257deb8-0fab-4f64-af01-7509cf371c2c.jpeg)

Dengan N sebagai jumlah dataset, yi sebagai nilai sebenarnya dan y_pred sebagai nilai prediksi. 

![Grafik metrik evaluasi](https://user-images.githubusercontent.com/88262711/195895537-c3b15e60-935e-4875-adfc-cace1d7804dd.png)

Gambar 4. Grafik Metrik Evaluasi

Gambar 4 memberikan informasi bahwa model yang terbentuk dengan menggunakan logaritma Random Forest memberikan nilai error yang paling kecil dibandingkan dengan metode yang lain. Sehingga model inilah yang dipillih sebagai model terbaik untuk melakukan deteksi penyakit Liver sejak dini. 

## *Conclussion*

Kesimpulan yang diperoleh melalui projek ini yaitu sebagai berikut.

1. Terdapat 3 pasang variabel yang memiliki korelasi yang cukup tinggi. Variabel tersebut teridiri dari; *Total Bilirubin* dan *Direct Bilirubin*; *Alamine Aminotransferase* dan *Aspartate Aminotransferase*; dan yang terakhir yaitu antara variabel *Albumin* dan *Albumin and Globulin Ratio*. Berdasarkan 3 pasang varaibel tersebut, antara variabel *Total Bilirubin* dan *Direct Bilirubin* memiliiki nilai korelasi yang cukup tinggi sebesar 0,97, dimana korelasi yang cukup tinggi antara variabel Independen mengindikasikan adanya Multikolinearitas. Sehingga untuk tahap selanjutnya, dilakukan reduksi dimensi menggunakan PCA agar kedua variabel tersebut  dapat membentuk satu dimensi.

2. Terdapat 4 model *Machine Learning* yang terbentuk, dimana model yang memiliki nilai error terkecil yaitu model dari algoritma *Random Forest*. Sehingga untuk dapat mendeteksi penyakit Liver, agar dapat dittanggulangi sejak dini yaitu menggunakan model yang terbentuk dari *Random Forest*.

Saran yang diberikan berdasarkan pengerjaan projek yaitu dapat menambah jumlah data pasien penyakit Liver agar model yang terbentuk semakin baik. Serta selain menggunakan metode regresi, untuk memprediksi penderita penyakit Liver dapat menggunakan metode yang lain seperti klasifikasi, peramalan dan yang lainnya.


## Referensi

[[1]](https://www.alodokter.com/penyakit-liver) Alodokter. (2022). *Penyakit Liver*. Diakses pada 11 Oktober 2022 https://www.alodokter.com/penyakit-liver

[[2]](https://litbang.kemendagri.go.id/website/liver-disebut-penyebab-kematian-terbesar-di-usia-35-49-tahun/) Kemendagri. (2019). *Liver Disebut Penyebab Kematian Terbesar di Usia 35-49 Tahun)*. Diakses pada 11 Oktober 2022 https://litbang.kemendagri.go.id/website/liver-disebut-penyebab-kematian-terbesar-di-usia-35-49-tahun/

[[3]](https://jurnal.tau.ac.id/index.php/snartek/article/view/101/69) Noviarindini. (2019). *Prediksi Penyakit Liver Dengan Menggunakan Metode Naive Bayes Dan K-Nearest Neighbour (KNN)*. Prosiding TAU SNAR-TEK 2019 Seminar Nasional Rekayasa dan Teknologi ISSN:2715-6982

[[4]](https://media.neliti.com/media/publications/323508-klasifikasi-pasien-penderita-penyakit-li-496b23e3.pdf) Pusporani. (2019). *Klasifikasi Pasien Penderita Penyakit Liver dengan Pendekatan *Machine Learning**. INFERENSI, Vol. 2(1), March 2019, ISSN: 0216-308X hal 25-32. 

**--Ini adalah bagian akhir laporan--**

Berikut merupakan penilaian yang diberikan oleh *reviewer* dari keseluruhan elemen projek.
![Submission 1 (Bintang 5)](https://user-images.githubusercontent.com/88262711/195965265-37249a9a-e60c-49f2-9ce3-4bdf3169c16d.PNG)

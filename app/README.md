# Store3D — コンビニ棚 3D スキャナ (Pixel 8a)

店内を歩きながら ARCore で壁・棚面・水平面（冷凍ケース上面）を記録し、
カメラ画像から棚の境界と棚の種類を認識して、ノイズの少ない箱モデルの店舗 3D を作る Android アプリ。

## 使い方
1. Pixel 8a に `app/build/outputs/apk/debug/app-debug.apk` を入れる（ARCore 必須）。
2. 棚の正面にカメラを向け、通路をゆっくり歩く。認識済みの棚が半透明の箱でライブ表示される。
3. 「完了して3D化」で `Android/data/com.example.store3d/files/scan_<時刻>/` に保存し、3D ビューアを開く
   （ドラッグで回転、ピンチでズーム）。
   - `raw.json` 平面と認識レイの生データ（再構成をやり直せる）
   - `model.json` 箱モデル、`model.obj` / `model.mtl` 他ツール用

## 処理の流れ
| 段階 | 実装 |
|---|---|
| 位置記録 | ARCore 平面検出（垂直＋水平）、0.5 秒ごとにカメラ画像・姿勢・平面をスナップショット |
| 棚の境界 | `shelf_boundary.tflite`：画像の列ごとの境界確率 → ピーク → 3D 面へレイ投影 |
| 棚の種類 | `shelf_classifier.tflite`：境界で区切った各区間を 26 クラスに分類 |
| ノイズ除去 | 店舗の主方向に 90° スナップ → 同一面を統合 → 2 フレーム以上で見えた境界のみ採用 → 区間ごとに確率を合算してラベル決定。2.2 m 超の区間は等分 |
| 3D 表現 | 床・棚（奥行 0.45 m）・冷凍ケース（床から 0.5〜1.3 m の水平面）の箱、正面にラベル |

ロジック本体は `Reconstructor.kt`（JVM ユニットテストあり）。

## ビルド
```
JAVA_HOME=<JDK17+> ./gradlew testDebugUnitTest assembleDebug
```

## モデルの再学習
```
python -m venv tools/.venv && tools/.venv/Scripts/pip install -r tools/requirements.txt
tools/.venv/Scripts/python tools/train.py --data <training_images のパス>
```
`training_images/labels.csv`（file, shelf）と `boundaries.csv`（file, cuts＝境界の x 位置 0〜1 をスペース区切り）を使い、
`app/src/main/assets/` に tflite と `labels.txt` を書き出す。

## 学習データのダッシュボード
```
python dashboard/serve.py --data <training_images のパス>   # → http://127.0.0.1:8765
```
クラス別枚数・未ラベル/不足クラスなどの問題点・全画像（境界線を赤で重ね表示）を確認できる。Pillow があればサムネイルを `dashboard/.cache/` に作る。

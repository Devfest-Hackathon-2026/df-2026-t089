# ShelfSight iOS 版

Android 版（`app/`）の買い物の案内（`assist`）を、同じ構成で Swift に移したものです。操作と話す内容は Android 版と同じです。

**iOS 版と Android 版は、いつも機能を同じにします。** どちらかで機能を足す・変える・止めるときは、もう片方も同じ変更をしてください（定数、判定のしきい値、話す文、画面に出す内容も合わせる）。

今の動き（Android 版と同じ）:

- 歩いている間、商品の枠（SKU-110K の YOLO）を 1 秒ごとに出し、切り抜きを優先度の高いものから MobileCLIP-S0 と分類層で 122 カテゴリに分け、AR 空間の地図（`ShelfMap`）と店のモデル（`StoreModel`、棚の配置は貪欲法）を作る
- 商品の位置と距離（Android 版の Depth API を使うところ）は、すべて **LiDAR の深度（ARKit の `sceneDepth`）** で測る。LiDAR のない機種では、壁の平面に当てられる商品だけを地図に置く
- 人の検出は Android 版と同じ YOLO の person。近づいてくる人の方向から警告音を鳴らす
- 音は Android 版と同じ頭部インパルス応答（SADIE II KU100）で両耳へ畳み込む
- 「3D」のタブで店のモデルを線画で（2D への切り替えと凡例も）、「立体音響」のタブで動く音源を確かめられる
- GPS で店が分かれば、観測をバックエンドへ送り（サーバが OR-Tools で解いて保存）、前に保存された地図を読み込む

| 操作 | すること |
|---|---|
| 1 回たたく | 棚に向かっていればその棚の商品を上・真ん中・下に分けて、近くに棚がいくつかあれば方向ごとにチャイムを鳴らして言う |
| 2 回たたく | 手前の商品の名前とカテゴリを言う（LiDAR で近い商品を測れなければ画面の中央の商品） |
| 長押し | 話しかける。「〇〇はどこ？」などに、店の地図と今の位置を知っている LLM が答える |

VoiceOver が有効なときは、ダブルタップが 1 回たたく、ダブルタップ長押しが長押し、「商品名を調べる」のアクションが 2 回たたくです。

## Android 版との対応

| Android（`app/src/main/java/jp/devfest/shelfsight/`） | iOS（`ios/ShelfSight/`） | 変えたところ |
|---|---|---|
| `assist/AssistActivity.kt` | `Assist/AssistViewController.swift`（`Assist` と画面） | ARCore → ARKit、GL スレッド → AR のキュー |
| `assist/*.kt`（判定・説明・地図・補正・立体音響・声） | `Assist/*.swift` | 処理と定数はそのまま |
| `assist/SpatialMixer.kt`（AudioTrack） | `Assist/SpatialMixer.swift` | AVAudioEngine の AVAudioSourceNode に書く |
| `assist/Voice.kt`（TextToSpeech・SpeechRecognizer） | `Assist/Voice.swift` | AVSpeechSynthesizer・SFSpeechRecognizer（端末内で聞き取れるときは端末内） |
| `perception/PersonDetector.kt`（MediaPipe） | `Perception/PersonDetector.swift` | Vision の人の検出（OS に入っているモデル） |
| `perception/PersonTracker.kt`・`PersonDistance.kt` | `Perception/PersonTracker.swift` | そのまま |
| `assist/SceneTiming.kt` | `Assist/SceneTiming.swift` | そのまま |
| `cloud/GeminiClient.kt` | `Cloud/GeminiClient.swift` | URLSession |
| `ocr/CloudVisionOcr.kt`（応答の読み方） | `OCR/ProductReading.swift` | ML Kit の予備はなし |
| `ar/FrameCapture.kt`・`Unprojector.kt` | `AR/*.swift` | 座標の決まりは同じ（カメラ座標は +X が右、+Y が上、-Z が前） |

手前の商品 1 つ（③）は、Cloud Vision で印字の文字だけを読み、その文字から種類を決めます（`Assist/ProductCategory.swift`）。端末の辞書（`Assist/category_rules.json`）で決めきれないときだけ、読み取った文字を Gemini に渡して決めます（画像は送りません）。商品名、種類の順に言い、お酒の手がかりがあるときは「お酒」と必ず言います。Android 版も同じです（辞書は `app/src/main/assets/category_rules.json` に同じものを置き、単体テストで同じかを確かめています）。調べるたびに、送った画像・応答・答えを `Documents/lookup/` に残します（「ファイル」アプリか USB でつないだ PC から取り出せる）。

モデルと照合先は Android 版と同じファイル（`app/src/main/assets/products/` と `coarse/categories.json`）を、ビルドのときにそのままアプリに入れます。

入れていないもの:

- 録画と再生、再生の事前計算（ARCore の録画機能に合わせた開発用の機能で、ARKit には同じ機能がない）
- 歩行中の棚の種類の判別（`walking/`）、地図づくりや Store3D の開発用の画面（Android 版でも動かしていない）

## GitHub Actions でビルドを確かめる

`.github/workflows/ios-build.yml` が、`ios/` などの変更を push したときと手動で、署名なしのビルドと推定のテスト（Android 版と同じデータで `StoreModel` が同じ結果になること）を動かします。シークレットはいりません。

商品の分類のモデル（`mobileclip_s0.onnx` と `category_head.*`）も git に入っているので、Actions で作ったアプリでも分類まで動きます。

## GitHub Actions から TestFlight へ

`.github/workflows/ios-testflight.yml` が、Actions の画面で「Run workflow」を押したときと、`main` に `ios/` の変更を push したときに動き、ビルドして App Store Connect に上げます。上げたビルドは、処理が終わると TestFlight の内部テストに出ます。

この提出用のリポジトリでは GitHub Actions が使えないので、開発用のリポジトリで動かしてください。

### 最初に 1 回だけすること

1. Apple Developer Program に登録したチームで、App Store Connect に新しい App を作る（プラットフォームは iOS、バンドル ID は `jp.devfest.shelfsight`。別の ID にするときは下の `IOS_BUNDLE_ID` に書く）。バンドル ID が一覧にないときは、Certificates, Identifiers & Profiles で先に登録する
2. App Store Connect の「ユーザとアクセス」→「統合」→「App Store Connect API」で、**Admin** のキーを作り、`.p8` をダウンロードする（署名の証明書をクラウドで作るので App Manager では足りない）
3. GitHub のリポジトリの Settings → Secrets and variables → Actions に、次を入れる

| 種類 | 名前 | 中身 |
|---|---|---|
| Secret | `APPSTORE_API_KEY_ID` | API キーのキー ID |
| Secret | `APPSTORE_API_ISSUER_ID` | API キーの Issuer ID |
| Secret | `APPSTORE_API_PRIVATE_KEY` | `.p8` ファイルの中身（`-----BEGIN PRIVATE KEY-----` から全部） |
| Secret | `APPLE_TEAM_ID` | チーム ID（10 文字、Membership details にある） |
| Secret | `GEMINI_API_KEY` | 任意。棚の説明の文と質問に使う |
| Secret | `GCP_API_KEY` | 任意。Cloud Vision（長押しで商品を調べる）に使う |
| Variable | `IOS_BUNDLE_ID` | 任意。バンドル ID を変えるとき |

### 署名（証明書とプロファイル）

署名は手動です。App Store 配布用の証明書（Apple Distribution）と、このアプリの App Store 用のプロビジョニングプロファイルを、App Store Connect API で 1 回だけ作り、次の Secrets に入れてあります。CI は一時的なキーチェーンに入れて使い、終わったら消します。

| Secret | 中身 |
|---|---|
| `DIST_CERT_P12` | 証明書と秘密鍵の p12（base64） |
| `DIST_CERT_PASSWORD` | p12 のパスワード |
| `APPSTORE_PROFILE` | プロファイル（base64） |
| `APPSTORE_PROFILE_NAME` | プロファイルの名前 |

控えは作った PC の `%USERPROFILE%\.shelfsight-signing\` にあります（リポジトリには入れない）。証明書とプロファイルは 2027-11-06 に切れるので、その前に作り直して Secrets を入れ替えてください。

自動署名にしていないのは、アーカイブのときに開発用のプロファイル（チームに iPhone の登録が要る）が要るためです。署名せずにアーカイブして書き出しのときだけ署名すると、中の `onnxruntime.framework` が署名し直されず、App Store Connect に断られました。

ビルド番号には Actions の実行番号（`github.run_number`）を使います。バージョン（`0.1.0`）を上げるときは `ios/project.yml` の `MARKETING_VERSION` を変えます。

API キーはアプリに入るので、TestFlight は内部テスト（チームのメンバー）だけにしています（`testFlightInternalTestingOnly`）。外部のテスターに配るときは、キーを入れずにビルドするか、キーを守る仕組み（サーバーを通すなど）を作ってからにしてください。

## 手元でビルドする（Mac）

```sh
brew install xcodegen
cd ios
cp Secrets.example.xcconfig Secrets.xcconfig   # チーム ID と API キーを書く
xcodegen generate
open ShelfSight.xcodeproj
```

実機（ARKit が必要）を選んで実行します。初回はカメラ・マイク・音声認識の許可を求めます。

`productOnly` をやめたときの記録（商品の解析のたびの画像と `records.jsonl`、書式は Android 版と同じ）は、「ファイル」アプリの「このiPhone内」→「ShelfSight」→「records」にあります。

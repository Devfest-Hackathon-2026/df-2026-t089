# ShelfSight iOS 版

Android 版（`app/`）の買い物の案内（`assist`）を、同じ構成で Swift に移したものです。操作と話す内容は Android 版と同じです。

**iOS 版と Android 版は、いつも機能を同じにします。** どちらかで機能を足す・変える・止めるときは、もう片方も同じ変更をしてください（定数、判定のしきい値、話す文、画面に出す内容も合わせる）。

今の動き（Android 版の `PRODUCT_ONLY = true` と同じ。iOS 版では `Assist.productOnly`）:

- 商品の解析（枠を出して名前を付ける）を、前の解析が終わったらすぐ次を回し、かかった時間を画面に出す
- 棚が 1 つ写ったときの説明（②）、長押し（③ または ②）、質問は動く
- 人の検出（④）、記録、面の検出、深度は止めている
- 棚の種類の判別（歩行中の 8 列 × 7 分類）、棚のラベルを AR 空間に貯めること、3D の地図は入れていない（Android 版も止めている）

| 操作 | すること |
|---|---|
| 長押し | 「調べています」と言い、手前の商品 1 つなら社名と商品名、棚の広い範囲なら並んでいる商品の説明、その間なら「一歩下がって」と頼む |
| 2 回たたく | 質問する（「何がある？」「これは何？」「止めて」「もう一度」などはその操作、それ以外は LLM が今のカメラの画像を見て答える） |

VoiceOver が有効なときは、ダブルタップが質問、ダブルタップ長押しか「調べる」のアクションが長押しです。

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

モデルと照合先は Android 版と同じファイル（`app/src/main/assets/products/` と `coarse/categories.json`）を、ビルドのときにそのままアプリに入れます。

入れていないもの（Android 版でも動かしていないもの）:

- 棚の種類の判別（`walking/`）、棚のラベル（`ShelfLabels`・`MapBias`・`ShelfMap`）、3D の地図（`Map3DOverlay`・`StoreMapper`・`MapRenderer`）
- 開発用の画面（地図づくり、Store3D）

深度は LiDAR のある iPhone（Pro の機種）でだけ取れます。スマホのすぐ前に商品があるかの判定（`NearProduct`、長押しで商品を調べる ③）は、商品の枠を使わず画面の中央の深度だけで決めるので、LiDAR のない iPhone では ③ が動きません（画面の「③ 近い商品」にそう出ます）。Android 版は、ARCore が普通のカメラからも深度を出すので、ほとんどの機種で動きます。

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

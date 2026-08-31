# tagger-neo

画像データセットのタグを高速に整理する、ネイティブ Rust デスクトップアプリです。
画像と同名の UTF-8 `.txt`（カンマ区切り）を読み書きし、WD14 Tagger をアプリ内で実行できます。

## 機能

- サブフォルダを含む JPEG / PNG / WebP / BMP / TIFF の一覧表示
- サムネイルを見ながら個別 caption 編集
- AND / OR の include filter と exclude filter
- タグ候補をクリックして選択（検索、Prefix / Suffix / Regex、頻度・長さソート）
- Positive / Negative の独立フィルターと画像選択フィルター
- 表示中／チェック済み／現在画像への Batch Edit
- 共通タグ編集、追加・先頭追加・複数削除・置換・全文正規表現置換・重複除去
- タグのアルファベット／出現頻度ソート、タグ数による切り詰め
- 複数選択、Undo / Redo、atomic な一括保存
- `.000`～`.999` caption backup、caption拡張子指定、ファイル名fallback
- 画像・caption・backupの一括移動／確認付き削除
- kohya_ss metadata JSONの読み込み／書き出し・merge
- WD14のモデル選択、単一画像／表示中画像への一括推論
- DirectML による GPU 推論（DirectX 12 対応 GPU）
- General / Character の個別しきい値、Rating、undesired tags、append / replace
- モデルの初回自動取得とローカルキャッシュ

## 起動

Rust 1.88 以上と Visual Studio Build Tools の C++ ワークロードが必要です。

```powershell
cargo run --release
```

フォルダを直接指定して起動することもできます。

```powershell
cargo run --release -- "D:\dataset"
```

Windows 実行ファイルは次で生成できます。

```powershell
cargo build --release
```

生成先は `target/release/tagger-neo.exe` です。

## 操作

- `Ctrl+O`: フォルダを開く
- `Ctrl+S`: 全変更を保存
- `Ctrl+Z` / `Ctrl+Y`: Undo / Redo
- サムネイルクリック: 編集対象を切り替え
- サムネイル上のチェック: 一括選択
- サムネイル下のタグをクリック: Batch対象タグを選択
- `Ctrl` + タグクリック: Positive Filter
- `Alt` + タグクリック: Negative Filter
- `Shift` + タグクリック: その画像からタグを削除
- `Enter` / `Delete`: 現在画像をチェック／チェック解除
- ボタンの説明は hover tooltip に表示

一括編集と WD14 の「表示中」実行は、現在のフィルター結果だけを対象にします。

## WD14

WD14パネルのプルダウンから、kohya_ss GUIと同じ次の9モデルを選択できます。

- ConvNeXt v2 / ConvNeXtV2 v2 / ViT v2 / SwinV2 v2 / MOAT v2
- SwinV2 v3 / ViT v3 / ConvNeXt v3 / EVA02-Large v3

初回使用時に選択モデルの `model.onnx` と `selected_tags.csv` を Hugging Faceから取得します。
モデルは種類により約310 MiB～1.2 GiBです。各モデルは別々のフォルダへキャッシュされ、
一度取得したモデルを切り替えて再利用できます。
推論は ONNX Runtime の DirectML Execution Provider を使用し、以後はキャッシュを使用します。
`DirectML.dll` は Release 実行ファイルと同じフォルダへ実体を配置します。

## 参考

- [Dataset Tag Editor Standalone](https://github.com/toshiaki1729/dataset-tag-editor-standalone)
- [kohya_ss GUI](https://github.com/bmaltais/kohya_ss/blob/master/kohya_gui/wd14_caption_gui.py)
- [SmilingWolf WD14 models](https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger-v2)

## License

GPL v3

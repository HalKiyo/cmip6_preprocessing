# cmip6_preprocessing

CMIP6 historical (r1i1p1f1) の月平均データを読み込み、気候値・標準偏差・偏差・標準化偏差を
1x1 度と 5x5 度の両方で計算して pickle に保存します。

## 構成

```
cmip6_preprocessing/
├── config.py    # 入出力ディレクトリ、モデルごとのファイル分割・土壌層の深さなどの設定
├── core.py      # 読み込み、単位変換、気候値・偏差・標準化、5x5 度への粗視化、保存
├── plot.py      # 確認用の地図描画 (--plot のときだけ読み込む)
└── __main__.py  # コマンドライン
```

以前は `モデル名/anom_coarse_conc_r1i1p1f1_<変数>.py` が 43 モデル × 5 変数で 215 本あり、
ほぼ同じ処理がコピーされていました。現在は処理を `core.py` に一つだけ置き、
モデルごとに違う部分 (ファイルの年の区切り、`tsl` の深さ、経度の入れ替えなど) は
`config.py` の `MODELS` に書くだけになっています。

## 使い方

```bash
python -m cmip6_preprocessing MIROC6 pr          # 1 モデル 1 変数
python -m cmip6_preprocessing MIROC6             # 1 モデルの全変数 (pr tsl mrso snc tos)
python -m cmip6_preprocessing MIROC6 pr tos      # 変数を複数指定
python -m cmip6_preprocessing --all              # 全モデル全変数
python -m cmip6_preprocessing --all tos          # 全モデルの tos
python -m cmip6_preprocessing MIROC6 pr --plot   # 確認用の図を表示
python -m cmip6_preprocessing MIROC6 pr --no-save
python -m cmip6_preprocessing --list             # モデル一覧
```

`--all` で途中のモデルが失敗しても残りは続行し、最後に失敗した組み合わせを表示します。

Python から使う場合

```python
from cmip6_preprocessing import get_settings, preprocess, save

settings = get_settings("MIROC6", "pr")
result = preprocess(settings)   # {"pr_raw": ..., "pr_std": ..., ...}
save(result, "MIROC6", "pr")
```

## 入出力

* 入力 `{DATA_DIR}/{model}/{variable}/{variable}_{begin}-{end}.nc`
* 出力 `{SAVE_DIR}/{model}/{variable}_{model}.pickle`

`DATA_DIR` と `SAVE_DIR` は `config.py` の先頭で変更できます。
pickle のキーは従来と同じ
`{variable}_raw`, `_clim`, `_variance`, `_anom`, `_std`,
`_coarse`, `_coarse_clim`, `_coarse_variance`, `_coarse_anom`, `_coarse_std` です。

## モデルの追加

`config.py` の `MODELS` に 1 行足します。

```python
"NEW-MODEL": {"files": [(1850, 1899), *chunks(1900, 1999, 50), (2000, 2014)], "depth": 1},
```

| キー | 意味 |
| --- | --- |
| `files` | 入力ファイルの (開始年, 終了年) を時系列順に並べたもの。`chunks(開始, 終了, 年数)` で等間隔のファイルをまとめて書けます |
| `depth` | `tsl` で平均する土壌層 (0 から depth まで) |
| `conc` | 経度を 180 度ずらす (ICON-ESM-LR など) |
| `masked_depth_mean` | `tsl` の深さ平均で欠損層を除いて平均する |
| `drop_last_months` | 保存前に末尾の月を削る (FGOALS-g3 は 2016 年まであるので 24) |
| `variables` | 特定の変数だけ上の設定を変える |

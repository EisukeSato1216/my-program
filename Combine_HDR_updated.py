import cv2
import numpy as np
import os

# 画像フォルダと出力フォルダのパスを指定
response_function_image_folder = r"******"

# フォルダ内の画像ファイル名を取得
response_function_image_paths = [os.path.join(response_function_image_folder, fname) for fname in os.listdir(response_function_image_folder) if fname.endswith(".bmp")]

# 画像が読み込まれたか確認
if len(response_function_image_paths) == 0:
    print("画像が見つかりませんでした。フォルダのパスを確認してください。")
    exit()

# ファイル名から露光時間をマイクロ秒から秒に変換 
response_function_exposure_times = np.array([int(os.path.basename(path).split('.')[0]) for path in response_function_image_paths], dtype=np.float32) / 1e6

# 画像を読み込む
images = []
for path in response_function_image_paths:
    img = cv2.imread(path)
    if img is not None:
        images.append(img)
    else:
        print(f"画像の読み込みに失敗しました: {path}")

# 画像の数と露光時間の数が一致しているか確認
if len(images) != len(response_function_exposure_times):
    print("画像の数と露光時間の数が一致していません。処理を終了します。")
    exit()

# カメラの応答関数を計算
calibrateDebevec = cv2.createCalibrateDebevec()
try:
    responseDebevec = calibrateDebevec.process(images, response_function_exposure_times)
    print("カメラの応答関数の計算が正常に完了しました。")
except cv2.error as e:
    print(f"OpenCVエラー: {e}")
    exit()

# 画像フォルダのパス
image_folder = r"*****" # ユーザーの画像フォルダパスに変更
output_folder = r"*****" # 出力フォルダパスに変更

# 出力フォルダが存在しない場合は作成
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# フォルダ内の画像ファイルをリスト化
image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

# 対応する露光時間
hdr_exposure_times = np.array([5120 / 1e6, 160 / 1e6], dtype=np.float32)


# 2枚ずつ画像をペアにしてHDR合成を行う
for i in range(0, len(image_files), 2):
    if i+1 < len(image_files):  # 2枚目が存在する場合のみ処理
        img1_path = os.path.join(image_folder, image_files[i])
        img2_path = os.path.join(image_folder, image_files[i+1])

        # 画像を読み込む
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        hdr_images = [img1, img2]

        if img1 is None or img2 is None:
            print(f"画像の読み込みに失敗しました: {img1_path} または {img2_path}")
            continue

        # HDR画像を生成
        mergeDebevec = cv2.createMergeDebevec()
        hdrDebevec = mergeDebevec.process(hdr_images, hdr_exposure_times, responseDebevec)

        # トーンマッピングでHDR画像をLDRに変換
        tonemap = cv2.createTonemapReinhard(1.0, 0, 0, 0)
        ldrReinhard = tonemap.process(hdrDebevec)

        # NaNや無限大を0に置き換える
        ldrReinhard = np.nan_to_num(ldrReinhard, nan=0.0, posinf=255.0, neginf=0.0)

        # LDR画像を保存
        ldrReinhard = np.clip(ldrReinhard * 255, 0, 255).astype('uint8')
        output_path = os.path.join(output_folder, f"combined_{i//2+1}.png")
        cv2.imwrite(output_path, ldrReinhard)
        print(f"{output_path} 保存完了")

print("画像のペア合成が完了しました！")

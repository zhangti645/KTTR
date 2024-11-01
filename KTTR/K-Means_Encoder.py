import os
import yaml
from argparse import ArgumentParser
import numpy as np
from skimage.transform import resize
import torch
from sync_batchnorm import DataParallelWithCallback
from modules.keypoint_detector import KPDetector
import time
import cv2
from sklearn.cluster import KMeans
from arithmetic.value_encoder import final_encoder_expgolomb
from arithmetic.value_decoder import final_decoder_expgolomb, data_convert_inverse_expgolomb


def load_checkpoints(scalenumber, config_path, checkpoint_path, cpu=False):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['common_params'])
    if not cpu:
        kp_detector.cuda()

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)

    kp_detector.load_state_dict(checkpoint['kp_detector' + '_' + str(scalenumber)], strict=False)

    if not cpu:
        kp_detector = DataParallelWithCallback(kp_detector)
    kp_detector.eval()

    return kp_detector


def select_key_frames(features, num_clusters):
    """
    根据特征将帧聚类，并选择每个聚类中心最近的帧作为关键帧参考帧。

    参数：
    features: 形状为 (num_frames, feature_dim) 的张量，表示每帧的紧凑特征。
    num_clusters: 聚类的数量。

    返回：
    key_frame_indices: 关键帧参考帧的帧序列号列表。
    """
    features_np = features.cpu().detach().numpy()
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features_np)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    key_frame_indices = []
    for i in range(num_clusters):
        cluster_indices = np.where(labels == i)[0]
        cluster_features = features_np[cluster_indices]
        distances = np.linalg.norm(cluster_features - cluster_centers[i], axis=1)
        min_distance_index = cluster_indices[np.argmin(distances)]
        key_frame_indices.append(min_distance_index)

    return key_frame_indices


def RawReader_planar(FileName, ImgWidth, ImgHeight, NumFramesToBeComputed):
    with open(FileName, 'rb') as f:
        data = f.read()
    data = np.frombuffer(data, dtype=np.uint8)

    frames = NumFramesToBeComputed
    width = ImgWidth
    height = ImgHeight

    listR = []
    listG = []
    listB = []
    for i in range(frames):
        R = data[i * width * height:(i + 1) * width * height].reshape((height, width))
        G = data[(frames + i) * width * height:(frames + i + 1) * width * height].reshape((height, width))
        B = data[(2 * frames + i) * width * height:(2 * frames + i + 1) * width * height].reshape((height, width))
        listR.append(R)
        listG.append(G)
        listB.append(B)
    return listR, listG, listB


if __name__ == "__main__":
    parser = ArgumentParser()

    frames = 250  # 250

    scalenumber = 4  # 4/6/8

    scalefactor = 1 / scalenumber
    width = 64 * scalenumber
    height = 64 * scalenumber

    Qstep = 40  # 16

    max_ref_num = 3  # 3/4/5
    threshold = 0.25  # 0.2/0.25/0.3/0.35/0.4/0.45

    modeldir = '2ref'
    config_path = './checkpoint/' + modeldir + '/vox-256.yaml'
    checkpoint_path = './checkpoint/' + modeldir + '/0099-checkpoint.pth.tar'

    kp_detector = load_checkpoints(scalenumber, config_path, checkpoint_path, cpu=False)

    seqlist = ['001']
    qplist = ['32']

    model_dirname = './experiment/' + modeldir + "/" + "Qstep_" + str(Qstep) + "/"

    for seqIdx, seq in enumerate(seqlist):
        for qpIdx, QP in enumerate(qplist):
            original_seq = './Testrgb/' + str(width) + '/' + str(seq) + '_' + str(width) + 'x' + str(width) + '.rgb'

            listR, listG, listB = RawReader_planar(original_seq, width, height, frames)

            driving_kp = model_dirname + '/kp/' + str(width) + '/' + seq + '_QP' + str(QP) + '/'
            os.makedirs(driving_kp, exist_ok=True)

            dir_enc = model_dirname + '/enc/' + str(width) + '/' + seq + '_QP' + str(QP) + '/'
            os.makedirs(dir_enc, exist_ok=True)

            f_org = open(original_seq, 'rb')
            ref_rgb_list = []

            ref_kp_list = []
            ref_multiper_list = []

            seq_kp_integer = []

            start = time.time()

            sum_bits = 0
            all_features = []

            for frame_idx in range(frames):
                frame_idx_str = str(frame_idx).zfill(4)
                img_input = np.fromfile(f_org, np.uint8, 3 * height * width).reshape((3, height, width))

                img_input_resized = resize(img_input, (3, height, width))  # normalize to 0-1
                with torch.no_grad():
                    frame_tensor = torch.tensor(img_input_resized[np.newaxis].astype(np.float32)).cuda()
                    kp_frame, _ = kp_detector(frame_tensor, scalefactor)
                    all_features.append(kp_frame['value'])

            # 将所有特征堆叠成一个张量
            all_features = torch.stack(all_features)

            # 选择关键帧
            key_frame_indices = select_key_frames(all_features, num_clusters=max_ref_num)
            key_frame_set = set(key_frame_indices)

            for frame_idx in range(frames):
                frame_idx_str = str(frame_idx).zfill(4)
                img_input = np.fromfile(f_org, np.uint8, 3 * height * width).reshape((3, height, width))

                if frame_idx in key_frame_set:  # 处理关键参考帧
                    f_temp = open(dir_enc + 'frame' + frame_idx_str + '_org.rgb', 'w')
                    img_input.tofile(f_temp)
                    f_temp.close()

                    os.system("./vtm/encode.sh " + dir_enc + 'frame' + frame_idx_str + " " + QP + " " + str(
                        width) + " " + str(height))

                    bin_file = dir_enc + 'frame' + frame_idx_str + '.bin'
                    bits = os.path.getsize(bin_file) * 8
                    sum_bits += bits

                    f_temp = open(dir_enc + 'frame' + frame_idx_str + '_rec.rgb', 'rb')
                    img_rec = np.fromfile(f_temp, np.uint8, 3 * height * width).reshape((3, height, width))
                    ref_rgb_list.append(img_rec)

                    img_rec = resize(img_rec, (3, height, width))
                    with torch.no_grad():
                        reference = torch.tensor(img_rec[np.newaxis].astype(np.float32)).cuda()
                        kp_reference, multiscale_perceptural_representation_reference = kp_detector(reference,
                                                                                                    scalefactor)
                        ref_kp_list.append(kp_reference)
                        ref_multiper_list.append(multiscale_perceptural_representation_reference)

                        kp_value = kp_reference['value']
                        kp_value_list = str(kp_value.tolist()).replace(' ', '')

                        with open(driving_kp + '/frame' + frame_idx_str + '.txt', 'w') as f:
                            f.write(kp_value_list)

                        kp_value_frame = eval(kp_value_list)
                        seq_kp_integer.append(kp_value_frame)
                else:  # 处理P帧
                    interframe = cv2.merge([listR[frame_idx], listG[frame_idx], listB[frame_idx]])
                    interframe = resize(interframe, (width, height))[..., :3]

                    with torch.no_grad():
                        interframe = torch.tensor(interframe[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).cuda()
                        kp_interframe, multiscale_perceptural_representation_interframe = kp_detector(interframe,
                                                                                                      scalefactor)

                        # 计算当前帧与所有关键帧的残差
                        residuals = [kp_interframe['value'] - ref_kp['value'] for ref_kp in ref_kp_list]

                        # 对所有残差进行编码和量化
                        encoded_residuals = []
                        for residual in residuals:
                            residual_list = residual.tolist()
                            residual_list = [int(round(i * Qstep)) for i in residual_list]
                            encoded_residuals.append(residual_list)

                        # 将编码后的残差写入文件
                        with open(driving_kp + '/frame' + frame_idx_str + '.txt', 'w') as f:
                            for encoded_residual in encoded_residuals:
                                encoded_residual_str = str(encoded_residual).replace(' ', '')
                                f.write(encoded_residual_str + '\n')

                        # 将残差值保存到seq_kp_integer
                        seq_kp_integer.append([encoded_residuals])

            rec_sem = []
            for frame in range(1, frames):
                frame_idx = str(frame).zfill(4)
                if frame == 1:
                    rec_sem.append(seq_kp_integer[0])

                    kp_difference = (np.array(seq_kp_integer[frame]) - np.array(seq_kp_integer[frame - 1])).tolist()
                    kp_difference = [i * Qstep for i in kp_difference]
                    kp_difference = list(map(round, kp_difference))

                    bin_file = driving_kp + '/frame' + frame_idx + '.bin'
                    final_encoder_expgolomb(kp_difference, bin_file)

                    bits = os.path.getsize(bin_file) * 8
                    sum_bits += bits

                    res_dec = final_decoder_expgolomb(bin_file)
                    res_difference_dec = data_convert_inverse_expgolomb(res_dec)
                    res_difference_dec = [i / Qstep for i in res_difference_dec]
                    rec_semantics = (np.array(res_difference_dec) + np.array(rec_sem[frame - 1])).tolist()
                    rec_sem.append(rec_semantics)
                else:
                    kp_difference = (np.array(seq_kp_integer[frame]) - np.array(rec_sem[frame - 1])).tolist()
                    kp_difference = [i * Qstep for i in kp_difference]
                    kp_difference = list(map(round, kp_difference))

                    bin_file = driving_kp + '/frame' + frame_idx + '.bin'
                    final_encoder_expgolomb(kp_difference, bin_file)

                    bits = os.path.getsize(bin_file) * 8
                    sum_bits += bits

                    res_dec = final_decoder_expgolomb(bin_file)
                    res_difference_dec = data_convert_inverse_expgolomb(res_dec)
                    res_difference_dec = [i / Qstep for i in res_difference_dec]
                    rec_semantics = (np.array(res_difference_dec) + np.array(rec_sem[frame - 1])).tolist()
                    rec_sem.append(rec_semantics)

            end = time.time()
            print("Extracting kp success. Time is %.4fs. Key points coding %d bits." % (end - start, sum_bits))

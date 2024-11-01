import os, sys
import yaml
from argparse import ArgumentParser
import numpy as np
from skimage.transform import resize
import torch
from sync_batchnorm import DataParallelWithCallback
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
import json
import time
import cv2
from arithmetic.value_decoder import *
from flowvisual import *
from fusion import FusionNetwork


def load_checkpoints(scalenumber, config_path, checkpoint_path, cpu=False):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['common_params'])

    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['common_params'])
    if not cpu:
        kp_detector.cuda()

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint['generator' + '_' + str(scalenumber)], strict=False)
    kp_detector.load_state_dict(checkpoint['kp_detector' + '_' + str(scalenumber)], strict=False)

    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)
    generator.eval()
    kp_detector.eval()

    return generator, kp_detector


def make_prediction(reference_frame, kp_reference, kp_current, generator, scalefactor, relative=False, adapt_movement_scale=False, cpu=False):
    kp_norm = normalize_kp(kp_source=kp_reference, kp_driving=kp_current,
                           kp_driving_initial=kp_reference, use_relative_movement=relative,
                           use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)

    out = generator(reference_frame, scalefactor, kp_reference, kp_norm, reference_frame, kp_reference)

    prediction = np.transpose(out['prediction'].data.cpu().numpy(), [0, 1, 2, 3])[0]

    return prediction


if __name__ == "__main__":
    parser = ArgumentParser()

    modeldir = 'temporal_adaptive_2ref'
    config_path = './checkpoint/' + modeldir + '/vox.yaml'
    checkpoint_path = './checkpoint/' + modeldir + '/0099-checkpoint.pth.tar'

    frames = 250
    Qstep = 40
    scalenumber = 4  # 4/6/8
    scalefactor = 1 / scalenumber
    width = 64 * scalenumber
    height = 64 * scalenumber

    generator, kp_detector = load_checkpoints(scalenumber, config_path, checkpoint_path, cpu=False)
    fusion_network = FusionNetwork(kp_detector=kp_detector, in_channels=3).cuda()
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    num_channel = config['common_params']['num_kp']
    N_size = int(width * scalefactor / 16)

    seqlist = ['001']
    qplist = ['32']
    model_dirname = './experiment/' + modeldir + "/" + "Qstep_" + str(Qstep) + "/"

    totalResult = np.zeros((len(seqlist) + 1, len(qplist)))
    for seqIdx, seq in enumerate(seqlist):
        for qpIdx, QP in enumerate(qplist):
            original_seq = './Testrgb/' + str(width) + '/' + str(seq) + '_' + str(width) + 'x' + str(width) + '.rgb'
            driving_kp = model_dirname + '/kp/' + str(width) + '/' + seq + '_QP' + str(QP) + '/'
            dir_dec = model_dirname + '/dec/' + str(width) + '/'
            os.makedirs(dir_dec, exist_ok=True)
            decode_seq = dir_dec + seq + '_QP' + str(QP) + '.rgb'

            dir_enc = model_dirname + '/enc/' + str(width) + '/' + seq + '_QP' + str(QP) + '/'
            os.makedirs(dir_enc, exist_ok=True)

            savetxt = model_dirname + '/resultBit/' + str(width) + '/'
            os.makedirs(savetxt, exist_ok=True)

            f_org = open(original_seq, 'rb')
            f_dec = open(decode_seq, 'w')
            ref_rgb_list = []
            ref_kp_list = []
            seq_kp_integer = []

            start = time.time()
            gene_time = 0
            sum_bits = 0

            for frame_idx in range(frames):
                frame_idx_str = str(frame_idx).zfill(4)
                img_input = np.fromfile(f_org, np.uint8, 3 * height * width).reshape((3, height, width))

                if frame_idx == 0:
                    f_temp = open(dir_enc + 'frame' + frame_idx_str + '_org.rgb', 'w')
                    img_input.tofile(f_temp)
                    f_temp.close()
                    os.system("./vtm/encode.sh " + dir_enc + 'frame' + frame_idx_str + " " + QP + " " + str(width) + " " + str(height))
                    bin_file = dir_enc + 'frame' + frame_idx_str + '.bin'
                    bits = os.path.getsize(bin_file) * 8
                    sum_bits += bits
                    f_temp = open(dir_enc + 'frame' + frame_idx_str + '_rec.rgb', 'rb')
                    img_rec = np.fromfile(f_temp, np.uint8, 3 * height * width).reshape((3, height, width))
                    img_rec.tofile(f_dec)
                    ref_rgb_list.append(img_rec)
                    img_rec = resize(img_rec, (3, height, width))
                    with torch.no_grad():
                        reference = torch.tensor(img_rec[np.newaxis].astype(np.float32)).cuda()
                        kp_reference, multiscale_perceptural_representation_reference = kp_detector(reference, scalefactor)
                        kp_value = kp_reference['value']
                        kp_value_list = kp_value.tolist()
                        kp_value_list = str(kp_value_list).replace(' ', '')
                        with open(driving_kp + '/frame' + frame_idx_str + '.txt', 'w') as f:
                            f.write(kp_value_list)
                        kp_value_frame = eval(kp_value_list)
                        seq_kp_integer.append(kp_value_frame)
                        ref_kp_list.append(kp_reference)

                else:
                    interframe = cv2.merge([listR[frame_idx], listG[frame_idx], listB[frame_idx]])
                    interframe = resize(interframe, (width, height))[..., :3]

                    with torch.no_grad():
                        interframe = torch.tensor(interframe[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).cuda()
                        kp_interframe, multiscale_perceptural_representation_interframe = kp_detector(interframe, scalefactor)

                    frame_index = str(frame_idx).zfill(4)
                    bin_save = driving_kp + '/frame' + frame_index + '.bin'
                    encoded_residuals = final_decoder_expgolomb(bin_save)
                    residuals = data_convert_inverse_expgolomb(encoded_residuals)
                    residuals = [residual / Qstep for residual in residuals]

                    predictions = []
                    weights = []
                    for ref_kp in ref_kp_list:
                        residual_tensor = torch.tensor(residuals).to('cuda:0')
                        kp_current_value = ref_kp['value'] + residual_tensor
                        kp_current = {'value': kp_current_value}
                        gene_start = time.time()
                        prediction = make_prediction(reference, ref_kp, kp_current, generator, scalefactor)
                        gene_end = time.time()
                        gene_time += gene_end - gene_start
                        predictions.append(torch.tensor(prediction).cuda())
                        weights.append(1 / (torch.norm(residual_tensor) + 1e-8))

                    weights = torch.tensor(weights).unsqueeze(0).cuda()
                    fused_result = fusion_network(predictions, weights)
                    fused_result_np = fused_result.cpu().numpy()

                    pre = (fused_result_np * 255).astype(np.uint8)
                    pre.tofile(f_dec)

                    bits = os.path.getsize(bin_save) * 8
                    sum_bits += bits

            f_org.close()
            f_dec.close()
            end = time.time()

            totalResult[seqIdx][qpIdx] = sum_bits
            print(seq + '_QP' + str(QP) + '.rgb', "success. Total time is %.4fs. Model inference time is %.4fs. Total bits are %d" % (end - start, gene_time, sum_bits))

    for qp in range(len(qplist)):
        for seq in range(len(seqlist)):
            totalResult[-1][qp] += totalResult[seq][qp]
        totalResult[-1][qp] /= len(seqlist)

    print(totalResult)
    np.set_printoptions(precision=5)
    totalResult = totalResult / 1000
    seqlength = frames / 25
    totalResult = totalResult / seqlength

    np.savetxt(savetxt + '/resultBit.txt', totalResult, fmt='%.5f')

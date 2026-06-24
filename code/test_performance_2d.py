import argparse
import os
import shutil

import h5py
import numpy as np
import torch
import SimpleITK as sitk
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from skimage.measure import label

from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data_split/Prostate', help='Name of Experiment')
parser.add_argument('--data_path', type=str,
                    default='/home/cq/dataset/Prostate/h5file', help='dataset true path')
parser.add_argument('--res_path', type=str, default='./results/Prostate', help='Path to results')
parser.add_argument('--exp', type=str, default='POST-NoT', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')
parser.add_argument('-lnum','--labeled_num', type=int, default=7, help='labeled data')
parser.add_argument('-n','--model_type',type=str,default='tea', help='')
parser.add_argument('--gpu_id','-g', type=int, default=1,
                        help='the id of gpu used to train the model')
parser.add_argument('--save_result','-save',action='store_true', default=False,help='if save the results')

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, jc, hd95, asd

def get_LargestCC(segmentation,num_classes):
    class_list = []
    for i in range(1, num_classes):
        temp_prob = segmentation == i * np.ones_like(segmentation)
        labels = label(temp_prob)
        # -- with 'try'
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
            class_list.append(largestCC * i)
        else:
            class_list.append(temp_prob)
    largestCC = []
    for i in range(1,num_classes):
        largestCC.append(class_list[i-1])
    return list(np.sum(largestCC,0))
def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.data_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out_main = net(input)
            if len(out_main)>1:
                out_main=out_main[0]
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            # pred = get_LargestCC(pred,FLAGS.num_classes)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, FLAGS.num_classes):
        if np.sum(prediction == i)==0:
            metric_i = 0,0,0,0
        else:
            metric_i = calculate_metric_percase(
                prediction == i, label == i)
        metric_list.append(metric_i)

    if FLAGS.save_result:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        img_itk.SetSpacing((1, 1, 10))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        prd_itk.SetSpacing((1, 1, 10))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        lab_itk.SetSpacing((1, 1, 10))
        sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return np.array(metric_list)


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    snapshot_path = "{}/{}_labeled_{}/{}".format(FLAGS.res_path,
                                                 FLAGS.labeled_num, FLAGS.exp, FLAGS.model)
    test_save_path = "{}/{}_labeled_{}/{}_predictions_{}/".format(FLAGS.res_path, FLAGS.labeled_num, FLAGS.exp, FLAGS.model,FLAGS.model_type)
    # snapshot_path = "{}/{}_{}_labeled/{}".format(FLAGS.res_path,
    #                                              FLAGS.exp,FLAGS.labeled_num,  FLAGS.model)
    # test_save_path = "{}/{}_{}_labeled/{}_predictions_{}/".format(FLAGS.res_path,FLAGS.exp, FLAGS.labeled_num,FLAGS.model,FLAGS.model_type)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)
    save_model_path = os.path.join(snapshot_path, 'student/ep_416_dice_0.8576.pth'.format(FLAGS.model,FLAGS.model_type))
    # save_model_path = os.path.join(snapshot_path, '{}_best_{}_model.pth'.format(FLAGS.model,FLAGS.model_type))
    net.load_state_dict(torch.load(save_model_path))
    print("init weight from {}".format(save_model_path))
    net.eval()

    metric_list = []
    for case in tqdm(image_list):
        metric = test_single_volume(case, net, test_save_path, FLAGS)#(3,4)
        metric_list.append(metric)
    metric_list = np.asarray(metric_list)#(20,3,4)
    avg_metric = np.mean(metric_list,0)#(3,4)
    return avg_metric, test_save_path,save_model_path


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    # 3. setup gpus and randomness
    # if FLAGS.gpu_id in range(8):
    if FLAGS.gpu_id in range(10):
        gid = FLAGS.gpu_id
    else:
        gid = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gid)

    metric, test_save_path,save_model_path = Inference(FLAGS)
    avg_metric = np.mean(metric,0)
    print('metric is {} \n'.format(metric))
    print('average metric is {}\n'.format(avg_metric))
    save_per = '../performance_{}.txt'.format(FLAGS.model_type)
    with open(test_save_path+save_per, 'w') as f:
        f.writelines("init weight from {} \n".format(save_model_path))
        f.writelines('metric is {} \n'.format(metric))
        f.writelines('average metric is {}\n'.format(avg_metric))

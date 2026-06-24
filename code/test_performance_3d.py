import os
import argparse
import torch
import torch.nn as nn

from networks.net_factory import net_factory
from val_3D import test_all_case,var_all_case_LA

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data_split/LA', help='Name of Dataset')
parser.add_argument('--data_path', type=str,
                    default='/home/cq/dataset/LA/Left_Atrium/data/', help='Name of Dataset')
parser.add_argument('--dataset', type=str, default='LA', help='Name of dataset')
parser.add_argument('--res_path', type=str, default='./results/LA', help='Path to results')


# parser.add_argument('--root_path', type=str,
#                     default='../data_split/Pancreas', help='Name of Dataset')
# parser.add_argument('--data_path', type=str,
#                     default='/home/cq/dataset/Pancreas/h5file/', help='Name of Dataset')
# parser.add_argument('--dataset', type=str, default='Pancreas', help='Name of dataset')
# parser.add_argument('--res_path', type=str, default='./results/Pancreas', help='Path to results')

parser.add_argument('--exp', type=str, default='Pancreas/POST', help='experiment_name')
parser.add_argument('--model', type=str, default='vnet', help='model_name')
parser.add_argument('--gpu','-g', type=int,nargs='+',  default=[0], help='GPU to use')
parser.add_argument('--detail', type=int,  default=1, help='print metrics for every samples?')
parser.add_argument('-n',"--model_type", type=str, default='tea',help='')
parser.add_argument('--nms', type=int, default=1, help='apply NMS post-procssing?')
parser.add_argument('--labeled_num', type=int, default=4, help='labeled data')
parser.add_argument('--save_result','-save',action='store_true', default=False,help='if save the results')

FLAGS = parser.parse_args()

# snapshot_path = "{}/{}_{}_labeled/{}".format(FLAGS.res_path, FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
# test_save_path = "{}/{}_{}_labeled/{}_predictions/".format(FLAGS.res_path, FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
snapshot_path = "{}/{}_labeled_{}/{}".format(FLAGS.res_path, FLAGS.labeled_num, FLAGS.exp, FLAGS.model)
test_save_path = "{}/{}_labeled_{}/{}_predictions_{}/".format(FLAGS.res_path, FLAGS.labeled_num, FLAGS.exp, FLAGS.model,FLAGS.model_type)
num_classes = 2

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

with open(FLAGS.root_path + '/test.list', 'r') as f:
    image_list = f.readlines()
# image_list = [FLAGS.root_path + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]

flag_pancreas = 'pancreas' in FLAGS.dataset.lower()
if flag_pancreas:
    image_list = [item.replace('\n', '') for item in image_list]
    image_list = [os.path.join(FLAGS.data_path, "data", f"{item}_norm.h5") for item in image_list]
else:
    image_list = [FLAGS.data_path + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]


def test_calculate_metric():
    model = net_factory(net_type=FLAGS.model, in_chns=1, class_num=num_classes)
    if len(FLAGS.gpu) > 1:
        model = nn.DataParallel(model, device_ids=list(range(len(FLAGS.gpu))))
    # save_model_path = os.path.join(snapshot_path, '{}_best_{}_model.pth'.format(FLAGS.model,FLAGS.model_type))
    save_model_path = os.path.join(snapshot_path, 'teacher1/ep_184_dice_0.8895.pth'.format(FLAGS.model,FLAGS.model_type))

    model.load_state_dict({k.replace('module.', ''): v for k,v in torch.load(save_model_path).items()})
    # model = model.module
    print("init weight from {}".format(save_model_path))
    model.eval()

    if not flag_pancreas:
        avg_metric = test_all_case(save_model_path,model, image_list, num_classes=num_classes,
                            patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                            save_result=FLAGS.save_result, test_save_path=test_save_path,
                            metric_detail=FLAGS.detail, nms=FLAGS.nms,model_type=FLAGS.model_type)
    else:
        avg_metric = test_all_case(save_model_path,model, image_list, num_classes=num_classes,
                           patch_size=(96, 96, 96), stride_xy=16, stride_z=16,
                           save_result=FLAGS.save_result, test_save_path=test_save_path,
                           metric_detail=FLAGS.detail, nms=FLAGS.nms,model_type=FLAGS.model_type)
    return avg_metric


if __name__ == '__main__':
    gpu_list_str = ','.join(map(str,FLAGS.gpu))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list_str
    metric = test_calculate_metric()
    print(metric)


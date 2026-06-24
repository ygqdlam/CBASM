import os
import argparse
def save_pth_to_txt(path,del_pth):
    save_file = os.path.join(path,'result_summary.txt')
    max_iter = 0
    best_pth = ''
    with open(save_file,'a') as f:
        for pth_name in os.listdir(path):
            if pth_name.startswith('ep_'):
                assert pth_name.endswith('pth')
                f.writelines(f'{pth_name}\n')
                iter = int(pth_name.split('_')[1].strip())
                if max_iter <= iter:
                    max_iter = iter
                    best_pth = pth_name
        if del_pth:
            for pth_name in os.listdir(path):
                if pth_name.startswith('ep_') and pth_name != best_pth:
                    assert pth_name.endswith('pth')
                    os.remove(os.path.join(path,pth_name))
parser = argparse.ArgumentParser()
parser.add_argument('-f','--filename',type=str,default='4_labeled_post')
parser.add_argument('-t','--trainpath',type=str,default='teacher')
parser.add_argument('--delpth','-d',action='store_true')
if __name__ == '__main__':
    args = parser.parse_args()
    file_path = args.filename
    path = f'../results/Pancreas/{file_path}/vnet/{args.trainpath}'
    save_pth_to_txt(path,args.delpth)
    print(f'success {file_path}_{args.trainpath}')
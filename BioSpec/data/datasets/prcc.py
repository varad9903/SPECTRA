import os
import re
import glob
import random
import math
import logging
import numpy as np
import os.path as osp
from scipy.io import loadmat
from tools.utils import mkdir_if_missing, write_json, read_json
import json

class PRCC(object):
    """ PRCC

    Reference:
        Yang et al. Person Re-identification by Contour Sketch under Moderate Clothing Change. TPAMI, 2019.
    """
    dataset_dir = 'PRCC/prcc'
    def __init__(self, root='data',caption_dir='',caption_model='EVA02-CLIP-bigE-14',load_sum_ft=False, **kwargs):
        self.root = root
        standard_dir = osp.join(root, self.dataset_dir)
        fallback_dir = osp.join(root, 'PRCC')
        if not osp.exists(standard_dir) and osp.exists(osp.join(fallback_dir, 'rgb')):
            self.dataset_dir = fallback_dir
        else:
            self.dataset_dir = standard_dir
        logger = logging.getLogger('BioSpec')
        self.logger=logger
        
        self.train_dir = osp.join(self.dataset_dir, 'rgb/train')
        self.val_dir = osp.join(self.dataset_dir, 'rgb/val')
        self.test_dir = osp.join(self.dataset_dir, 'rgb/test')
        self._check_before_run()
        
        self.load_sum_ft=load_sum_ft
        self.bio_index=list(map(int,kwargs['bio_index']))
        self.non_bio_index=list(map(int,kwargs['nonbio_index']))
        
       
        self.caption_dir =os.path.join(caption_dir,caption_model)
        self.ft_name='ft_'+caption_model
            
        train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes,num_camera = \
            self._process_dir_train(self.train_dir)
        val, num_val_pids, num_val_imgs, num_val_clothes, _,_ = \
            self._process_dir_train(self.val_dir)

        query_same, query_diff, gallery, num_test_pids, \
            num_query_imgs_same, num_query_imgs_diff, num_gallery_imgs, \
            num_test_clothes, gallery_idx = self._process_dir_test(self.test_dir)

        num_total_pids = num_train_pids + num_test_pids
        num_test_imgs = num_query_imgs_same + num_query_imgs_diff + num_gallery_imgs
        num_total_imgs = num_train_imgs + num_val_imgs + num_test_imgs
        num_total_clothes = num_train_clothes + num_test_clothes

        
        logger.info("=> PRCC loaded")
        logger.info("Dataset statistics:")
        logger.info("  --------------------------------------------")
        logger.info("  subset      | # ids | # images | # clothes")
        logger.info("  --------------------------------------------")
        logger.info("  train       | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_clothes))
        logger.info("  val         | {:5d} | {:8d} | {:9d}".format(num_val_pids, num_val_imgs, num_val_clothes))
        logger.info("  test        | {:5d} | {:8d} | {:9d}".format(num_test_pids, num_test_imgs, num_test_clothes))
        logger.info("  query(same) | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs_same))
        logger.info("  query(diff) | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs_diff))
        logger.info("  gallery     | {:5d} | {:8d} |".format(num_test_pids, num_gallery_imgs))
        logger.info("  --------------------------------------------")
        logger.info("  total       | {:5d} | {:8d} | {:9d}".format(num_total_pids, num_total_imgs, num_total_clothes))
        logger.info("  --------------------------------------------")
        

        self.train = train
        self.query_same = query_same
        self.query_diff = query_diff
        self.num_gallery_imgs=num_gallery_imgs
        self.num_query_imgs_diff = num_query_imgs_diff
        self.num_query_imgs_same = num_query_imgs_same
        self.gallery = gallery


        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.pid2clothes = pid2clothes
        self.gallery_idx = gallery_idx
        self.num_camera=num_camera
        
        self.num_test_pids = num_test_pids

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir_train(self, dir_path):
        if self.load_sum_ft:
            self.logger.info("loading summery feature")
            sum_id_file=os.path.join(os.path.dirname(self.caption_dir),'train_caption_summary_biometric.json')
            with open(sum_id_file,'r') as f:
                sum_id_info=json.load(f)
                
                
        
        caption_path=os.path.join(self.caption_dir,os.path.basename(dir_path)+'.npz')
        all_caption = np.load(caption_path, allow_pickle=True)
        all_caption_features = all_caption['data']
        all_caption_files = all_caption['metadata']
        ftNums=int(all_caption_features.shape[0]/all_caption_files.shape[0])
        all_caption_files=list(all_caption_files)
        
    
    
        pdirs = glob.glob(osp.join(dir_path, '*'))
        pdirs.sort()

        pid_container = set()
        clothes_container = set()
        camera_container = set()
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
            img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
            for img_dir in img_dirs:
                cam = osp.basename(img_dir)[0]
                camera_container.add(cam)
                if cam in ['A', 'B']:
                    clothes_container.add(osp.basename(pdir))
                else:
                    clothes_container.add(osp.basename(pdir)+osp.basename(img_dir)[0])
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        camera_container=sorted(camera_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id:label for label, clothes_id in enumerate(clothes_container)}
        cam2label = {'A': 0, 'B': 1, 'C': 2}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)
        num_camera=len(camera_container)

        dataset = []
        pid2clothes = np.zeros((num_pids, num_clothes))
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
            for img_dir in img_dirs:
                cam = osp.basename(img_dir)[0]
                label = pid2label[pid]
                camid = cam2label[cam]
                if cam in ['A', 'B']:
                    clothes=osp.basename(pdir)
                else:
                    clothes=osp.basename(osp.basename(pdir)+osp.basename(img_dir)[0])
                clothes_id = clothes2label[clothes]
                 
                relative_path = img_dir[len(dir_path)+1:][:-4].replace('\\', '/')
                caption_feature_index=all_caption_files.index(relative_path)
                caption_feature_load=all_caption_features[caption_feature_index*ftNums:(caption_feature_index+1)*ftNums]
                caption_feature=caption_feature_load[self.bio_index+self.non_bio_index]
              
                        
                if self.load_sum_ft and 0 in self.bio_index:
                    id_ft=np.asarray(sum_id_info[str(pid)][self.ft_name])
                    caption_feature[self.bio_index.index(0)]=id_ft
                    
                        
               
                data={
                    "image_path": img_dir,
                    "pid": label,
                    "camid": camid,
                    "clothes_id": clothes_id,
                    "caption_feature":caption_feature,
                    }
                dataset.append(data) 
                
                pid2clothes[label, clothes_id] = 1            
        
        num_imgs = len(dataset)

        return dataset, num_pids, num_imgs, num_clothes, pid2clothes,num_camera

    def _process_dir_test(self, test_path):
        pdirs = glob.glob(osp.join(test_path, '*'))
        pdirs.sort()

        pid_container = set()
        for pdir in glob.glob(osp.join(test_path, 'A', '*')):
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
        pid_container = sorted(pid_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        cam2label = {'A': 0, 'B': 1, 'C': 2}


        num_pids = len(pid_container)
        num_clothes = num_pids * 2

        query_dataset_same_clothes = []
        query_dataset_diff_clothes = []
        gallery_dataset = []
        
        for cam in ['A', 'B', 'C']:
            pdirs = glob.glob(osp.join(test_path, cam, '*'))
            for pdir in pdirs:
                pid = int(osp.basename(pdir))
                img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
                for img_dir in img_dirs:
                    pid_cls = pid2label[pid]
                    camid = cam2label[cam]
                    data={
                        "image_path": img_dir,
                        "pid": pid_cls,
                        "camid": camid,
                        "clothes_id": 0,
                        }
            
                    if cam == 'A':
                        clothes_id = pid2label[pid] * 2
                        data["clothes_id"]=clothes_id
                        gallery_dataset.append(data) 
                    elif cam == 'B':
                        clothes_id = pid2label[pid] * 2
                        data["clothes_id"]=clothes_id
                        query_dataset_same_clothes.append(data) 
                    else:
                        clothes_id = pid2label[pid] * 2 + 1
                        data["clothes_id"]=clothes_id
                        query_dataset_diff_clothes.append(data) 

        pid2imgidx = {}
       
        for idx, data in enumerate(gallery_dataset):
            if data["pid"] not in pid2imgidx:
                pid2imgidx[pid] = []
            pid2imgidx[pid].append(idx)

        gallery_idx = {}
        random.seed(3)
        for idx in range(0, 10):
            gallery_idx[idx] = []
            for pid in pid2imgidx:
                gallery_idx[idx].append(random.choice(pid2imgidx[pid]))
                 
        num_imgs_query_same = len(query_dataset_same_clothes)
        num_imgs_query_diff = len(query_dataset_diff_clothes)
        num_imgs_gallery = len(gallery_dataset)

        return query_dataset_same_clothes, query_dataset_diff_clothes, gallery_dataset, \
               num_pids, num_imgs_query_same, num_imgs_query_diff, num_imgs_gallery, \
               num_clothes, gallery_idx

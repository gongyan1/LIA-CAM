from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

import os.path as osp
from typing import Callable, Dict, List, Optional, Sequence, Union

import mmengine


@DATASETS.register_module()
class Laneimg_Withangle(BaseSegDataset):

    METAINFO = dict(
        classes=('background', 'lane'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self, **kwargs) -> None:
        super().__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.data_prefix['img_path'])
    
    
    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        assert osp.isfile(self.ann_file)
        lines = mmengine.list_from_file(
            self.ann_file, file_client_args=self.file_client_args)
        for line in lines:
            line = line.strip().split()
            img_name = line[0]
            angle = float(line[1])
            data_info = dict(
                img_path=osp.join(img_dir, img_name + self.img_suffix))
            if ann_dir is not None:
                seg_map = img_name + self.seg_map_suffix
                data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
            data_info['label_map'] = self.label_map
            data_info['reduce_zero_label'] = self.reduce_zero_label
            data_info['seg_fields'] = []
            data_info['angle'] = angle
            data_list.append(data_info)
        
        return data_list

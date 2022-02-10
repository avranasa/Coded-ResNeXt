import warnings
from contextlib import contextmanager
import os
import shutil
import tempfile
from typing import Any, Dict, List, Iterator, Optional, Tuple
import torch, time
from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.utils import check_integrity, extract_archive, verify_str_arg

ARCHIVE_META = {
    'train': ('ILSVRC2012_img_train.tar', '1d675b47d978889d74fa0da5fadfb00e'),
    'val': ('ILSVRC2012_img_val.tar', '29b22e2961454d5413ddabcf34fc5622'),
    'devkit': ('ILSVRC2012_devkit_t12.tar.gz', 'fa75699e90414af021442c21a62c3abf')
    }

META_FILE = "meta.bin"


class MyImageNet(ImageFolder):
    """
        ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.

        Args:
            root (string): Root directory of the ImageNet Dataset.
            split (string, optional): The dataset split, supports ``train``, or ``val``.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
            loader (callable, optional): A function to load an image given its path.

        Attributes:
            classes (list): List of the class name tuples.
            class_to_idx (dict): Dict with items (class_name, class_index).
            wnids (list): List of the WordNet IDs.
            wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
            imgs (list): List of (image path, class_index) tuples
            targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root: str, split: str = 'train', download: Optional[str] = None, **kwargs: Any) -> None:
        if download is True:
            msg = ("The dataset is no longer publicly accessible. You need to "
                   "download the archives externally and place them in the root "
                   "directory.")
            raise RuntimeError(msg)
        elif download is False:
            msg = ("The use of the download flag is deprecated, since the dataset "
                   "is no longer publicly accessible.")
            warnings.warn(msg, RuntimeWarning)

        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "val"))

        #Extract from my google drive address
        self.PATH_2_FILES = '/content/drive/MyDrive/Imagenet'

        self.parse_archives()
        wnid_to_classes = load_meta_file(self.root)[0]

        super(MyImageNet, self).__init__(self.split_folder, **kwargs)
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}

    def parse_archives(self) -> None:
        if not check_integrity(os.path.join(self.root, META_FILE)):
            print(META_FILE, " was not found and is created in directory: ", self.root)
            parse_devkit_archive(self.root, self.PATH_2_FILES)

        if not os.path.isdir(self.split_folder):
            if self.split == 'train':
                parse_train_archive(self.root, self.PATH_2_FILES+'/train')
            elif self.split == 'val':
                parse_val_archive(self.root, self.PATH_2_FILES)

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)


def load_meta_file(root: str, file: Optional[str] = None) -> Tuple[Dict[str, str], List[str]]:
    if file is None:
        file = META_FILE
    file = os.path.join(root, file)

    if check_integrity(file):
        return torch.load(file)
    else:
        msg = ("The meta file {} is not present in the root directory or is corrupted. "
               "This file is automatically created by the ImageNet dataset.")
        raise RuntimeError(msg.format(file, root))


def parse_devkit_archive(root: str, path_2_file: str, file: Optional[str] = None) -> None:
    """Parse the devkit archive of the ImageNet2012 classification dataset and save
    the meta information in a binary file.

    Args:
        root (str):Where the meta.bin file will be created
        path_2_file (str): Directory containing the devkit archive
        file (str, optional): Name of devkit archive. Defaults to
            'ILSVRC2012_devkit_t12.tar.gz'
    """
    import scipy.io as sio

    def parse_meta_mat(devkit_root: str) -> Tuple[Dict[int, str], Dict[str, str]]:
        metafile = os.path.join(devkit_root, "data", "meta.mat")
        meta = sio.loadmat(metafile, squeeze_me=True)['synsets']
        nums_children = list(zip(*meta))[4]
        meta = [meta[idx] for idx, num_children in enumerate(nums_children)
                if num_children == 0]
        idcs, wnids, classes = list(zip(*meta))[:3]
        classes = [tuple(clss.split(', ')) for clss in classes]
        idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
        wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
        return idx_to_wnid, wnid_to_classes

    def parse_val_groundtruth_txt(devkit_root: str) -> List[int]:
        file = os.path.join(devkit_root, "data",
                            "ILSVRC2012_validation_ground_truth.txt")
        with open(file, 'r') as txtfh:
            val_idcs = txtfh.readlines()
        return [int(val_idx) for val_idx in val_idcs]

    @contextmanager
    def get_tmp_dir() -> Iterator[str]:
        tmp_dir = tempfile.mkdtemp()
        try:
            yield tmp_dir
        finally:
            shutil.rmtree(tmp_dir)

    archive_meta = ARCHIVE_META["devkit"]
    if file is None:
        file = archive_meta[0]
    

    with get_tmp_dir() as tmp_dir:
        extract_archive(os.path.join(path_2_file, file), tmp_dir)

        devkit_root = os.path.join(tmp_dir, "ILSVRC2012_devkit_t12")
        idx_to_wnid, wnid_to_classes = parse_meta_mat(devkit_root)
        val_idcs = parse_val_groundtruth_txt(devkit_root)
        val_wnids = [idx_to_wnid[idx] for idx in val_idcs]

        torch.save((wnid_to_classes, val_wnids), os.path.join(root, META_FILE))


def parse_train_archive(root: str, path_2_file: str, folder: str = "train") -> None:#changed
    """Parse the train images archive of the ImageNet2012 classification dataset and
    prepare it for usage with the ImageNet dataset.

    Args:
        root (str): Directory where the 'train' folder will be created that will store the 1000 folders with the 
                    training images for each class
        path_2_file: Contains one tar per training class containing the training images per class
        folder (str, optional): Optional name for train images folder. Defaults to
            'train'
    """
    archive_meta = ARCHIVE_META["train"]
    md5 = archive_meta[1]

    train_root = os.path.join(root, folder)#where the images will be stored    
    os.mkdir(train_root)
    Name_folders =  [ train_root + '/'+archive.split('.tar')[0] for archive in os.listdir(path_2_file)]
    From_tars = [ path_2_file+'/'+archive  for archive in os.listdir(path_2_file)]
    N_samples_total = 0 
    for i, to_folder in enumerate(Name_folders):
        os.mkdir(to_folder)
        print('Created directory ', to_folder, ' and putting images of class ', i+1)
        if i <= 11:
            time.sleep(10)
        extract_archive(From_tars[i], to_folder, remove_finished=False)
        N_samples = len( os.listdir(to_folder))
        N_samples_total += N_samples
        print(' ->Number of training samples for this class', N_samples ,' and in total:',N_samples_total)


def parse_val_archive( root: str, path_2_file: str, file: Optional[str] = None, wnids: Optional[List[str]] = None, folder: str = "val") -> None:
    """Parse the validation images archive of the ImageNet2012 classification dataset
    and prepare it for usage with the ImageNet dataset.

    Args:
        root (str): Directory where the 'folder'(default folder=val) will be created that will store the validation images
        path_2_file: Contains one tar per training class containing the validation images per class
        file (str, optional): Name of validation images archive. Defaults to
            'ILSVRC2012_img_val.tar'
        wnids (list, optional): List of WordNet IDs of the validation images. If None
            is given, the IDs are loaded from the meta file in the root directory
        folder (str, optional): Optional name for validation images folder. Defaults to
            'val'
    """
    archive_meta = ARCHIVE_META["val"]
    if file is None:
        file = archive_meta[0]
    if wnids is None:
        wnids = load_meta_file(root)[1]

    val_root = os.path.join(root, folder)
    path_to_tar = os.path.join(path_2_file, file) 
    extract_archive(path_to_tar, val_root)

    images = sorted([os.path.join(val_root, image) for image in os.listdir(val_root)])
    print('The folder ', val_root, ' was created containing ', len(images), ' images.')

    for wnid in set(wnids):
        os.mkdir(os.path.join(val_root, wnid))

    for wnid, img_file in zip(wnids, images):
        shutil.move(img_file, os.path.join(val_root, wnid, os.path.basename(img_file)))







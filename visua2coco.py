import argparse
from config import get_cfg_defaults
from collections import OrderedDict, defaultdict


def get_annots(cfg, mode):
    annots = defaultdict(list)
    fn = cfg.CROPPED_ANNOT_FILE if mode == 'train' else cfg.CROPPED_ANNOT_FILE_TEST
    with open(fn, 'r') as f:
        for line in f:
            elems = line.rstrip().split(',')
            jpg_file, groundtruth = elems[0], list(map(int, elems[1:]))
            annots[jpg_file] = groundtruth
    return annots


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', dest='cfg_file',
                        default=None, type=str, help='Path to config file.')
    parser.add_argument('--mode', dest='mode',
                        default='train', type=str, help='train or test.')
    parser.add_argument('--output_dir', dest='output_dir',
                        default='.', type=str, help='Path to output directory.')
    args = parser.parse_args()

    cfg = get_cfg_defaults()

    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)

    annots = get_annots(cfg, args.mode)
    print(annots)

    # image_idxes = get_image_idxes(annots)





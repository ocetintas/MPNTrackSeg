_SPLITS = {}

#################
# MOT15
#################

# sequences used for training
mot15_train_seqs = ['KITTI-17', 'ETH-Sunnyday', 'ETH-Bahnhof', 'PETS09-S2L1', 'TUD-Stadtmitte']

# Additional train sequences not used for tranining (since they are present in MOT17 etc.)
add_mot15_train_seqs = ['ETH-Pedcross2', 'TUD-Campus', 'KITTI-13', 'Venice-2', 'ADL-Rundle-8', 'ADL-Rundle-6']
_SPLITS['mot15_train_gt'] = {'2DMOT2015/train': [f'{seq}-GT' for seq in mot15_train_seqs]}
_SPLITS['mot15_train'] = {'2DMOT2015/train': mot15_train_seqs + add_mot15_train_seqs}

# Test sequences
test_seqs =  ['TUD-Crossing', 'PETS09-S2L2', 'ETH-Jelmoli', 'ETH-Linthescher', 'ETH-Crossing', 'AVG-TownCentre',
                  'ADL-Rundle-1', 'ADL-Rundle-3', 'KITTI-16', 'KITTI-19', 'Venice-1']
_SPLITS['mot15_test'] = {'2DMOT2015/test': test_seqs}


#################
# MOT17
#################
dets = ('DPM', 'FRCNN', 'SDP')

# Train sequences:
train_seq_nums=  (2, 4, 5, 9, 10, 11, 13)
_SPLITS['mot17_train_gt'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-GT' for seq_num in train_seq_nums]}
_SPLITS['mot17_train'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in train_seq_nums for det in dets]}

# Cross Validation splits
_SPLITS['mot17_split_1_train_gt'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-GT' for seq_num in (2, 5, 9, 10, 13)]}
_SPLITS['split_1_val'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in (4, 11) for det in dets]}

_SPLITS['mot17_split_2_train_gt'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-GT' for seq_num in (2, 4, 11, 10, 13)]}
_SPLITS['split_2_val'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in (5, 9) for det in dets]}

_SPLITS['mot17_split_3_train_gt'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-GT' for seq_num in (4, 5, 9, 11)]}
_SPLITS['split_3_val'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in (2, 10, 13) for det in dets]}

_SPLITS['debug'] = {'MOT17Labels/train': ['MOT17-02-FRCNN']}


# Test sequences
test_seq_nums=  (1, 3, 6, 7, 8, 12, 14)
_SPLITS['mot17_test'] = {'MOT17Labels/test': [f'MOT17-{seq_num:02}-{det}' for seq_num in test_seq_nums for det in dets]}

# Combinations:
_SPLITS['all_train'] = {**_SPLITS['mot17_train_gt'], **_SPLITS['mot15_train_gt']}
_SPLITS['all_test'] = {**_SPLITS['mot17_test'], **_SPLITS['mot15_test']}



#################
# MOTS20
#################

# Training set
train_seq_nums = (2, 5, 9, 11)

# Test set
test_seq_nums = (1, 6, 7, 12)

_SPLITS['mots20_train_gt'] = {'MOTS20/train/': [f'MOTS20-{seq_num:02}-GT' for seq_num in train_seq_nums]}
_SPLITS['mots20_train'] = {'MOTS20/train/': [f'MOTS20-{seq_num:02}' for seq_num in train_seq_nums]}
_SPLITS['mots20_test'] = {'MOTS20/test/': [f'MOTS20-{seq_num:02}' for seq_num in test_seq_nums]}
_SPLITS['mots20_test_debug'] = {'MOTS20/test/': [f'MOTS20-{seq_num:02}' for seq_num in (1,)]}

_SPLITS['mots20_train_split'] = {'MOTS20/train/': [f'MOTS20-{seq_num:02}' for seq_num in (5, 9, 11)]}
_SPLITS['mots20_val_split'] = {'MOTS20/train/': [f'MOTS20-{seq_num:02}' for seq_num in (2,)]}


_SPLITS['mots20_train_gt_split'] = {'MOTS20/train/': [f'MOTS20-{seq_num:02}-GT' for seq_num in (5, 9, 11)]}
_SPLITS['mots20_single'] = {'MOTS20/train/': [f'MOTS20-09']}



#################
# KITTI-MOTS
#################

# Training set
train_seq_nums = (13, 16, 17, 19)

# Test set
test_seq_nums = ()

_SPLITS['kittimots_train_gt'] = {'KITTIMOTS/train/': [f'KITTIMOTS-{seq_num:02}-GT' for seq_num in train_seq_nums]}
_SPLITS['kittimots_train'] = {'KITTIMOTS/train/': [f'KITTIMOTS-{seq_num:02}' for seq_num in train_seq_nums]}
_SPLITS['kittimots_test'] = {'KITTIMOTS/test/': [f'KITTIMOTS-{seq_num:02}' for seq_num in test_seq_nums]}

_SPLITS['kittimots_train_split'] = {'KITTIMOTS/train/': [f'KITTIMOTS-{seq_num:02}' for seq_num in (0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20)]}
_SPLITS['kittimots_val_split'] = {'KITTIMOTS/train/': [f'KITTIMOTS-{seq_num:02}' for seq_num in (2, 7, 8, 10, 13, 14, 16, 18)]}
_SPLITS['kittimots_alltrain'] = {'KITTIMOTS/train/': [f'KITTIMOTS-{seq_num:02}' for seq_num in range(0, 21)]}
_SPLITS['kittimots_test'] = {'KITTIMOTS/test/': [f'KITTIMOTS-{seq_num:02}' for seq_num in range(0, 29) if seq_num not in (1, 6)]}
_SPLITS['kittimots_val_all'] = {'KITTIMOTS/train/': [f'KITTIMOTS-{seq_num:02}' for seq_num in range(0, 21) if seq_num not in (6, )]}

_SPLITS['allmots_train_split'] = {'KITTIMOTS/train/': [f'KITTIMOTS-{seq_num:02}' for seq_num in (0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20)],
                                  'MOTS20/train/': [f'MOTS20-{seq_num:02}' for seq_num in (2, 5, 9, 11)]}
_SPLITS['allmots_val_split'] = {'KITTIMOTS/train/': [f'KITTIMOTS-{seq_num:02}' for seq_num in (2, 7, 8, 10, 13, 14, 16, 18)]}

_SPLITS['kittimots_smalltrain_split'] = {'KITTIMOTS/train/': [f'KITTIMOTS-{seq_num:02}' for seq_num in (13, 16, 17)]}
_SPLITS['kittimots_smallval_split'] = {'KITTIMOTS/train/': [f'KITTIMOTS-{seq_num:02}' for seq_num in (19,)]}

_SPLITS['kittimots_test_debug'] = {'KITTIMOTS/test/': [f'KITTIMOTS-{seq_num:02}' for seq_num in (0,)]}
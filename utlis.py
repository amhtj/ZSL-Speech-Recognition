import numpy as np

import torch
import torch.nn as nn

import tensorflow 
import tensorflow_hub as hub

from sklearn import preprocessing

import sys
import os
from pathlib import Path

import random

from src.zeroshot_networks.vae import EncoderTemplate, DecoderTemplate
from src.dataset_loaders._data_loader import DATA_LOADER as dataloader

# Data loader

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i

    return mapped_label

class DATA_LOADER(object):
    def __init__(self, dataset, aux_datasource, device='cuda'):

        print("The current working directory is")
        print(os.getcwd())
        folder = str(Path(os.getcwd()))
        if folder[-5:] == 'model':
            diploma_directory = Path(os.getcwd()).parent
        else:
            diploma_directory = folder

        print('Diploma dir:')
        print(diploma_directory)
        data_path = str(diploma_directory) + '/data'
        print('Data Path')
        print(data_path)
        sys.path.append(data_path)

        self.data_path = data_path
        self.device = device
        self.dataset = dataset
        self.auxiliary_data_source = aux_datasource

        self.all_data_sources = ['libri_features'] + [self.auxiliary_data_source]

        if self.dataset == 'google_speech_commands':
            self.datadir = self.data_path + '/GSC/'
        elif self.dataset == 'librispeech':
            self.datadir = self.data_path + '/LIBRI/'

        self.read_matdataset()
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def next_batch(self, batch_size):

        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.data['train_seen']['jasper_features'][idx]
        batch_label =  self.data['train_seen']['labels'][idx]
        batch_att = self.aux_data[batch_label]
        return batch_label, [batch_feature, batch_att]

    def gen_next_batch(self, batch_size, dset_part='train'):
        if dset_part == 'train':
            features = self.data['train_seen']['jasper_features']
            labels = self.data['train_seen']['labels']
        elif dset_part == 'test':
            features = self.data['test_unseen']['jasper_features']
            labels = self.data['test_unseen']['labels']
        else:
            raise ValueError('Your dataset do not supported')

        iter_len = len(features) // batch_size + 1
        for current_batch in range(iter_len):
            current_idx = current_batch * batch_size
            end_idx = current_idx + batch_size

            batch_features = features[current_idx:end_idx]
            batch_label = labels[current_idx:end_idx]
            batch_attr = self.aux_data[batch_label]

            yield batch_label, [batch_features, batch_attr]

        scaler = preprocessing.MinMaxScaler()

        train_feature = scaler.fit_transform(feature[trainval_loc])
        test_seen_feature = scaler.fit_transform(feature[test_seen_loc])
        test_unseen_feature = scaler.fit_transform(feature[test_unseen_loc])

        train_feature = torch.from_numpy(train_feature).float().to(self.device)
        test_seen_feature = torch.from_numpy(test_seen_feature).float().to(self.device)
        test_unseen_feature = torch.from_numpy(test_unseen_feature).float().to(self.device)

        train_label = torch.from_numpy(label[trainval_loc]).long().to(self.device)
        test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long().to(self.device)
        test_seen_label = torch.from_numpy(label[test_seen_loc]).long().to(self.device)

        self.seenclasses = torch.from_numpy(np.unique(train_label.cpu().numpy())).to(self.device)
        self.novelclasses = torch.from_numpy(np.unique(test_unseen_label.cpu().numpy())).to(self.device)
        self.ntrain = train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.novelclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()

        self.train_mapped_label = map_label(train_label, self.seenclasses)

        self.data = {}
        self.data['train_seen'] = {}
        self.data['train_seen']['jasper_features'] = train_feature
        self.data['train_seen']['labels'] = train_label
        self.data['train_seen'][self.auxiliary_data_source] = self.aux_data[train_label]


        self.data['train_unseen'] = {}
        self.data['train_unseen']['jasper_features'] = None
        self.data['train_unseen']['labels'] = None

        self.data['test_seen'] = {}
        self.data['test_seen']['jasper_features'] = test_seen_feature
        self.data['test_seen']['labels'] = test_seen_label

        self.data['test_unseen'] = {}
        self.data['test_unseen']['jasper_features'] = test_unseen_feature
        self.data['test_unseen'][self.auxiliary_data_source] = self.aux_data[test_unseen_label]
        self.data['test_unseen']['labels'] = test_unseen_label

        self.novelclass_aux_data = self.aux_data[self.novelclasses]
        self.seenclass_aux_data = self.aux_data[self.seenclasses]


# Data loader


        iter_len = len(features) // batch_size + 1
        for current_batch in range(iter_len):
            current_idx = current_batch * batch_size
            end_idx = current_idx + batch_size

            batch_features = features[current_idx:end_idx]
            batch_label = labels[current_idx:end_idx]
            batch_attr = self.aux_data[batch_label]

            yield batch_label, [batch_features, batch_attr]

        scaler = preprocessing.MinMaxScaler()

        train_feature = scaler.fit_transform(feature[trainval_loc])
        test_seen_feature = scaler.fit_transform(feature[test_seen_loc])
        test_unseen_feature = scaler.fit_transform(feature[test_unseen_loc])

        train_feature = torch.from_numpy(train_feature).float().to(self.device)
        test_seen_feature = torch.from_numpy(test_seen_feature).float().to(self.device)
        test_unseen_feature = torch.from_numpy(test_unseen_feature).float().to(self.device)

        train_label = torch.from_numpy(label[trainval_loc]).long().to(self.device)
        test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long().to(self.device)
        test_seen_label = torch.from_numpy(label[test_seen_loc]).long().to(self.device)

        self.seenclasses = torch.from_numpy(np.unique(train_label.cpu().numpy())).to(self.device)
        self.novelclasses = torch.from_numpy(np.unique(test_unseen_label.cpu().numpy())).to(self.device)
        self.ntrain = train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.novelclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()

        self.train_mapped_label = map_label(train_label, self.seenclasses)

        self.data = {}
        self.data['train_seen'] = {}
        self.data['train_seen']['jasper_features'] = train_feature
        self.data['train_seen']['labels'] = train_label
        self.data['train_seen'][self.auxiliary_data_source] = self.aux_data[train_label]


        self.data['train_unseen'] = {}
        self.data['train_unseen']['jasper_features'] = None
        self.data['train_unseen']['labels'] = None

        self.data['test_seen'] = {}
        self.data['test_seen']['jasper_features'] = test_seen_feature
        self.data['test_seen']['labels'] = test_seen_label

        self.data['test_unseen'] = {}
        self.data['test_unseen']['jasper_features'] = test_unseen_feature
        self.data['test_unseen'][self.auxiliary_data_source] = self.aux_data[test_unseen_label]
        self.data['test_unseen']['labels'] = test_unseen_label

        self.novelclass_aux_data = self.aux_data[self.novelclasses]
        self.seenclass_aux_data = self.aux_data[self.seenclasses]
 
# Feature extraction 

class StatsPoolLayer(nn.Module):
    def __init__(self, feat_in, pool_mode='xvector'):
        super().__init__()
        self.feat_in = 0
        if pool_mode == 'gram':
            gram = True
            super_vector = False
        elif pool_mode == 'superVector':
            gram = True
            super_vector = True
        else:
            gram = False
            super_vector = False

        if gram:
            self.feat_in += feat_in ** 2
        else:
            self.feat_in += 2 * feat_in

        if super_vector and gram:
            self.feat_in += 2 * feat_in

        self.gram = gram
        self.super = super_vector

        class MaskedConv1d(nn.Module):
            __constants__ = ["use_conv_mask", "real_out_channels", "heads"]

            def __init__(
                    self,
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=1,
                    heads=-1,
                    bias=False,
                    use_mask=True,
                    quantize=False,
            ):
                super(MaskedConv1d, self).__init__()

                self.real_out_channels = out_channels
                if heads != -1:
                    in_channels = heads
                    out_channels = heads
                    groups = heads

                self._padding = padding

                if type(padding) in (tuple, list):
                    self.pad_layer = nn.ConstantPad1d(padding, value=0.0)

                else:
                    self.pad_layer = None

                    self.conv = nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias,
                    )
                self.use_mask = use_mask
                self.heads = heads

                self.same_padding = (self.conv.stride[0] == 1) and (
                        2 * self.conv.padding[0] == self.conv.dilation[0] * (self.conv.kernel_size[0] - 1)
                )
                if self.pad_layer is None:
                    self.same_padding_asymmetric = False
                else:
                    self.same_padding_asymmetric = (self.conv.stride[0] == 1) and (
                            sum(self._padding) == self.conv.dilation[0] * (self.conv.kernel_size[0] - 1)
                    )

                if self.use_mask:
                    self.max_len = 0
                    self.lens = None

                    def extract_features(self, x, pooling_mask=None):

                        if pooling_mask is not None:
                            x[pooling_mask] = 0

                        x_conv = self.pos_conv(x.transpose(1, 2))
                        x_conv = x_conv.transpose(1, 2)
                        x += x_conv

                        if not self.layer_norm_first:
                            x = self.layer_norm(x)

                        x = self.feature_dropout(x)


                        x = x.transpose(0, 1)

                        x = self.transformer_encoder(x, src_key_pooling_mask=pooling_mask)


                        x = x.transpose(0, 1)

                        return x

# USE 

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
model = hub.load(module_url)
print ("module %s loaded" % module_url)
def embed(input):
  return model(input)

logging.set_verbosity(logging.ERROR)

message_embeddings = embed(messages)

for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
    message_embedding_snippet = ", ".join(
        (str(x) for x in message_embedding[:3]))

# Word error rate

logging.root.setLevel(logging.INFO)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--hypo", help="hypo transcription", required=True)
    parser.add_argument(
        "-r", "--reference", help="reference transcription", required=True
    )
    return parser

def compute_wer(ref_uid_to_tra, hyp_uid_to_tra, g2p):
    d_cnt = 0
    w_cnt = 0
    w_cnt_h = 0
    for uid in hyp_uid_to_tra:
        ref = ref_uid_to_tra[uid].split()
        if g2p is not None:
            hyp = g2p(hyp_uid_to_tra[uid])
            hyp = [p for p in hyp if p != "'" and p != " "]
            hyp = [p[:-1] if p[-1].isnumeric() else p for p in hyp]
        else:
            hyp = hyp_uid_to_tra[uid].split()
        d_cnt += editdistance.eval(ref, hyp)
        w_cnt += len(ref)
        w_cnt_h += len(hyp)
    wer = float(d_cnt) / w_cnt
    logger.debug(
        (
            f"wer = {wer * 100:.2f}%; num. of ref words = {w_cnt}; "
            f"num. of hyp words = {w_cnt_h}; num. of sentences = {len(ref_uid_to_tra)}"
        )
    )
    return wer


def main():
    args = get_parser().parse_args()

    errs = 0
    count = 0
    with open(args.hypo, "r") as hf, open(args.reference, "r") as rf:
        for h, r in zip(hf, rf):
            h = h.rstrip().split()
            r = r.rstrip().split()
            errs += editdistance.eval(r, h)
            count += len(r)

    logger.info(f"UER: {errs / count * 100:.2f}%")


if __name__ == "__main__":
    main()


def load_tra(tra_path):
    with open(tra_path, "r") as f:
        uid_to_tra = {}
        for line in f:
            uid, tra = line.split(None, 1)
            uid_to_tra[uid] = tra
    logger.debug(f"loaded {len(uid_to_tra)} utterances from {tra_path}")
    return uid_to_tra

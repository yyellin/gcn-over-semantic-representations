"""
Data structure for encapsulating TACRED input

  Authors: Jonathan Yellin
  Status: prototype

"""

from collections import namedtuple
import torch
from utils.torch_utils import  get_long_tensor, set_cuda, change_lr, get_optimizer


class Input(namedtuple('Input', 'batch_size, word, mask, pos, ner, coref, ucca_enc, len, head, ucca_head, ucca_multi_head, ucca_dist_from_mh_path, subj_p, obj_p, id, orig_idx')):
    """
    'Input' objects are similar to 'Batch'; however, all fields that need to be in tensor form, are captured as tensors. Fields
    that do not need to be in Tensor form continue to be represented in their native formats
    """

    @classmethod
    def unpack_batch(cls, batch, cuda):

        words = set_cuda(get_long_tensor(batch.word, batch.batch_size), cuda)
        masks = set_cuda(torch.eq(words, 0), cuda)
        pos = set_cuda(get_long_tensor(batch.pos, batch.batch_size), cuda)
        ner = set_cuda(get_long_tensor(batch.ner, batch.batch_size), cuda)
        coref = set_cuda(get_long_tensor(batch.coref, batch.batch_size), cuda)
        ucca_enc = set_cuda(get_long_tensor(batch.ucca_enc, batch.batch_size), cuda)

        rel = set_cuda(torch.LongTensor(batch.rel), cuda)

        input = Input(batch_size=batch.batch_size,
                      word=words,
                      mask=masks,
                      pos=pos,
                      ner=ner,
                      coref=coref,
                      ucca_enc=ucca_enc,
                      len=batch.len,
                      head=batch.head,
                      ucca_head=batch.ucca_head,
                      ucca_multi_head=batch.ucca_multi_head,
                      ucca_dist_from_mh_path=batch.ucca_dist_from_mh_path,
                      subj_p=batch.subj_p,
                      obj_p=batch.obj_p,
                      id=batch.id,
                      orig_idx=batch.orig_idx)

        return input, rel

    pass

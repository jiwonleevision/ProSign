import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.nn import all_gather

import time
import datetime
from collections import defaultdict, deque
import os


def count_parameters_in_MB(model):
    return sum(v.numel() for v in model.parameters() if v.requires_grad) / 1e6

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank, timeout=datetime.timedelta(minutes=10))
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def compute_recall(sim_matrix):
    # sim_matrix: [N, N] (행: 비디오, 열: 텍스트)
    # 정답은 대각선(0,0), (1,1) ... (n,n)에 있음
    n_samples = sim_matrix.size(0)
    targets = torch.arange(n_samples).to(sim_matrix.device)
    
    # 정렬하여 순위(Rank) 확인
    _, indices = sim_matrix.sort(descending=True, dim=-1)
    
    # 각 행(비디오)에서 정답(텍스트)이 몇 번째에 있는지 찾기
    prediction_ranks = (indices == targets.unsqueeze(-1)).nonzero()[:, 1]
    
    r1 = (prediction_ranks < 1).float().mean() * 100
    r5 = (prediction_ranks < 5).float().mean() * 100
    r10 = (prediction_ranks < 10).float().mean() * 100
    
    return {"R1": r1.item(), "R5": r5.item(), "R10": r10.item()}

def compute_recall_multi_positive(sim_matrix, query_pronunciations, target_pronunciations):
    """
    sim_matrix: [Nq, Nt]  (행: query video, 열: target text)
    query_pronunciations: list[str] 길이 Nq
    target_pronunciations: list[str] 길이 Nt
    """

    device = sim_matrix.device
    _, indices = sim_matrix.sort(descending=True, dim=-1)  # [Nq, Nt]

    r1_list, r5_list, r10_list = [], [], []

    for i in range(sim_matrix.size(0)):
        q_pron = query_pronunciations[i]

        # query와 같은 pronunciation을 가진 모든 정답 인덱스
        positive_indices = {
            j for j, t_pron in enumerate(target_pronunciations)
            if t_pron == q_pron
        }

        ranked = indices[i]

        top1 = ranked[:1].tolist()
        top5 = ranked[:5].tolist()
        top10 = ranked[:10].tolist()

        r1_list.append(any(idx in positive_indices for idx in top1))
        r5_list.append(any(idx in positive_indices for idx in top5))
        r10_list.append(any(idx in positive_indices for idx in top10))

    r1 = torch.tensor(r1_list, dtype=torch.float32, device=device).mean() * 100
    r5 = torch.tensor(r5_list, dtype=torch.float32, device=device).mean() * 100
    r10 = torch.tensor(r10_list, dtype=torch.float32, device=device).mean() * 100

    return {"R1": r1.item(), "R5": r5.item(), "R10": r10.item()}

def all_gather_batch(tensors):
    """
    모든 GPU에서 텐서를 모아 하나로 합칩니다.
    """
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensors

    # 또는 최신 PyTorch 버전 사용 시:
    tensor_list = [torch.zeros_like(tensors) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensors)
    
    return torch.cat(tensor_list, dim=0)

def all_gather_object_list(local_list):
    if not dist.is_available() or not dist.is_initialized():
        return local_list

    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_list)

    merged = []
    for part in gathered:
        merged.extend(part)
    return merged

def gather_features_with_grad(x):
    gathered = all_gather(x)          # tuple/list of tensors from all ranks
    x_all = torch.cat(gathered, dim=0)
    return x_all

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
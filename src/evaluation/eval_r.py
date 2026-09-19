




def i2t_SCAN_NN(sims, return_ranks=False):
    """
    Images->Text (Image Annotation)
    sims: (N, N) matrix of similarity im-cap
    """
    npts = sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)

    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]  # 对每个图像与文本的相似度从大到小排序，返回排序后的索引
        rank = np.where(inds == index)[0][0]  # 找到对应文本描述在排序后的索引中的位置
        ranks[index] = rank  # 存储当前图像的最佳排名
        top1[index] = inds[0]  # 存储最相似的文本描述的索引

    # 计算指标
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r20 = 100.0 * len(np.where(ranks < 20)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    r70 = 100.0 * len(np.where(ranks < 70)[0]) / len(ranks)
    r100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)

    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, r20, r50, r70, r100, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, r20, r50, r70, r100, medr, meanr)

def t2i_SCAN_NN(sims, return_ranks=False):
    """
    Text->Images (Text Annotation)
    sims: (N, N) matrix of similarity text-image
    """
    npts = sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)

    sims=sims.T

    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]  # 对每个文本与所有图像的相似度进行降序排序，并返回排序后的索引
        rank = np.where(inds == index)[0][0]  # 找到对应图像在排序后的索引中的位置
        ranks[index] = rank  # 存储当前文本的最佳排名
        top1[index] = inds[0]  # 存储最相似的图像的索引

    # 计算指标
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r20 = 100.0 * len(np.where(ranks < 20)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    r70 = 100.0 * len(np.where(ranks < 70)[0]) / len(ranks)
    r100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)

    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, r20, r50, r70, r100, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, r20, r50, r70, r100, medr, meanr)
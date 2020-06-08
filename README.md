

## 训练
1. 下载voc2007到data文件夹下，目录结构如下

```sh
├─Annotations
├─ImageSets
│  ├─Action
│  ├─Layout
│  ├─Main
│  └─Segmentation
├─JPEGImages
├─labels
├─SegmentationClass
└─SegmentationObject
```
2. 使用`voc_label`生成标签
3. 下载`vgg16_bn-6c64b313.pth`到`weights`文件夹下,这个可以百度一下就能找到
4. `python train.py` 开始训练

'''
## 物体探测(Object Detection)
物体探测的目标是输入一张图片，要求输出图像中包含的对象类别和位置
  ![](https://upload-images.jianshu.io/upload_images/2709767-60e9f5a4dbdd8b4d.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)
  object detection 可以分开看成两个任务：
  1. 识别存在对象的区域
  2. 识别出区域内对象的类别
## 物体探测的方法
### 1. 暴力检测法
  最笨的方法就是，遍历图片每个像素，然后搜索不同大小的**框**，逐一检测是否存在某个对象。显然这种会存在巨量的框，效率低到令人发指。
### 2. 候选框检测法(Two Stage Detection)
后来就有聪明人提出根据图像的纹理颜色特征提出**框**,其中比较著名的方法就是**selective search**，它从一张图片中提取出大约2000个框，然后对每个框进行识别分类，显然这相较于第一种方法效率有了极大的提升
![selective search例子](https://upload-images.jianshu.io/upload_images/944794-4b2a557c7411a305.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这方面比较著名的网络是RCNN家族,第一个版本的RCNN性能只有0.02 FPS(每秒处理几张图)，后来的Fast RCNN提高到了0.43 FPS，Faster RCNN提到了5 FPS。
其中，虽然Faster Rcnn 使用RPN代替了SS，但是其本质没有改变都是两步操作：先提出候选框，然后进行分类。于是这类网络被称为Two Stage Detection
### 3. 一步检测法(One stage Detection)
尽管上面的Fater RCNN达到了5FPS，但是还是不能满足我们的实际需求。我们要对上面提出的方法进行改进。首先我们分析Two Stage的检测方法为什么慢？
1. 太多的候选框，一张图里面物体的个数是有限的，但是Two Stage动则给出几百上千个候选框,每个框都要进行特征提取，其中大部分候选框里没有物体，浪费了计算资源。
2. 候选框和类别识别是分开的两个步骤，尤其是RCNN和Fast RCNN还是使用了在cpu上运行的低效的**Selective Search**，Faster RCNN中的RPN不过是相当于GPU版的**Selective Search**

针对上面的缺点，有人(RGB)提出了YOLO网络,YOLO将候选框和类别统一为一个回归任务。下面我们仔细来看YOLO是怎么做的

## YOLO的基本思想
YOLO的意思是YOU  Only Look Once,它将候选框的提出和对象识别合并为了一个步骤。
![image](https://upload-images.jianshu.io/upload_images/944794-3f7a4f0f3dac230b.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
YOLO将图片划分为7x7的网格，每个网格预测出两个框，那么整张图一共7x7x2=98个框。

### YOLO和RCNN的对话
```
RCNN：我们先来研究一下图片，嗯，这些位置很可能存在一些对象，你们对这些位置再检测一下看到底是哪些对象在里面。
YOLO：我们把图片大致分成98个区域，每个区域看下有没有对象存在，以及具体位置在哪里。
RCNN：你这么简单粗暴真的没问题吗？
YOLO：当然没有......咳，其实是有一点点问题的，准确率要低一点，但是我非常快！快！快！
RCNN：为什么你用那么粗略的候选区，最后也能得到还不错的bounding box呢？
YOLO：你不是用过边框回归吗？我拿来用用怎么不行了。
```


从上面我们可以知道，RCNN系列会首先提出一些候选框，然后对这些框进行微调(回归),使得候选框接近真是框，YOLO的思想是反正最后还要微调，我只要确定大致范围，然后直接回归出来不得了。省的费劲找候选框了。

![边框回归](https://upload-images.jianshu.io/upload_images/2709767-8734522ef384bd4b.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)

当然边框回归首先你要有一个大致的框的位置，那么这个框的大致位置是怎么来的？YOLO采用了一个比较巧妙的办法，物体位置的中心点。讲YOLO的文章一般都会有这么一句话：将一幅图像分成 SxS 个网格（grid cell），**如果某个 object 的中心落在这个网格中，则这个网格就负责预测这个 object。**
起初看这句话很迷惑，啥意思，这是干啥呢，为什么要这么做啊？搞不懂啊。
我个人觉得，其背后的含义是给定训练数据，我们需要学习如何预测物体的中心点位置
![image](https://upload-images.jianshu.io/upload_images/944794-19af49fce565cc55.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
例如，上面的狗狗的中心点落在(5,2),正常情况下以(5,2)为中心点预测框的IOU肯定比以(5,1),(5,3)等为中心预测的框的IOU高。所以只要我们能够学习如何更好预测中心点的位置，就有更大概率更好的预测出物体的位置(边框)

## 结构
出去最后的候选框步骤，YOLO的结构很简单，就是一个单纯的backbone后面接一个reshape，reshape到7x7x30
![image](https://upload-images.jianshu.io/upload_images/944794-571fe2578d4f0950.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
## 输入和输出
YOLO的输入要求原图缩放到448x448大小，输出是7x7x30的tensor

根据YOLO的射击，输入图像被划分为7x7的网格，每个网格预测两个框(2x5),每个框含有5个值，分别是边界框的中心x,y（相对于所属网格的边界），边界框的宽高w,h（相对于原始输入图像的宽高的比例），外加20个分类的概率
![image](https://upload-images.jianshu.io/upload_images/944794-ad57659c233f5ad1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 讨论
1. **YOLO的主要缺点**
    每个30维向量中只有一组（20个）对象分类的概率，也就只能预测出一个对象。所以输出的 7*7=49个 30维向量，最多表示出49个对象。 总共有 49*2=98 个候选区（bounding box）
    YOLO对相互靠的很近的物体，还有很小的群体 检测效果不好，这是因为一个网格中只预测了两个框，并且只属于一类。对测试图像中，同一类物体出现的新的不常见的长宽比和其他情况是。泛化能力偏弱。
    由于损失函数的问题，定位误差是影响检测效果的主要原因。尤其是大小物体的处理上，还有待加强

2. **YOLO的bounding box和Faster
    RCNN的Anchor**
    Faster RCNN等一些算法采用每个grid中手工设置n个Anchor（先验框，预先设置好位置的bounding box）的设计，每个Anchor有不同的大小和宽高比。YOLO的bounding box看起来很像一个grid中2个Anchor，但它们不是。YOLO并没有预先设置2个bounding box的大小和形状，也没有对每个bounding box分别输出一个对象的预测。它的意思仅仅是对一个对象预测出2个bounding box，选择预测得相对比较准的那个。
    训练开始阶段，网络预测的bounding box是随机大小的，但训练过程中总是选择IOU相对好一些的那个，随着训练的进行，每个bounding box会逐渐擅长对某些情况的预测（可能是对象大小、宽高比、不同类型的对象等）。所以，这是一种进化或者非监督学习的思想
    > 如果一开始就选择了更好的、更有代表性的先验框维度，那么网络就更容易学到准确的预测位置。
    为了使网络更易学到准确的预测位置，所以再YOLO V2作者使用了K-means聚类方法类训练bounding boxes，可以自动找到更好的框宽高维度

## YOLO的实现和训练
> 项目地址 https://github.com/Hexmagic/pytorch_yolov1.git
### 模型结构
```python
class YoLo(nn.Module):
    def __init__(self, features, num_classes=20):
        super(YoLo, self).__init__()
        self.features = features
        self.classify = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                                      nn.LeakyReLU(0.1, True),
                                      nn.Linear(4096, 1470))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        x = torch.sigmoid(x)
        return x.view(-1, 7, 7, 30)
```
其中feature是vgg的卷积部分，最好使用imagenet预训练的vgg加载初始化，论文中作者就是使用vgg现在imagenet上作了预训练。这一步很重要!!!,不然直接训练你会发现class loss不容易下降的，**要先学会分类才更容易进行探测**

### 数据集
数据的源格式是下面这个样子的：
```
1 0.48652694610778446 0.7020000000000001 0.5538922155688623 0.596
14 0.5179640718562875 0.463 0.592814371257485 0.922
14 0.877245508982036 0.506 0.24550898203592816 0.936
```
其中第一列是标签的index，总共有20个类别，最低为0，最高为19。图片读取什么的都很简单，这里我们看怎么上面的label和box提取出来，生成我们需要的$7\times 7\times 30$格式
```python
    def make_target(self, labels, boxes):
        '''
        labels = [1,14,14]
        boxes = [
            [0.4865 0.702 0.553 0.596]
            [0.5179 0.463 0.592 0.922]  
            [0.8772 0.506 0.245 0.936]
        ]      
        返回 shape为 (7,7,30) 的Tenser
        '''
        # 生成预测目标和预测分类
        np_target = np.zeros((7,7,30)) # 占个坑
        step = 1 / 7
        for i in range(len(boxes)):
            box = boxes[i]# box 坐标，中心坐标和宽高，这里都是相对于整张图的比例，我们需要的中心坐标的值为相对于当前cell的比例。
            label_index = labels[i]
            label = np.zeros((20)) # 初始化20个分类
            label[label_index] = 1.0 # 设置box的分类
            cx, cy, w, h = box
            # 获取中心点所在的格子,3.5 实际是第四个格子，但是0为第一个，所以索引为3
            bx = math.floor(cx / step) # 获取格子的横坐标
            by = math.floor(cy / step) # 获取格子的纵坐标
            cx = cx % step / step # 获取相对当期格子左上角的横坐标偏离比
            cy = cy % step / step
            box = [cx, cy, w, h] # 我们需要的坐标格式
            # 这里采取的方式，和上面的图片不一致，
            np_target[by][bx][:4] = box 
            np_target[by][bx][4] = 1
            np_target[by][bx][5:9] = box
            np_target[by][bx][9] = 1
            np_target[by][bx][10:] = label # 设置分类
        return np_target
```
### loss函数
loss函数是yolo中比较难的部分，下面给出实现。最好根据公式一一对应，其实分开函数实现更简单
```python
class YoloLoss(Module):
    def __init__(self, num_class=20):
        super(YoloLoss, self).__init__()
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.S = 7
        self.B = 2
        self.C = num_class

    def conver_box(self, box, index):
        x, y, w, h = box
        i, j = index
        step = 1 / self.S
        x = (x + j) * step
        y = (y + i) * step
        # x, y, w, h = x.item(), y.item(), w.item(), h.item()
        a, b, c, d = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
        return [max(a.item(), 0), max(b.item(), 0), w, h]

    def compute_iou(self, box1, box2, index):
        box1 = self.conver_box(box1, index)
        box2 = self.conver_box(box2, index)
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        # 获取相交
        inter_w = (w1 + w2) - (max(x1 + w1, x2 + w2) - min(x1, x2))
        inter_h = (h1 + h2) - (max(y1 + h1, y2 + h2) - min(y1, y2))

        if inter_h <= 0 or inter_w <= 0:  #代表相交区域面积为0
            return 0
        #往下进行应该inter 和 union都是正值
        inter = inter_w * inter_h
        union = w1 * h1 + w2 * h2 - inter
        return (inter / union).item()

    def forward(self, pred, target):
        batch_size = pred.size(0)
        # 获取有目标的格子的mask
        mask = target[:, :, :, 4] > 0
        # 获取GT有目标格子的数据，（box,conf,box,conf,class)
        target_cell = target[mask]
        # 获取pred有目标的格子的数据(box,conf,box,conf,class)
        pred_cell = pred[mask]
        # 开始计算loss
        arry = mask.cpu().numpy()
        indexs = np.argwhere(arry == True)
        '''
        获取mask中对应的index形式，[ 
        [0,2,3],
        [0,3,4],
        [1,2,3],
        [1,4,2]]
        其中第一列为格子对应第几张图片，因为我们一般会传入好几张图作为一个batch
        第二列为，该图中有目标的格子的纵坐标，第三列为横坐标
        '''
        for i in range(len(target_cell)):
            box = target_cell[i][:4] # 获取target的box，这里只用前四个就行，我们知道target的两个box是一样的
            index = indexs[i][1:] # 获取上面的index[2,3],不需要知道第几张图，因为这里pred_cell 和target_cell是一一对应的
            pbox1, pbox2 = pred_cell[i][:4], pred_cell[i][5:9]# 获取预测的两个框
            iou1, iou2 = self.compute_iou(box, pbox1, index), self.compute_iou(
                box, pbox2, index) #分别计算两个框和真实框的iou
            # 选择iou最大的那个框计算confidence loss， 这里我们要说明confidence loss为**预测框和真实框计算的iou和预测值的loss**
            if iou1 > iou2:
                target_cell[i][4] = iou1 
            else:
                target_cell[i][4] = iou2
                pred_cell[i][:4] = pbox2
                pred_cell[i][4] = pred_cell[i][9]

        noobj_mask = target[:, :, :, 4] == 0
        noobj_pred = pred[noobj_mask][:, :10].contiguous().view(-1, 5)
        noobj_target = target[noobj_mask][:, :10].contiguous().view(-1, 5)

        noobj_loss = F.mse_loss(noobj_target[:, 4],
                                noobj_pred[:, 4],
                                reduction='sum')
        obj_loss = F.mse_loss(pred_cell[:, 4],
                              target_cell[:, 4],
                              reduction='sum')
        xy_loss = F.mse_loss(pred_cell[:, :2],
                             target_cell[:, :2],
                             reduction='sum')
        wh_loss = F.mse_loss(pred_cell[:, 2:4],
                             target_cell[:, 2:4],
                             reduction='sum')

        class_loss = F.mse_loss(pred_cell[:, 10:],
                                target_cell[:, 10:],
                                reduction='sum')
        loss = [
            obj_loss, self.lambda_noobj * noobj_loss,
            self.lambda_coord * xy_loss, self.lambda_coord * wh_loss,
            class_loss
        ]
        loss = [ele / batch_size for ele in loss]
        return loss
```
### 训练部分
这里给出训练模板，包括可视化和学习率衰减。至于怎么衰减需要你手动调整。还有这个网络需要很长时间训练。训练了快100个epoch，只能做到训练数据拟合，验证集效果不太好。这个大概需要几百epoch训练，
```

from visdom import Visdom


def update_lr(optimizer, epoch):
    if epoch == 10:
        lr = 0.0007
    elif epoch == 25:
        lr = 0.0006
    elif epoch == 50:
        lr = 0.0005
    elif epoch == 60:
        lr = 0.0001
    else:
        return

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train():
    dom = Visdom()
    train_loader = DataLoader(VOCDataset(mode='train'),
                              batch_size=32,
                              num_workers=8,
                              drop_last=True,
                              pin_memory=True,
                              shuffle=True)
    valid_loader = DataLoader(VOCDataset(mode='val'),
                              batch_size=4,
                              num_workers=4,
                              drop_last=True,
                              shuffle=True)
    net = yolo().cuda()
    #net = torch.load('weights/95_net.pk')
    criterion = YoloLoss().cuda()
    optim = SGD(params=net.parameters(),
                lr=0.001,
                weight_decay=5e-4,
                momentum=0.9)
    #optim = Adam(params=net.parameters())
    t_obj_loss,t_nobj_loss,t_xy_loss,t_wh_loss,t_class_loss=[],[],[],[],[]
    v_obj_loss,v_nobj_loss,v_xy_loss,v_wh_loss,v_class_loss=[],[],[],[],[]
    valid_loss = []
    train_loss = []

    for epoch in range(0, 80):
        train_bar = tqdm(train_loader, dynamic_ncols=True)
        val_bar = tqdm(valid_loader, dynamic_ncols=True)
        train_bar.set_description_str(f"epoch/{epoch}")
        update_lr(optim, epoch)
        net.train()
        for i, ele in enumerate(train_bar):
            img, target = ele
            img, target = Variable(img).cuda(), Variable(target).cuda()
            output = net(img)
            optim.zero_grad()
            obj_loss, noobj_loss, xy_loss, wh_loss, class_loss = criterion(
                output, target.float())
            loss = obj_loss + noobj_loss + xy_loss + wh_loss + 2 * class_loss
            loss.backward()
            train_loss.append(loss.item())
            t_obj_loss.append(obj_loss.item())
            t_nobj_loss.append(noobj_loss.item())
            t_xy_loss.append(xy_loss.item())
            t_wh_loss.append(wh_loss.item())
            t_class_loss.append(class_loss.item())
            optim.step()
            if i % 10 == 0:
                loss_list = [
                    np.mean(x) for x in [
                        t_obj_loss, t_nobj_loss, t_xy_loss, t_wh_loss,
                        t_class_loss
                    ]
                ]
                train_bar.set_postfix_str(
                    "o:{:.2f} n:{:.2f} x:{:.2f} w:{:.2f} c:{:.2f}".format(
                        *loss_list))
                #train_bar.set_postfix_str(f"loss {np.mean(train_loss)}")
                dom.line(train_loss, win='train', opts={'title': 'Train loss'})
                dom.line(t_obj_loss, win='obj', opts={'title': 'obj'})
                dom.line(t_nobj_loss, win='noobj', opts={'title': 'noobj'})
                dom.line(t_xy_loss, win='xy', opts={'title': 'xy'})
                dom.line(t_wh_loss, win='wh', opts={'title': 'wh'})
                dom.line(t_class_loss, win='class', opts={'title': 'class'})
        if epoch % 5 == 0:
            torch.save(net, f'weights/{epoch}_net.pk')
        net.eval()
        with torch.no_grad():
            for i, ele in enumerate(val_bar):
                img, target = ele
                img, target = Variable(img).cuda(), Variable(target).cuda()
                output = net(img)
                obj_loss, noobj_loss, xy_loss, wh_loss, class_loss = criterion(
                    output, target.float())
                v_obj_loss.append(obj_loss.item())
                v_nobj_loss.append(noobj_loss.item())
                v_xy_loss.append(xy_loss.item())
                v_wh_loss.append(wh_loss.item())
                v_class_loss.append(class_loss.item())
                loss = obj_loss + noobj_loss + xy_loss + wh_loss + class_loss
                valid_loss.append(loss.item())
                if i % 10 == 0:
                    loss_list = [
                        np.mean(x) for x in [
                            v_obj_loss, v_nobj_loss, v_xy_loss, v_wh_loss,
                            v_class_loss
                        ]
                    ]
                    val_bar.set_postfix_str(
                        "o:{:.2f} n:{:.2f} x:{:.2f} w:{:.2f}c:{:.2f}".format(
                            *loss_list))
                    dom.line(valid_loss,
                             win='valid_loss',
                             opts=dict(title="Valid loss"))

    torch.save(net, f'weights/{epoch}_net.pk')


if __name__ == "__main__":
    train()

```
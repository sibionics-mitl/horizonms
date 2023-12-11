import torch


def encode_box(bbox, input_size, stride):
    xmin, ymin, xmax, ymax = bbox
    cx = (xmax + xmin) / 2
    cy = (ymax + ymin) / 2
    box_w = xmax - xmin
    box_h = ymax - ymin    

    cx_s = cx / stride
    cy_s = cy / stride
    grid_x = int(cx_s)
    grid_y = int(cy_s)
    
    tx = cx_s - grid_x
    ty = cy_s - grid_y
    tw = box_w/input_size[1]#torch.sqrt(box_w/input_size[1])
    th = box_h/input_size[0]#torch.sqrt(box_h/input_size[0])
    return [grid_x, grid_y], [tx, ty, tw, th]


def decode_box(txtytwth, input_size, grid, stride):
    xs = (txtytwth[0] + grid[0])*2*stride
    ys = (txtytwth[1] + grid[1])*2*stride
    w = input_size[1] * txtytwth[2]
    h = input_size[0] * txtytwth[3]
    xmax, xmin = (xs+w)/2, (xs-w)/2
    ymax, ymin = (ys+h)/2, (ys-h)/2
    return [xmin, ymin, xmax, ymax]


def create_grid(input_size, stride, device):
    hs, ws = input_size[0]//stride, input_size[1]//stride
    grid_y, grid_x = torch.meshgrid([torch.arange(hs, device=device),
                                     torch.arange(ws, device=device)])
    grid_xy = torch.stack([grid_x, grid_y], dim=0).float()
    return grid_xy


# def decode_box_batch(txtytwth, input_size, stride):
#     grid_xy = create_grid(input_size, stride, txtytwth.device)
#     xs = (txtytwth[:,0::4] + grid_xy[0])*2*stride
#     ys = (txtytwth[:,1::4] + grid_xy[1])*2*stride
#     w = input_size[1] * txtytwth[:,2::4]
#     h = input_size[0] * txtytwth[:,3::4]
#     xmax, xmin = (xs+w)/2, (xs-w)/2
#     ymax, ymin = (ys+h)/2, (ys-h)/2
#     return torch.stack([xmin, ymin, xmax, ymax], dim=-1)


def generate_target_yolov2(input_shape, stride, targets, num_boxes, num_classes):
    bbox_key = None
    for key, value in targets.items():
        if value.type == 'bboxes':
            bbox_key = key
    assert bbox_key is not None, "Annotations should contain bboxes."
    bboxes_list = targets[bbox_key].value
    dtype, device = bboxes_list[0].dtype, bboxes_list[0].device

    batch_size = len(bboxes_list)
    h, w = input_shape[0], input_shape[1]
    hs, ws = h // stride, w // stride
    gt_cls = torch.zeros((batch_size, num_classes, hs, ws), dtype=dtype, device=device)
    gt_objectness = torch.zeros((batch_size, num_boxes, hs, ws), dtype=dtype, device=device)
    gt_bboxes = torch.zeros((batch_size, 4*num_boxes, hs, ws), dtype=dtype, device=device)

    for batch_index, bboxes in enumerate(bboxes_list):
        for k in range(bboxes.shape[0]):
            bbox = bboxes[k,:]
            [grid_x, grid_y], gt_bbox = encode_box(bbox[:-1], input_shape, stride)
            if grid_x < ws and grid_y < hs:
                gt_cls[batch_index, int(bbox[-1]), grid_y, grid_x] = 1.0
                gt_objectness[batch_index, :, grid_y, grid_x] = 1.0
                gt_bboxes[batch_index, ::4, grid_y, grid_x] = gt_bbox[0]
                gt_bboxes[batch_index, 1::4, grid_y, grid_x] = gt_bbox[1]
                gt_bboxes[batch_index, 2::4, grid_y, grid_x] = gt_bbox[2]
                gt_bboxes[batch_index, 3::4, grid_y, grid_x] = gt_bbox[3]

    return gt_cls, gt_objectness, gt_bboxes


def generate_stride(feature_shape, input_shape):
    stride = input_shape[0]/feature_shape[0]
    assert input_shape[0]/feature_shape[0] == input_shape[1]/feature_shape[1]
    return stride


def generate_feature_shape(stride, input_shape):
    return (input_shape[0]//stride, input_shape[1]//stride)


def generate_target_yolov1(feature_shape, input_shape, targets, num_boxes, num_classes):
    bbox_key = None
    for key, value in targets.items():
        if value.type == 'bboxes':
            bbox_key = key
    assert bbox_key is not None, "Annotations should contain bboxes."
    bboxes_list = targets[bbox_key].value
    dtype, device = bboxes_list[0].dtype, bboxes_list[0].device

    batch_size = len(bboxes_list)
    hs, ws = feature_shape
    stride = input_shape[0]/feature_shape[0]
    assert input_shape[0]/feature_shape[0] == input_shape[1]/feature_shape[1]
    gt_cls = torch.zeros((batch_size, num_classes, hs, ws), dtype=dtype, device=device)
    gt_objectness = torch.zeros((batch_size, num_boxes, hs, ws), dtype=dtype, device=device)
    gt_bboxes = torch.zeros((batch_size, 4*num_boxes, hs, ws), dtype=dtype, device=device)

    for batch_index, bboxes in enumerate(bboxes_list):
        for k in range(bboxes.shape[0]):
            bbox = bboxes[k,:]
            [grid_x, grid_y], gt_bbox = encode_box(bbox[:-1], input_shape, stride)
            
            if grid_x < ws and grid_y < hs:
                gt_cls[batch_index, int(bbox[-1]), grid_y, grid_x] = 1.0
                gt_objectness[batch_index, :, grid_y, grid_x] = 1.0
                gt_bboxes[batch_index, ::4, grid_y, grid_x] = gt_bbox[0]
                gt_bboxes[batch_index, 1::4, grid_y, grid_x] = gt_bbox[1]
                gt_bboxes[batch_index, 2::4, grid_y, grid_x] = gt_bbox[2]
                gt_bboxes[batch_index, 3::4, grid_y, grid_x] = gt_bbox[3]

    return gt_cls, gt_objectness, gt_bboxes


# def encoder(bboxes_list, feature_shape):
#     batch_size = len(bboxes_list)
#     hs, ws = feature_shape
#     target = torch.zeros((batch_size, hs, ws, 30))
#     for batch_index, bboxes in enumerate(bboxes_list):
#         cell_size = [1./hs, 1./ws]
#         wh = bboxes[:,2:4] - bboxes[:,:2]
#         cxcy = (bboxes[:,2:4] + bboxes[:,:2])/2
#         labels = bboxes[:, -1]
#         for i in range(bboxes.shape[0]):
#             cxcy_sample = cxcy[i]
#             ij = (cxcy_sample/cell_size).ceil()-1 #
#             y, x = int(ij[1]), int(ij[0])
#             target[batch_index, y, x, 4] = 1
#             target[batch_index, y, x, 9] = 1
#             target[batch_index, y, x, int(labels[i])+9] = 1
#             xy = ij*cell_size #匹配到的网格的左上角相对坐标
#             delta_xy = (cxcy_sample -xy)/cell_size
#             target[batch_index, y, x, 2:4] = wh[i]
#             target[batch_index, y, x, :2] = delta_xy
#             target[batch_index, y, x, 7:9] = wh[i]
#             target[batch_index, y, x, 5:7] = delta_xy
#         return target


class BoxCoderYoloV1():
    def __init__(self, stride):
        self.stride = stride
    
    def decode(self, txtytwth, input_shape):
        grid_xy = create_grid(input_shape, self.stride, txtytwth.device)
        xs = (txtytwth[:,0::4] + grid_xy[0])*2*self.stride
        ys = (txtytwth[:,1::4] + grid_xy[1])*2*self.stride
        w = input_shape[1] * txtytwth[:,2::4]
        h = input_shape[0] * txtytwth[:,3::4]
        xmax, xmin = (xs+w)/2, (xs-w)/2
        ymax, ymin = (ys+h)/2, (ys-h)/2
        return torch.stack([xmin, ymin, xmax, ymax], dim=-1)
    
    def encode_single(self, box, input_shape):
        return encode_box(box, input_shape, self.stride)

    def decode_single(self, txtytwth, input_size, grid):
        return decode_box(txtytwth, input_size, grid, self.stride)